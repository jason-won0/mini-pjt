"""A4 노드 (MCP 클라이언트) 테스트. LLM은 가짜 응답, 청구 저장소는 테스트용 임시 파일."""
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from src import agent, guards, llm, tools
from src.schemas import Intake, QueryResponse
from src.tests import REAL_CALL_MCP, TEST_CLAIMS
from src.tests.fakes import default_value

FULL = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
              injury="minor", damage_level="moderate")


class FakeLLM:
    def __init__(self, intakes):
        self.intakes = list(intakes)

    def structured(self, schema, messages):
        if schema is Intake:
            return llm.LLMResult(value=self.intakes.pop(0), model_id="f")
        return llm.LLMResult(value=default_value(schema), model_id="f")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="f")


def ask(fake, questions, thread_id=None, customer_id="C001"):
    agent.reset_memory()
    out = []
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        for q in questions:
            out.append(agent.run_query(q, customer_id=customer_id, thread_id=thread_id))
            thread_id = thread_id or out[-1]["trace"]["thread_id"]
    return out


class A4AfterDebateTest(unittest.TestCase):
    def test_accident_issues_claim_and_documents(self):
        [r] = ask(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        QueryResponse.model_validate(r)
        t, a = r["trace"], r["answer"]
        self.assertRegex(t["claim_no"], r"^CLM-\d{8}-\d{3}$")
        self.assertIn(f"청구번호: {t['claim_no']} (상태: 접수)", a)
        self.assertIn("[필요 서류]\n- 사고 경위서", a)
        self.assertIn("- 진단서", a)  # 부상 있음
        self.assertTrue(a.endswith(guards.DISCLAIMER))  # 면책 문구는 여전히 맨 끝
        self.assertEqual(tools.get_claim(t["claim_no"], "C001").status, "접수")

    def test_same_thread_keeps_one_claim(self):
        r1, r2 = ask(FakeLLM([FULL, FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요", "다시 한 번 정리해 주세요"])
        self.assertEqual(r1["trace"]["claim_no"], r2["trace"]["claim_no"])

    def test_injury_unknown_documents_are_conditional(self):
        intake = FULL.model_copy(update={"injury": None})
        r = ask(FakeLLM([intake, Intake(intent="accident")]), ["어제 강남역에서 뒤차가 받았어요", "모르겠어요"])[-1]
        self.assertIn("- 부상이 있다면 추가로: 진단서", r["answer"])

    def test_followup_turn_does_not_save(self):
        r = ask(FakeLLM([Intake(intent="accident", accident_type="후미추돌")]), ["뒤에서 받혔어요"])[0]
        self.assertEqual(r["trace"]["route"], "followup")
        self.assertIsNone(r["trace"]["claim_no"])

    def test_store_failure(self):
        with patch.object(tools, "save_claim", side_effect=tools.StoreError("disk")):
            [r] = ask(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        self.assertEqual(r["trace"]["error"], "store_failed")
        self.assertIsNone(r["trace"]["claim_no"])
        self.assertIn(guards.RETRY_LATER, r["answer"])
        self.assertTrue(r["answer"].endswith(guards.DISCLAIMER))


class A4DirectTest(unittest.TestCase):
    def test_new_thread_status_lookup(self):
        [r1] = ask(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        claim_no = r1["trace"]["claim_no"]
        # 새 대화(새 thread_id)에서 그 번호로 조회 → 토론 없이 상태가 나온다
        [r2] = ask(FakeLLM([Intake(intent="claim_status", claim_no=claim_no)]), [f"{claim_no} 진행 상황 알려줘"])
        t = r2["trace"]
        self.assertNotEqual(t["thread_id"], r1["trace"]["thread_id"])
        self.assertEqual((t["route"], t["rounds"], t["claim_no"]), ("direct", 0, claim_no))
        self.assertIn("진행 상태는 **접수**", r2["answer"])
        self.assertNotIn(guards.DISCLAIMER, r2["answer"])  # 추정이 없으므로

    def test_sample_and_other_customer(self):
        [r] = ask(FakeLLM([Intake(intent="claim_status")]), ["CLM-20261101-002 어떻게 됐어요"])
        self.assertIn("**심사중**", r["answer"])
        [r] = ask(FakeLLM([Intake(intent="claim_status")]), ["CLM-20261101-002 어떻게 됐어요"], customer_id="C003")
        self.assertEqual(r["answer"], guards.CLAIM_NOT_FOUND)
        self.assertIsNone(r["trace"]["claim_no"])

    def test_status_without_number_asks(self):
        [r] = ask(FakeLLM([Intake(intent="claim_status")]), ["제 청구 진행 상황 알려주세요"])
        self.assertEqual(r["answer"], guards.ASK_CLAIM_NO)

    def test_documents_general_and_after_accident(self):
        [r] = ask(FakeLLM([Intake(intent="documents")]), ["필요한 서류가 뭐예요?"])
        self.assertIn("일반적으로 필요한 서류", r["answer"])
        self.assertIn("- 부상이 있다면 추가로: 진단서", r["answer"])
        r1, r2 = ask(FakeLLM([FULL, Intake(intent="documents")]),
                     ["어제 강남역에서 뒤차가 받았고 목이 아파요", "필요한 서류가 뭐예요?"])
        self.assertIn("말씀하신 사고(후미추돌) 기준", r2["answer"])
        self.assertIn("- 진단서", r2["answer"])


class A4RealMCPTest(unittest.TestCase):
    """실제 MCP 경로: stdio로 mcp_server.py를 띄운다 (네트워크 없음)."""

    def setUp(self):
        self._env = patch.dict(os.environ, {"MINI_PJT_CLAIMS_PATH": str(TEST_CLAIMS)})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_accident_through_real_mcp(self):
        with patch.object(agent, "call_mcp", REAL_CALL_MCP):
            [r] = ask(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        self.assertRegex(r["trace"]["claim_no"], r"^CLM-\d{8}-\d{3}$")
        self.assertIsNone(r["trace"]["error"])
        self.assertEqual(tools.get_claim(r["trace"]["claim_no"], "C001").status, "접수")  # 임시 저장소에 저장됨

    def test_wrong_server_path_is_mcp_failed(self):
        with patch.object(agent, "call_mcp", REAL_CALL_MCP), \
                patch.object(agent, "MCP_SERVER", Path("C:/nowhere/mcp_server.py")):
            [r] = ask(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        self.assertEqual(r["trace"]["error"], "mcp_failed")
        self.assertIsNone(r["trace"]["claim_no"])
        self.assertIn(guards.RETRY_LATER, r["answer"])
        self.assertTrue(r["answer"].endswith(guards.DISCLAIMER))
        QueryResponse.model_validate(r)

    def test_a4_does_not_import_claim_functions(self):
        # CLAUDE.md: A4는 tools.py의 청구 함수를 직접 부르지 않는다 (MCP로만)
        src = Path(agent.__file__).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"tools\.(save_claim|get_claim|required_documents)\(", src))


if __name__ == "__main__":
    unittest.main()
