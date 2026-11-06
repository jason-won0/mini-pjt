"""라우팅과 복합 요청 테스트 (LLM은 가짜 응답)."""
import unittest
from unittest.mock import patch

from src import agent, guards, llm
from src.schemas import Intake, QueryResponse
from src.tests.test_followup import SequenceLLM

FULL = dict(accident_type="후미추돌", occurred_at="어제", location="강남역", injury="none")


def ask(question: str, intake: Intake, thread_id=None):
    fake = SequenceLLM([intake])
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        return agent.run_query(question, thread_id=thread_id)


class RoutingFourCasesTest(unittest.TestCase):
    """확인 기준: 사고·조회·서류·복합 질문 4종의 intent·route·deferred_intents."""

    def setUp(self):
        agent.reset_memory()

    def check(self, resp, intent, route, deferred):
        QueryResponse.model_validate(resp)
        t = resp["trace"]
        self.assertEqual((t["intent"], t["route"], t["deferred_intents"]), (intent, route, deferred))

    def test_accident(self):
        r = ask("어제 강남역에서 뒤차가 받았어요. 다친 데는 없어요", Intake(intent="accident", **FULL))
        self.check(r, "accident", "debate", [])

    def test_claim_status(self):
        r = ask("CLM-20261104-001 진행 상황 알려줘", Intake(intent="claim_status", claim_no="CLM-20261104-001"))
        self.check(r, "claim_status", "direct", [])
        self.assertEqual(r["answer"], guards.CLAIM_NOT_FOUND)  # C001의 청구가 아니므로 (A4가 실제로 조회)
        self.assertNotIn(guards.DISCLAIMER, r["answer"])

    def test_documents(self):
        r = ask("필요한 서류가 뭐예요?", Intake(intent="documents"))
        self.check(r, "documents", "direct", [])

    def test_composite_accident_wins(self):
        intake = Intake(intent="accident", deferred_intents=["claim_status"], **FULL)
        r = ask("지난번 CLM-20261101-002 상태도 알려주고, 이번 사고 보험금도 계산해 줘", intake)
        self.check(r, "accident", "debate", ["claim_status"])
        self.assertIn("'청구 조회' 요청은 한 번에 하나씩", r["answer"])
        self.assertTrue(r["answer"].endswith(guards.DISCLAIMER))  # 안내 뒤에도 면책 문구가 맨 끝


class CodeRuleTest(unittest.TestCase):
    def setUp(self):
        agent.reset_memory()

    def test_code_rules(self):
        self.assertEqual(agent.code_intents("CLM-20261104-001 어떻게 됐어요"), ["claim_status"])
        self.assertEqual(agent.code_intents("청구 진행 상황 알려주세요"), ["claim_status"])
        self.assertEqual(agent.code_intents("준비물이 뭐가 있나요"), ["documents"])
        self.assertEqual(agent.code_intents("뒤에서 받혔어요"), [])

    def test_code_rule_overrides_wrong_llm_intent(self):
        # LLM이 서류 안내로 잘못 분류해도 청구번호 패턴이 있으면 조회가 우선한다
        r = ask("CLM-20261104-001 서류 냈는데 어떻게 됐나요", Intake(intent="documents"))
        t = r["trace"]
        self.assertEqual((t["intent"], t["route"], t["deferred_intents"]), ("claim_status", "direct", ["documents"]))

    def test_code_rule_rescues_refuse(self):
        # LLM이 거절로 분류해도 코드 규칙에 걸리면 처리한다
        r = ask("필요한 서류 알려줘", Intake(intent="refuse"))
        self.assertEqual(r["trace"]["route"], "direct")

    def test_refuse(self):
        r = ask("오늘 주식 뭐 사요?", Intake(intent="refuse"))
        self.assertEqual((r["trace"]["intent"], r["trace"]["route"]), ("refuse", "refused"))
        self.assertEqual(r["answer"], guards.REFUSAL_TEMP)

    def test_composite_with_followup(self):
        # 필수 항목이 비었으면 되묻기 경로에서도 미룬 요청을 안내한다
        intake = Intake(intent="accident", accident_type="후미추돌", deferred_intents=["documents"])
        r = ask("뒤에서 받혔는데 필요한 서류도 알려줘", intake)
        self.assertEqual(r["trace"]["route"], "followup")
        self.assertEqual(r["trace"]["deferred_intents"], ["documents"])
        self.assertIn("'서류 안내' 요청은", r["answer"])

    def test_decide_priority(self):
        intake = Intake(intent="documents", deferred_intents=["claim_status", "accident"])
        self.assertEqual(agent.decide_intents([], intake), ("accident", ["claim_status", "documents"]))


class SupervisorFailureTest(unittest.TestCase):
    def setUp(self):
        agent.reset_memory()

    def run_failing(self, attempts):
        def boom(*a, **k):
            raise llm.LLMUnavailable(attempts)
        with patch.object(agent, "call_structured", boom):
            return agent.run_query("뒤에서 받혔어요")

    def test_schema_failure_is_agent_failed(self):
        r = self.run_failing([("m1", "ValidationError"), ("m2", "OutputParserException")])
        self.assertEqual(r["trace"]["error"], "agent_failed:supervisor")
        self.assertEqual(r["answer"], guards.ERROR_MESSAGE)
        QueryResponse.model_validate(r)

    def test_network_failure_is_llm_unavailable(self):
        r = self.run_failing([("m1", "ThrottlingException"), ("m2", "ValidationError")])
        self.assertEqual(r["trace"]["error"], "llm_unavailable")


if __name__ == "__main__":
    unittest.main()
