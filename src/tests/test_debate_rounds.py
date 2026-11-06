"""토론 서브그래프 테스트: 최대 3라운드, 조기 종료, A3 실패 (LLM은 가짜 응답)."""
import unittest
from unittest.mock import patch

from src import agent, debate, guards, llm
from src.schemas import A2Review, A3Opinion, Claim, Intake, QueryResponse, Verdict
from src.tests.fakes import default_value

FULL = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
              injury="minor", damage_level="moderate")
BASES = ["FT-01-M1", "FT-01-M2", "ART-13", "ART-12", "ART-14", "ART-10"]


class DebateLLM:
    """A3가 라운드마다 새 주장을 내거나(always_new) '추가 쟁점 없음'을 낸다. A2는 받은 주장을 모두 확인한다."""

    def __init__(self, always_new: bool, a3_fails: bool = False):
        self.always_new = always_new
        self.a3_fails = a3_fails
        self.a3_calls = 0
        self.review_targets: list[list[str]] = []

    def structured(self, schema, messages):
        if schema is Intake:
            return llm.LLMResult(value=FULL, model_id="fake")
        if schema is A3Opinion:
            if self.a3_fails:
                raise llm.LLMUnavailable([("m1", "ValidationError"), ("m2", "ValidationError")])
            self.a3_calls += 1
            if not self.always_new:
                return llm.LLMResult(value=A3Opinion(claims=[], no_new_issues=True), model_id="fake")
            # 라운드마다 근거가 다른 새 주장을 낸다 (no_new_issues=False)
            basis = BASES[self.a3_calls - 1]
            kind = "과실조정" if basis.startswith("FT") else "면책"
            c = Claim(kind=kind, coverage="공통", text=f"{self.a3_calls}번째 주장", candidate_basis_ids=[basis])
            return llm.LLMResult(value=A3Opinion(claims=[c], no_new_issues=False), model_id="fake")
        if schema is A2Review:
            text = str(messages[-1].content)
            ids = [line.split(" ")[1] for line in text.splitlines() if line.startswith("- A3-")]
            self.review_targets.append(ids)
            claims_basis = {i: b for i, b in zip(ids, BASES[len(sum(self.review_targets[:-1], [])):])}
            return llm.LLMResult(value=A2Review(verdicts=[
                Verdict(claim_id=i, verdict="확인", evidence_ids=[claims_basis[i]]) for i in ids]), model_id="fake")
        return llm.LLMResult(value=default_value(schema), model_id="fake")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="fake")


def run(fake: DebateLLM):
    agent.reset_memory()
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        return agent.run_query("어제 강남역에서 뒤차가 받았고 목이 아파요")


class DebateRoundsTest(unittest.TestCase):
    def test_always_new_issues_stops_at_three(self):
        fake = DebateLLM(always_new=True)
        resp = run(fake)
        QueryResponse.model_validate(resp)
        t = resp["trace"]
        self.assertEqual(t["rounds"], debate.MAX_ROUNDS)
        self.assertEqual(t["rounds"], 3)
        self.assertEqual(fake.a3_calls, 3)
        # 마지막 라운드 주장도 같은 라운드 안에서 판정을 받는다
        self.assertEqual([c["id"] for c in t["claims"]], ["A3-1-1", "A3-2-1", "A3-3-1"])
        self.assertTrue(all(c["verdict"] is not None for c in t["claims"]))
        self.assertEqual(t["claims"][-1]["round"], 3)
        # 라운드마다 새 주장만 검증한다
        self.assertEqual(fake.review_targets, [["A3-1-1"], ["A3-2-1"], ["A3-3-1"]])

    def test_debate_log_order(self):
        t = run(DebateLLM(always_new=True))["trace"]
        order = [(e["round"], e["agent"]) for e in t["debate_log"]]
        # 1라운드만 A1·A2 검색을 돌리고, 2라운드부터는 그 결과를 재사용한다
        expected = [(1, "A1"), (1, "A2-evidence"), (1, "A3"), (1, "A2-review"),
                    (2, "A3"), (2, "A2-review"), (3, "A3"), (3, "A2-review")]
        self.assertEqual(order, expected)

    def test_llm_calls_with_reuse(self):
        t = run(DebateLLM(always_new=True))["trace"]
        # 접수 1 + 1라운드 4 + 2·3라운드 각 2 + 답변 1 = 10
        self.assertEqual(len(t["models_used"]), 10)

    def test_no_new_issues_stops_after_one(self):
        fake = DebateLLM(always_new=False)
        t = run(fake)["trace"]
        self.assertEqual(t["rounds"], 1)
        self.assertEqual(fake.a3_calls, 1)
        self.assertEqual(t["claims"], [])
        self.assertEqual([e["agent"] for e in t["debate_log"]], ["A1", "A2-evidence", "A3"])  # 검증할 새 주장 없음

    def test_a3_failure_ends_debate_but_answers(self):
        resp = run(DebateLLM(always_new=True, a3_fails=True))
        t = resp["trace"]
        self.assertEqual(t["error"], "agent_failed:A3")
        self.assertEqual(t["rounds"], 1)
        self.assertEqual(t["claims"], [])
        self.assertTrue(resp["answer"].endswith(guards.DISCLAIMER))
        QueryResponse.model_validate(resp)

    def test_route_is_debate(self):
        self.assertEqual(run(DebateLLM(always_new=False))["trace"]["route"], "debate")


if __name__ == "__main__":
    unittest.main()
