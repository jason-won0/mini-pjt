"""토론 뒤 되묻기와 재토론 테스트 (LLM은 가짜 응답, 같은 thread_id로 멀티턴)."""
import unittest
from unittest.mock import patch

from src import agent, guards, llm
from src.schemas import A2Review, A3Opinion, Claim, Intake, QueryResponse, Verdict
from src.tests.fakes import default_value

FULL = dict(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
            injury="minor", damage_level="moderate")
SUDDEN = Claim(kind="과실조정", coverage="공통", text="고객이 급정거했다면 과실이 늘 수 있다", candidate_basis_ids=["FT-01-M1"])


class FollowupDebateLLM:
    """A3가 '급정거 여부'를 묻는다. 고객이 '급정거 안 함'이라고 답하면(facts) 급정거 주장을 기각한다."""

    def __init__(self, intakes, ask_info=True, a3_claims=None):
        self.intakes = list(intakes)
        self.ask_info = ask_info
        self.a3_claims = a3_claims if a3_claims is not None else [SUDDEN]
        self.a3_prompts: list[str] = []

    def structured(self, schema, messages):
        if schema is Intake:
            return llm.LLMResult(value=self.intakes.pop(0), model_id="f")
        if schema is A3Opinion:
            prompt = str(messages[-1].content)
            self.a3_prompts.append(prompt)
            first_round = "이미 낸 주장:\n없음" in prompt
            return llm.LLMResult(value=A3Opinion(
                claims=self.a3_claims if first_round else [], no_new_issues=not first_round,
                customer_info_needed=["사고 직전 급정거 여부"] if (self.ask_info and first_round) else []), model_id="f")
        if schema is A2Review:
            stopped = "급정거하지 않았음" in str(messages[-1].content)
            ids = [l.split(" ")[1] for l in str(messages[-1].content).splitlines() if l.startswith("- A3-")]
            return llm.LLMResult(value=A2Review(verdicts=[
                Verdict(claim_id=i, verdict="기각" if stopped else "확인", evidence_ids=["FT-01-M1"]) for i in ids]),
                model_id="f")
        return llm.LLMResult(value=default_value(schema), model_id="f")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="f")


def converse(fake, questions, thread_id="t-dfu"):
    agent.reset_memory()
    out = []
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        for q in questions:
            out.append(agent.run_query(q, thread_id=thread_id))
    return out


class DebateFollowupTest(unittest.TestCase):
    def test_1_info_needed_asks_once_then_re_debates(self):
        fake = FollowupDebateLLM([Intake(**FULL), Intake(intent="accident", facts=["급정거하지 않았음"])])
        r1, r2 = converse(fake, ["어제 강남역에서 뒤차가 받았고 목이 아파요", "급정거는 안 했어요"])
        t1 = r1["trace"]
        self.assertEqual((t1["route"], t1["followup_reason"]), ("followup", "debate"))
        self.assertIn("- 사고 직전 급정거 여부", r1["answer"])
        self.assertNotIn(guards.DISCLAIMER, r1["answer"])
        t2 = r2["trace"]
        self.assertEqual(t2["route"], "debate")  # 재토론 후 최종 답변
        self.assertTrue(r2["answer"].endswith(guards.DISCLAIMER))
        # 재토론에서 A3가 고객이 답한 사실을 보고, A2가 급정거 주장을 기각한다
        self.assertIn("급정거하지 않았음", fake.a3_prompts[-1])
        self.assertEqual([c["verdict"] for c in t2["claims"]], ["기각"])
        self.assertIn("고객 과실: 0~10%", r2["answer"])  # 기각돼서 과실 조정 없음
        QueryResponse.model_validate(r2)

    def test_2_intake_followup_used_then_no_debate_followup(self):
        partial = Intake(intent="accident", accident_type="후미추돌")  # 필수 항목 누락 → 접수 되묻기
        fake = FollowupDebateLLM([partial, Intake(intent="accident", occurred_at="어제", location="강남역",
                                                  injury="minor", damage_level="moderate")])
        r1, r2 = converse(fake, ["뒤에서 받혔어요", "어제 강남역이고 목이 아파요"])
        self.assertEqual(r1["trace"]["followup_reason"], "intake")
        t2 = r2["trace"]
        self.assertEqual(t2["route"], "debate")  # 토론에서 정보가 필요해도 다시 묻지 않는다
        self.assertIn("추가로 확인이 필요한 정보: 사고 직전 급정거 여부", r2["answer"])
        self.assertTrue(r2["answer"].endswith(guards.DISCLAIMER))

    def test_3_debate_followup_used_then_not_asked_again(self):
        # 되묻기에 답했지만 여전히 정보가 필요하다고 나와도 다시 묻지 않는다
        fake = FollowupDebateLLM([Intake(**FULL), Intake(intent="accident")])
        fake.ask_info = True
        r1, r2 = converse(fake, ["어제 강남역에서 뒤차가 받았고 목이 아파요", "잘 기억이 안 나요"])
        self.assertEqual(r1["trace"]["route"], "followup")
        self.assertEqual(r2["trace"]["route"], "debate")
        self.assertEqual(r2["trace"]["followup_reason"], "debate")

    def test_4_unresolved_only_is_not_asked(self):
        bad = Claim(kind="감액", coverage="대인", text="근거 없는 감액 주장", candidate_basis_ids=["ART-99"])
        fake = FollowupDebateLLM([Intake(**FULL)], ask_info=False, a3_claims=[bad])
        [r1] = converse(fake, ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        self.assertEqual(r1["trace"]["route"], "debate")
        self.assertFalse(r1["trace"]["followup_used"])
        self.assertIn("확인되지 않은 사항: 근거 없는 감액 주장", r1["answer"])

    def test_facts_are_merged(self):
        prev = Intake(intent="accident", facts=["안전띠를 맸음"])
        merged = agent.merge_intake(prev, Intake(intent="accident", facts=["급정거하지 않았음", "안전띠를 맸음"]), True)
        self.assertEqual(merged.facts, ["안전띠를 맸음", "급정거하지 않았음"])


if __name__ == "__main__":
    unittest.main()
