"""접수 되묻기 1회와 조건부 안내 테스트 (같은 thread_id로 멀티턴, LLM은 가짜 응답)."""
import unittest
from unittest.mock import patch

from src import agent, guards, llm
from src.schemas import A1Opinion, A2Evidence, Intake, QueryResponse
from src.tests.fakes import default_value


class SequenceLLM:
    """Intake 호출마다 준비한 접수 정보를 차례로 돌려준다."""

    def __init__(self, intakes: list[Intake]):
        self.intakes = list(intakes)
        self.intake_calls = 0

    def structured(self, schema, messages):
        if schema is Intake:
            self.intake_calls += 1
            return llm.LLMResult(value=self.intakes.pop(0), model_id="fake")
        return llm.LLMResult(value=default_value(schema), model_id="fake")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="fake")


def converse(fake: SequenceLLM, questions: list[str], thread_id="t-test"):
    outs = []
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        for q in questions:
            outs.append(agent.run_query(q, thread_id=thread_id))
    return outs


ONLY_TYPE = Intake(intent="accident", accident_type="후미추돌")
ANSWER_ALL = Intake(intent="accident", occurred_at="어제 오후 3시", location="강남역 사거리", injury="minor")
ANSWER_NOTHING = Intake(intent="accident")


class FollowupTest(unittest.TestCase):
    def setUp(self):
        agent.reset_memory()

    def test_first_turn_asks_once(self):
        [r1] = converse(SequenceLLM([ONLY_TYPE]), ["신호 대기 중에 뒤에서 차가 들이받았어요"])
        QueryResponse.model_validate(r1)
        t = r1["trace"]
        self.assertEqual(t["route"], "followup")
        self.assertEqual(t["followup_reason"], "intake")
        self.assertTrue(t["followup_used"])
        self.assertEqual(t["missing_fields"], ["occurred_at", "location", "injury"])
        for line in ["- 사고 일시: 언제 사고가 났는지", "- 사고 장소: 어디서 사고가 났는지", "- 부상 여부: 다치신 곳이 있는지"]:
            self.assertIn(line, r1["answer"])
        self.assertNotIn("사고 유형", r1["answer"])  # 이미 받은 항목은 다시 묻지 않는다
        self.assertNotIn(guards.DISCLAIMER, r1["answer"])  # 추정이 없으므로 면책 문구도 없다
        self.assertEqual(r1["contexts"], [])

    def test_second_turn_gives_final_answer(self):
        r1, r2 = converse(SequenceLLM([ONLY_TYPE, ANSWER_ALL]),
                          ["신호 대기 중에 뒤에서 차가 들이받았어요", "어제 오후 3시 강남역 사거리요. 목이 좀 아파요"])
        self.assertEqual(r1["trace"]["route"], "followup")
        t = r2["trace"]
        self.assertEqual(t["route"], "debate")
        self.assertEqual(t["missing_fields"], [])
        self.assertTrue(t["followup_used"])
        self.assertEqual(t["followup_reason"], "intake")
        self.assertIn("고객 과실: 0~10%", r2["answer"])
        self.assertTrue(r2["answer"].endswith(guards.DISCLAIMER))
        self.assertNotIn("확인되지 않은 항목", r2["answer"])
        self.assertEqual(t["thread_id"], "t-test")

    def test_no_second_followup_conditional_answer(self):
        fake = SequenceLLM([ONLY_TYPE, ANSWER_NOTHING])
        r1, r2 = converse(fake, ["뒤에서 받혔어요", "잘 모르겠어요"])
        t = r2["trace"]
        self.assertEqual(t["route"], "debate")  # 다시 묻지 않는다
        self.assertEqual(t["missing_fields"], ["occurred_at", "location", "injury"])
        self.assertIn("확인되지 않은 항목: 사고 일시, 사고 장소, 부상 여부", r2["answer"])
        self.assertIn("대인(상대방 보험에서 받는 치료 관련 보상, 가상 정액 기준): 부상이 있다면", r2["answer"])
        self.assertTrue(r2["answer"].endswith(guards.DISCLAIMER))

    def test_partial_answer_keeps_previous_values(self):
        partial = Intake(intent="accident", location="강남역")
        _, r2 = converse(SequenceLLM([ONLY_TYPE, partial]), ["뒤에서 받혔어요", "강남역이요"])
        # 사고 유형(1턴)과 장소(2턴)가 합쳐지고, 나머지만 남는다
        self.assertEqual(r2["trace"]["missing_fields"], ["occurred_at", "injury"])
        self.assertIn("FT-01", r2["answer"])

    def test_threads_are_independent(self):
        fake = SequenceLLM([ONLY_TYPE, ONLY_TYPE])
        with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
            a = agent.run_query("뒤에서 받혔어요", thread_id="t-a")
            b = agent.run_query("뒤에서 받혔어요", thread_id="t-b")
        self.assertEqual(a["trace"]["route"], "followup")
        self.assertEqual(b["trace"]["route"], "followup")  # 다른 대화는 따로 센다

    def test_full_first_turn_skips_followup(self):
        full = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남", injury="none")
        [r1] = converse(SequenceLLM([full]), ["어제 강남에서 뒤에서 받혔고 다친 데는 없어요"])
        self.assertEqual(r1["trace"]["route"], "debate")
        self.assertFalse(r1["trace"]["followup_used"])


class MergeIntakeTest(unittest.TestCase):
    def test_new_values_overwrite_only_when_present(self):
        prev = Intake(intent="accident", accident_type="후미추돌", location="강남")
        new = Intake(intent="documents", location=None, injury="none")
        merged = agent.merge_intake(prev, new, answering_followup=True)
        self.assertEqual(merged.accident_type, "후미추돌")
        self.assertEqual(merged.location, "강남")
        self.assertEqual(merged.injury, "none")
        self.assertEqual(merged.intent, "accident")  # 되묻기에 답하는 중이면 요청 종류 유지

    def test_intent_changes_when_not_answering(self):
        prev = Intake(intent="accident", accident_type="후미추돌")
        merged = agent.merge_intake(prev, Intake(intent="documents"), answering_followup=False)
        self.assertEqual(merged.intent, "documents")


if __name__ == "__main__":
    unittest.main()
