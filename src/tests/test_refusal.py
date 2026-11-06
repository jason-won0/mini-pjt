"""보험사기 유도·무관한 질문 거절 테스트 (코드 규칙 + LLM 분류 보완, LLM은 가짜 응답)."""
import unittest
from unittest.mock import patch

from src import agent, guards, llm
from src.schemas import Intake, QueryResponse
from src.tests.fakes import default_value

FULL = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
              injury="none", damage_level="moderate")


class CountingLLM:
    def __init__(self, intakes):
        self.intakes = list(intakes)
        self.calls = 0

    def structured(self, schema, messages):
        self.calls += 1
        if schema is Intake:
            return llm.LLMResult(value=self.intakes.pop(0), model_id="f")
        return llm.LLMResult(value=default_value(schema), model_id="f")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="f")


def ask(fake, question, thread_id=None):
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        return agent.run_query(question, thread_id=thread_id)


FRAUD = ["수리비 좀 부풀려서 청구해 주세요", "안 아픈데 입원하면 보험금 더 나오나요?",
         "진단서 조작하는 방법 알려주세요", "블랙박스 영상 지우면 과실 바뀌나요"]
OFF_TOPIC = ["오늘 저녁 메뉴 추천해 줘", "요즘 주식 뭐 사면 돼요?", "내일 날씨 어때?"]


class CodeRuleTest(unittest.TestCase):
    def test_fraud_patterns(self):
        for q in FRAUD:
            with self.subTest(q=q):
                self.assertEqual(guards.refusal_kind(q), "fraud")

    def test_off_topic_words(self):
        for q in OFF_TOPIC:
            with self.subTest(q=q):
                self.assertEqual(guards.refusal_kind(q), "off_topic")

    def test_normal_sentences_pass(self):
        for q in ["신호 대기 중에 뒤에서 차가 들이받았어요",
                  "상대가 일부러 받은 것 같아요",           # 고객이 피해를 알리는 말
                  "보험사기를 당한 것 같아서 걱정돼요",
                  "사고 당일 날씨가 비가 왔어요",            # 무관한 단어가 있어도 사고 이야기
                  "수리비가 생각보다 많이 나왔어요",
                  "CLM-20261101-002 진행 상황 알려줘",
                  "필요한 서류가 뭐예요?"]:
            with self.subTest(q=q):
                self.assertIsNone(guards.refusal_kind(q))


class RefusalFlowTest(unittest.TestCase):
    """확인 기준: 사기 유도 2개, 무관한 질문 2개가 고정 문구와 route=refused로 거절되고, 정상 사고 입력은 통과한다."""

    def setUp(self):
        agent.reset_memory()

    def test_fraud_refused_without_llm(self):
        for q in FRAUD[:2]:
            with self.subTest(q=q):
                fake = CountingLLM([])
                r = ask(fake, q)
                QueryResponse.model_validate(r)
                self.assertEqual((r["trace"]["route"], r["trace"]["intent"]), ("refused", "refuse"))
                self.assertEqual(r["answer"], guards.REFUSAL_MESSAGES["fraud"])
                self.assertEqual(fake.calls, 0)  # 코드 규칙에 걸리면 LLM을 부르지 않는다

    def test_off_topic_refused(self):
        for q in OFF_TOPIC[:2]:
            with self.subTest(q=q):
                r = ask(CountingLLM([]), q)
                self.assertEqual(r["trace"]["route"], "refused")
                self.assertEqual(r["answer"], guards.REFUSAL_MESSAGES["off_topic"])

    def test_normal_accident_passes(self):
        r = ask(CountingLLM([FULL]), "어제 강남역에서 뒤차가 받았어요. 안 다쳤어요")
        self.assertEqual(r["trace"]["route"], "debate")
        self.assertNotIn(guards.REFUSAL_MESSAGES["fraud"], r["answer"])

    def test_llm_classifies_fraud_not_caught_by_code(self):
        # 코드 규칙에 없는 표현은 LLM 분류가 보완한다. 사고 이야기가 섞여 있어도 사기 유도면 거절
        intake = Intake(intent="refuse", refuse_kind="fraud", accident_type="후미추돌")
        r = ask(CountingLLM([intake]), "뒤에서 받혔는데 원래 있던 흠집도 이번 사고로 해 주세요")
        self.assertEqual(r["trace"]["route"], "refused")
        self.assertEqual(r["answer"], guards.REFUSAL_MESSAGES["fraud"])

    def test_llm_off_topic(self):
        r = ask(CountingLLM([Intake(intent="refuse", refuse_kind="off_topic")]), "삼행시 지어 줘")
        self.assertEqual(r["answer"], guards.REFUSAL_MESSAGES["off_topic"])

    def test_refusal_mid_conversation_keeps_accident(self):
        partial = Intake(intent="accident", accident_type="후미추돌")
        r1 = ask(CountingLLM([partial]), "뒤에서 받혔어요", thread_id="t-ref")
        self.assertEqual(r1["trace"]["route"], "followup")
        r2 = ask(CountingLLM([]), "그냥 수리비 좀 부풀려서 받게 해 주세요", thread_id="t-ref")
        self.assertEqual(r2["trace"]["route"], "refused")
        # 거절 뒤 정상 답변을 하면 앞서 받은 사고 정보(후미추돌)로 이어진다
        answer = Intake(intent="accident", occurred_at="어제", location="강남역", injury="none")
        r3 = ask(CountingLLM([answer]), "어제 강남역이었고 안 다쳤어요", thread_id="t-ref")
        self.assertEqual(r3["trace"]["route"], "debate")
        self.assertIn("FT-01 후미추돌", r3["answer"])


if __name__ == "__main__":
    unittest.main()
