"""Streamlit 화면 테스트 (streamlit.testing AppTest, LLM은 가짜 응답)."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src import agent, guards, llm
from src.schemas import Intake
from src.tests.fakes import default_value

APP = str(Path(__file__).resolve().parents[1] / "app.py")
os.environ["MINI_PJT_TYPE_DELAY"] = "0"  # 화면 테스트에서는 타자 효과를 끈다
FULL = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
              injury="minor", damage_level="moderate")
ONLY_TYPE = Intake(intent="accident", accident_type="후미추돌")


class FakeLLM:
    def __init__(self, intakes):
        self.intakes = list(intakes)
        self.customer_ids = []

    def structured(self, schema, messages):
        if schema is Intake:
            return llm.LLMResult(value=self.intakes.pop(0), model_id="f")
        return llm.LLMResult(value=default_value(schema), model_id="f")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="f")


def chat_texts(at: AppTest) -> list[str]:
    """메시지마다 마지막 마크다운(답변 본문). 토론이 있으면 그 위에 토론 과정이 펼쳐져 있다."""
    return [m.markdown[-1].value for m in at.chat_message]


def message_body(at: AppTest, index: int = 1) -> str:
    """메시지 하나의 마크다운 전부 (토론 과정 + 주장 판정 + 답변)."""
    return "\n".join(m.value for m in at.chat_message[index].markdown)


class AppTestCase(unittest.TestCase):
    def setUp(self):
        agent.reset_memory()

    def run_app(self, fake, inputs, customer_id=None):
        with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
            at = AppTest.from_file(APP, default_timeout=60).run()
            if customer_id:
                at.selectbox(key="customer_id").set_value(customer_id).run()
            for text in inputs:
                at.chat_input[0].set_value(text).run()
        return at

    def test_screen_shows_customers_and_answer(self):
        at = self.run_app(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="customer_id").options, ["김가상 (C001) · 종합 가입 (대인·대물·자차)",
                                                                   "이예시 (C002) · 의무보험만 가입 (대인·대물 최소 한도, 자차 없음)",
                                                                   "박샘플 (C003) · 자차 없음 + 대차비용 지원 특약"])
        texts = chat_texts(at)
        self.assertEqual(len(texts), 2)  # 사용자 입력 + 답변
        self.assertIn("[추정 결과]", texts[1])
        self.assertIn(guards.DISCLAIMER, texts[1].replace("\\~", "~"))

    def test_tilde_is_escaped_not_strikethrough(self):
        at = self.run_app(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"])
        answer = chat_texts(at)[1]
        self.assertIn(r"0\~10%", answer)  # 취소선으로 그려지지 않도록 이스케이프

    def test_followup_continues_same_thread(self):
        at = self.run_app(FakeLLM([ONLY_TYPE, FULL]), ["뒤에서 받혔어요", "어제 강남역이고 목이 아파요"])
        texts = chat_texts(at)
        self.assertEqual(len(texts), 4)
        self.assertIn("사고 일시", texts[1])  # 1턴: 되묻기
        self.assertIn("[추정 결과]", texts[3])  # 2턴: 같은 대화로 이어져 최종 답변
        self.assertTrue(at.session_state["thread_id"].startswith("t-"))

    def test_selected_customer_is_used(self):
        at = self.run_app(FakeLLM([FULL]), ["어제 강남역에서 뒤차가 받았고 목이 아파요"], customer_id="C002")
        self.assertIn("자차 미가입", chat_texts(at)[1])


class T52Test(unittest.TestCase):
    """T5-2: 새 대화 버튼, 고객 변경 초기화, 토론 접기 영역."""

    def setUp(self):
        agent.reset_memory()

    def app_with_one_answer(self, fake, customer_id="C003"):
        self._patches = [patch.object(agent, "call_structured", fake.structured),
                         patch.object(agent, "call_text", fake.text)]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.selectbox(key="customer_id").set_value(customer_id).run()
        at.chat_input[0].set_value("어제 강남역에서 뒤차가 받았고 목이 아파요").run()
        return at

    def test_new_chat_clears_conversation_keeps_customer(self):
        at = self.app_with_one_answer(FakeLLM([FULL]))
        self.assertEqual(len(at.chat_message), 2)
        old_thread = at.session_state["thread_id"]
        self.assertIsNotNone(old_thread)
        at.button(key="new_chat").click().run()  # 새 대화
        self.assertEqual(len(at.chat_message), 0)
        self.assertIsNone(at.session_state["thread_id"])
        self.assertEqual(at.selectbox(key="customer_id").value, "C003")  # 고객 유지

    def test_new_chat_starts_new_thread(self):
        at = self.app_with_one_answer(FakeLLM([FULL, FULL]))
        old_thread = at.session_state["thread_id"]
        at.button(key="new_chat").click().run()
        at.chat_input[0].set_value("다른 사고예요. 어제 강남역에서 뒤차가 받았어요").run()
        self.assertNotEqual(at.session_state["thread_id"], old_thread)

    def test_changing_customer_resets_conversation(self):
        at = self.app_with_one_answer(FakeLLM([FULL]), customer_id="C001")
        self.assertEqual(len(at.chat_message), 2)
        at.selectbox(key="customer_id").set_value("C002").run()
        self.assertEqual(len(at.chat_message), 0)
        self.assertIsNone(at.session_state["thread_id"])

    def test_debate_shown_inline_with_rounds_and_verdicts(self):
        from src.schemas import A2Review, A3Opinion, Claim, Verdict
        claim = Claim(kind="면책", coverage="대물", text="사고와 무관한 수리는 보상되지 않는다", candidate_basis_ids=["ART-13"])

        class DebateFake(FakeLLM):
            def structured(self, schema, messages):
                if schema is A3Opinion:
                    first = "이미 낸 주장:\n없음" in str(messages[-1].content)
                    return llm.LLMResult(value=A3Opinion(claims=[claim] if first else [], no_new_issues=not first),
                                         model_id="f")
                if schema is A2Review:
                    return llm.LLMResult(value=A2Review(verdicts=[
                        Verdict(claim_id="A3-1-1", verdict="확인", evidence_ids=["ART-13"])]), model_id="f")
                return super().structured(schema, messages)

        at = self.app_with_one_answer(DebateFake([FULL]))
        self.assertEqual(len(at.expander), 0)  # 접는 영역이 없다
        body = message_body(at)
        for text in ["**🗣️ 에이전트 토론**", "**1라운드**", "**2라운드**", "A1 보장·과실·보험금 추정", "A2 약관·법률 자문가 (근거 검색)",
                     "A3 기업 CEO (비용 절감 반론)", "A2 약관·법률 자문가 (반론 검증)", "**확인** (근거 ART-13, 반영)"]:
            self.assertIn(text, body)
        # 2라운드는 A3(와 새 주장이 있을 때 A2 검증)만 있다
        round2 = body.split("**2라운드**")[1].split("**A3 주장별 판정**")[0]
        self.assertNotIn("A1 보장", round2)
        # 순서: 토론 → 주장 판정 → 답변
        self.assertLess(body.index("**1라운드**"), body.index("**A3 주장별 판정**"))
        self.assertLess(body.index("**A3 주장별 판정**"), body.index("[추정 결과]"))

    def test_rejected_claim_is_struck_through(self):
        from src.schemas import A2Review, A3Opinion, Claim, Verdict
        claim = Claim(kind="면책", coverage="대물", text="사고와 무관한 수리는 보상되지 않는다", candidate_basis_ids=["ART-13"])

        class RejectFake(FakeLLM):
            def structured(self, schema, messages):
                if schema is A3Opinion:
                    first = "이미 낸 주장:\n없음" in str(messages[-1].content)
                    return llm.LLMResult(value=A3Opinion(claims=[claim] if first else [], no_new_issues=not first),
                                         model_id="f")
                if schema is A2Review:
                    return llm.LLMResult(value=A2Review(verdicts=[
                        Verdict(claim_id="A3-1-1", verdict="기각", evidence_ids=["ART-13"])]), model_id="f")
                return super().structured(schema, messages)

        at = self.app_with_one_answer(RejectFake([FULL]))
        body = message_body(at)
        self.assertIn("- ❌ ~~`A3-1-1` 면책/대물 — 기각 (근거 ART-13): 사고와 무관한 수리는 보상되지 않는다~~ _(미반영)_", body)

    def test_live_output_stays_without_folding(self):
        at = self.app_with_one_answer(FakeLLM([FULL]))
        self.assertEqual(len(at.status), 0)
        self.assertEqual(len(at.expander), 0)
        self.assertNotIn("⏳", "\n".join(c.value for c in at.caption))  # 진행 중 한 줄은 끝나면 지운다
        body = message_body(at)
        self.assertIn("**🗣️ 에이전트 토론**", body)  # 실시간으로 쓴 토론이 그대로 남는다
        self.assertIn("[추정 결과]", body)

    def test_followup_message_has_no_debate(self):
        at = self.app_with_one_answer(FakeLLM([ONLY_TYPE]))
        self.assertNotIn("에이전트 토론", message_body(at))  # 되묻기는 토론 전이다

    def test_debate_survives_rerun(self):
        at = self.app_with_one_answer(FakeLLM([FULL]))
        before = message_body(at)
        at.run()  # 화면을 다시 그려도 지난 답변의 토론 과정이 같은 모양으로 남는다
        self.assertEqual(message_body(at), before)


class TemplateFormTest(unittest.TestCase):
    """첫 화면 사고 접수 양식."""

    def setUp(self):
        agent.reset_memory()
        self.fake = FakeLLM([FULL, FULL])
        self._patches = [patch.object(agent, "call_structured", self.fake.structured),
                         patch.object(agent, "call_text", self.fake.text)]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        self.at = AppTest.from_file(APP, default_timeout=60).run()

    def fill(self, **values):
        for key, value in values.items():
            self.at.text_input(key=f"tpl_{key}").set_value(value) if key != "description" \
                else self.at.text_area(key="tpl_description").set_value(value)
        next(b for b in self.at.button if b.label == "사고 접수하기").click().run()

    def test_form_shown_on_empty_chat(self):
        labels = [w.label for w in self.at.text_input] + [w.label for w in self.at.text_area]
        self.assertEqual(labels, ["사고 일시", "사고 장소", "부상 여부", "사고 내용"])

    def test_submit_sends_template_and_hides_form(self):
        self.fill(occurred_at="어제 오후 3시", location="강남역", injury="목이 조금 아파요",
                  description="신호대기 중에 뒤에서 차가 들이받았어요")
        texts = chat_texts(self.at)
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "사고 일시: 어제 오후 3시  \n사고 장소: 강남역  \n부상 여부: 목이 조금 아파요  \n"
                                   "사고 내용: 신호대기 중에 뒤에서 차가 들이받았어요")
        self.assertIn("[추정 결과]", texts[1])
        self.assertEqual(len(self.at.text_area), 0)  # 대화가 시작되면 양식이 사라진다

    def test_blank_fields_are_skipped_and_description_required(self):
        self.fill(occurred_at="어제")
        self.assertEqual(len(self.at.chat_message), 0)
        self.assertIn("사고 내용을 적어 주세요.", [w.value for w in self.at.warning])

    def test_new_chat_brings_form_back_empty(self):
        self.fill(location="강남역", description="뒤에서 받혔어요")
        self.at.button(key="new_chat").click().run()
        self.assertEqual(len(self.at.chat_message), 0)
        self.assertEqual(self.at.text_input(key="tpl_location").value, "")


if __name__ == "__main__":
    unittest.main()
