"""최소 그래프와 run_query() 테스트 (LLM은 가짜 응답으로 교체)."""
import unittest
from unittest.mock import patch

from src import agent, guards, llm, tools
from src.schemas import A1Opinion, A2Evidence, Intake, QueryResponse
from src.tests.fakes import default_value

FAKE_MODEL = "fake-model"


class FakeLLM:
    """LLM 호출을 기록하고 정해진 응답을 돌려준다."""

    def __init__(self, intake: Intake):
        self.intake = intake
        self.seen: list[str] = []

    def _record(self, messages):
        self.seen.extend(str(m.content) for m in messages)

    def structured(self, schema, messages):
        self._record(messages)
        value = self.intake if schema is Intake else default_value(schema)
        return llm.LLMResult(value=value, model_id=FAKE_MODEL)

    def text(self, messages):
        self._record(messages)
        return llm.LLMResult(value="뒤차가 추돌한 사고로 보입니다.", model_id=FAKE_MODEL)


def run_with(fake: FakeLLM, question: str, **kw):
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
        return agent.run_query(question, **kw)


# 필수 항목 4개가 모두 있는 접수 정보 (되묻기 없이 바로 최종 답변으로 간다)
FULL = dict(occurred_at="어제 오후 3시", location="강남역 사거리", injury="none")
REAR_END = Intake(intent="accident", accident_type="후미추돌", **FULL)


class RunQueryTest(unittest.TestCase):
    def test_minimal_path(self):
        resp = run_with(FakeLLM(REAR_END), "신호 대기 중에 뒤에서 차가 들이받았어요")
        QueryResponse.model_validate(resp)
        self.assertEqual(set(resp), {"answer", "contexts", "trace"})
        # T1-5 계산값(경미 45만~150만)과 같은 범위, 면책 문구, 기준표 ID
        self.assertIn("경미 45만 원 ~ 150만 원", resp["answer"])
        self.assertIn("고객 과실: 0~10%", resp["answer"])
        self.assertTrue(resp["answer"].endswith(guards.DISCLAIMER))
        self.assertEqual([c["id"] for c in resp["contexts"]], ["FT-01", "ART-04", "ART-06", "ART-02"])
        self.assertEqual(resp["trace"]["models_used"], [FAKE_MODEL] * 5)  # 접수, A1, A2 검색, A3, 답변 (새 주장이 없어 A2 검증은 건너뜀)
        self.assertTrue(resp["trace"]["thread_id"].startswith("t-"))

    def test_known_damage_level(self):
        intake = Intake(intent="accident", accident_type="후미추돌", damage_level="moderate", **FULL)
        resp = run_with(FakeLLM(intake), "뒤에서 받혔고 범퍼가 꽤 부서졌어요")
        dmg = tools.load_damage_table().vehicle_damage["moderate"]
        r = tools.calc_property(dmg, tools.find_fault_entry("후미추돌").customer_fault, 200_000_000)
        self.assertIn(agent.money_range(r.min, r.max), resp["answer"])

    def test_pii_never_reaches_llm_or_response(self):
        fake = FakeLLM(REAR_END)
        resp = run_with(fake, "뒤에서 받혔어요. 제 번호는 010-1234-5678, 차량 12가3456 입니다")
        joined = "\n".join(fake.seen)
        self.assertNotIn("010-1234-5678", joined)
        self.assertNotIn("12가3456", joined)
        self.assertIn("[전화번호]", joined)
        self.assertEqual(guards.find_pii(resp["answer"]), [])

    def test_pii_not_in_logs(self):
        with self.assertLogs("src.agent", level="INFO") as logs:
            run_with(FakeLLM(REAR_END), "뒤에서 받혔어요 010-1234-5678")
        self.assertNotIn("010-1234-5678", "\n".join(logs.output))

    def test_unknown_accident_type(self):
        resp = run_with(FakeLLM(Intake(intent="accident", accident_type="자전거와 충돌", **FULL)), "자전거와 부딪혔어요")
        self.assertIn(guards.NOT_CONFIRMED, resp["answer"])
        self.assertEqual(resp["contexts"], [])

    def test_parked_car_is_full_customer_fault(self):
        # 주차된 차를 긁은 사고는 FT-08(둘 다 이동 중, 40~60%)이 아니라 FT-11(고객 과실 100%)이다
        intake = Intake(intent="accident", accident_type="주차 차량 접촉", damage_level="minor", **FULL)
        resp = run_with(FakeLLM(intake), "집 아래 주차장에서 주차된 차를 긁었어요")
        self.assertIn("FT-11 주차 차량 접촉", resp["answer"])
        self.assertIn("고객 과실: 100% (상대방 0%)", resp["answer"])
        self.assertIn("상대방 과실이 없어 받을 금액이 없습니다", resp["answer"])
        self.assertNotIn("FT-08", resp["answer"])

    def test_role_mismatch_is_treated_as_unknown(self):
        # 주차해 둔 내 차가 긁혔다(피해)는데 유형은 FT-11(가해)로 왔다 → 100%로 안내하지 않고 기준 없음으로 처리
        intake = Intake(intent="accident", accident_type="주차 차량 접촉", customer_role="피해", **FULL)
        resp = run_with(FakeLLM(intake), "주차해 뒀는데 누가 제 차를 긁고 갔어요")
        self.assertIn(guards.NOT_CONFIRMED, resp["answer"])
        self.assertNotIn("100%", resp["answer"])
        self.assertIsNone(resp["trace"]["claim_no"])

    def test_unmatched_asks_once_then_closes(self):
        # 기준표에 없는 사고: 1턴은 상황을 묻고(되묻기 1번), 2턴에도 없으면 같은 질문을 반복하지 않고 마무리한다
        class TwoTurns(FakeLLM):
            def __init__(self):
                super().__init__(None)
                self.intakes = [Intake(intent="accident", accident_type="차선변경 접촉사고", **FULL),
                                Intake(intent="accident", customer_role="쌍방")]

            def structured(self, schema, messages):
                if schema is Intake:
                    self._record(messages)
                    return llm.LLMResult(value=self.intakes.pop(0), model_id=FAKE_MODEL)
                return super().structured(schema, messages)

        fake = TwoTurns()
        first = run_with(fake, "차선변경하다 접촉사고입니다")
        self.assertEqual(first["answer"], guards.UNMATCHED_ASK)
        self.assertEqual((first["trace"]["route"], first["trace"]["followup_used"]), ("followup", True))
        second = run_with(fake, "둘 다 차선변경하다 부딪혔어요", thread_id=first["trace"]["thread_id"])
        self.assertNotEqual(second["answer"], guards.UNMATCHED_ASK)
        self.assertTrue(second["answer"].startswith(guards.UNMATCHED_FINAL))
        # 방향이 쌍방이면 쌍방 유형만 비슷한 유형으로 보여준다
        self.assertIn("- 신호 없는 교차로:", second["answer"])
        self.assertNotIn("- 후미추돌:", second["answer"])
        self.assertIsNone(second["trace"]["claim_no"])

    def test_simultaneous_lane_change(self):
        intake = Intake(intent="accident", accident_type="동시 차로변경", customer_role="쌍방", damage_level="minor", **FULL)
        resp = run_with(FakeLLM(intake), "둘 다 차선변경하다 서로 부딪혔어요")
        self.assertIn("FT-13 동시 차로변경", resp["answer"])
        self.assertIn("고객 과실: 40~60% (상대방 40~60%)", resp["answer"])

    def test_parked_victim_gets_zero_fault(self):
        intake = Intake(intent="accident", accident_type="주차 중 피해", customer_role="피해", damage_level="minor", **FULL)
        resp = run_with(FakeLLM(intake), "주차해 뒀는데 누가 제 차를 긁고 갔어요")
        self.assertIn("FT-12 주차 중 피해", resp["answer"])
        self.assertIn("고객 과실: 0% (상대방 100%)", resp["answer"])

    def test_intake_prompt_forbids_forced_type(self):
        fake = FakeLLM(REAR_END)
        run_with(fake, "뒤에서 받혔어요")
        prompt = fake.seen[0]
        self.assertIn("억지로 고르지 말고 고객 표현을 그대로", prompt)
        self.assertIn("장소(주차장 등)나 낱말만 같다고 고르지 않는다", prompt)
        self.assertIn("주차 차량 접촉(", prompt)  # 새 유형과 설명이 목록에 들어간다

    def test_unknown_type_asks_for_situation_without_claim(self):
        resp = run_with(FakeLLM(Intake(intent="accident", accident_type="주차장에서 긁힘", **FULL)), "주차장에서 긁혔어요")
        self.assertIn(guards.NOT_CONFIRMED, resp["answer"])
        self.assertIn("상대 차가 주차·정차해 있었는지", resp["answer"])
        self.assertIsNone(resp["trace"]["claim_no"])  # 기준이 없으면 청구를 만들지 않는다

    def test_on_event_streams_debate_once_in_order(self):
        events = []
        resp = run_with(FakeLLM(REAR_END), "신호 대기 중에 뒤에서 차가 들이받았어요", on_event=events.append)
        debate = [(e["round"], e["agent"]) for e in events if e["type"] == "debate"]
        # 토론 발언은 생기는 순서대로 한 번씩만 알린다 (서브그래프·상위 그래프 중복 없음)
        self.assertEqual(debate, [(e["round"], e["agent"]) for e in resp["trace"]["debate_log"]])
        self.assertEqual(debate[:3], [(1, "A1"), (1, "A2-evidence"), (1, "A3")])
        nodes = [e["node"] for e in events if e["type"] == "node"]
        self.assertLess(nodes.index("supervisor_intake"), nodes.index("a1_estimate"))
        self.assertIn("write_answer", nodes)

    def test_on_event_does_not_change_result(self):
        plain = run_with(FakeLLM(REAR_END), "뒤에서 받혔어요", thread_id="t-a")
        streamed = run_with(FakeLLM(REAR_END), "뒤에서 받혔어요", thread_id="t-b", on_event=lambda e: None)
        import re
        no_claim_no = lambda a: re.sub(r"CLM-\d{8}-\d{3}", "CLM-…", a)  # 청구번호만 차례로 다르다
        self.assertEqual(no_claim_no(plain["answer"]), no_claim_no(streamed["answer"]))
        self.assertEqual(plain["trace"]["claims"], streamed["trace"]["claims"])

    def test_answer_tokens_are_streamed(self):
        # 고객 안내문은 채팅 모델의 토큰을 그대로 흘려보낸다 (LangGraph messages 스트림)
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage
        text = "뒤차가 추돌한 사고로 보입니다. 상대 보험으로 보상받을 수 있어 보입니다."
        model = GenericFakeChatModel(messages=iter([AIMessage(content=text)]))
        fake = FakeLLM(REAR_END)
        events = []
        with patch.object(agent, "call_structured", fake.structured), \
             patch.object(agent, "call_text", lambda m: llm.LLMResult(value=llm.get_text(model.invoke(m)), model_id="f")):
            resp = agent.run_query("뒤에서 받혔어요", on_event=events.append)
        tokens = [e for e in events if e["type"] == "token"]
        self.assertGreater(len(tokens), 3)  # 한 번에가 아니라 여러 조각으로 온다
        self.assertEqual({e["node"] for e in tokens}, {"write_answer"})  # 구조화 출력(토론)은 토큰으로 보내지 않는다
        self.assertEqual("".join(e["text"] for e in tokens), text)
        self.assertTrue(resp["answer"].startswith(text))  # 화면은 흘려보낸 뒤 나머지(숫자 블록)만 이어 쓴다

    def test_claims_event_after_synthesis(self):
        events = []
        resp = run_with(FakeLLM(REAR_END), "뒤에서 받혔어요", on_event=events.append)
        claims_events = [e for e in events if e["type"] == "claims"]
        self.assertEqual(len(claims_events), 1)  # 종합이 끝난 뒤 한 번
        self.assertEqual(claims_events[0]["claims"], resp["trace"]["claims"])  # trace.claims와 같은 형식
        nodes = [e.get("node") for e in events]
        self.assertLess(events.index(claims_events[0]), nodes.index("write_answer"))

    def test_answer_numbers_and_ids_are_removed(self):
        class NumberLLM(FakeLLM):
            def text(self, messages):
                return llm.LLMResult(value=("후미추돌 사고로 보입니다. 과실은 0~10%입니다. 근거는 ART-04입니다.\n"
                                            "```\n0~10%\n```\n상대 보험으로 보상받을 수 있어 보입니다."), model_id="f")
        resp = run_with(NumberLLM(REAR_END), "뒤에서 받혔어요")
        first = resp["answer"].split("\n\n")[0]  # LLM이 쓴 안내문 부분 (뒤는 코드 블록)
        self.assertEqual(first, "후미추돌 사고로 보입니다.\n상대 보험으로 보상받을 수 있어 보입니다.")

    def test_clean_answer_text(self):
        text, dropped = agent.clean_answer_text("보장 대상으로 보입니다. 과실은 90%입니다! 이상입니다.")
        self.assertEqual(text, "보장 대상으로 보입니다. 이상입니다.")
        self.assertEqual(dropped, ["과실은 90%입니다!"])
        self.assertEqual(agent.clean_answer_text("FT-01 기준입니다.")[0], guards.ANSWER_FALLBACK)

    def test_answer_uses_answer_models(self):
        seen = {}
        with patch.object(llm, "chat", lambda m, models=None: seen.setdefault("models", models) or llm.LLMResult("", "f")):
            agent.call_text([])
        self.assertEqual(seen["models"], llm.ANSWER_MODELS)
        self.assertEqual(llm.ANSWER_MODELS[0], "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
        self.assertEqual(llm.ANSWER_MODELS[-1], llm.PRIMARY_MODEL)  # Claude가 막히면 gpt-oss로 넘어간다

    def test_broken_callback_does_not_break_answer(self):
        def boom(event):
            raise RuntimeError("화면 오류")
        resp = run_with(FakeLLM(REAR_END), "뒤에서 받혔어요", on_event=boom)
        self.assertIn("[추정 결과]", resp["answer"])
        self.assertIsNone(resp["trace"]["error"])

    def test_empty_question(self):
        resp = agent.run_query("   ")
        self.assertEqual(resp["trace"]["error"], "bad_request")
        self.assertIsNone(resp["trace"]["route"])

    def test_llm_unavailable(self):
        def boom(*a, **k):
            raise llm.LLMUnavailable([("m", "Err")])
        with patch.object(agent, "call_structured", boom):
            resp = agent.run_query("뒤에서 받혔어요")
        self.assertEqual(resp["answer"], guards.ERROR_MESSAGE)
        self.assertEqual(resp["trace"]["error"], "llm_unavailable")
        QueryResponse.model_validate(resp)


class FormatTest(unittest.TestCase):
    def test_won(self):
        self.assertEqual(agent.won(450_000), "45만 원")
        self.assertEqual(agent.won(15_000_000), "1,500만 원")
        self.assertEqual(agent.won(864_196), "864,196원")
        self.assertEqual(agent.won(0), "0원")


if __name__ == "__main__":
    unittest.main()
