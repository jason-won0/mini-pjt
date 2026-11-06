"""종합 노드와 과실 조정 테스트 (코드만, LLM 없음)."""
import re
import unittest
from unittest.mock import patch

from src import agent, debate, llm, tools
from src.schemas import (
    A2Review, A3Opinion, Claim, Context, Estimate, FaultModifier, Intake, PercentRange, QueryResponse, Verdict,
)
from src.tests.fakes import default_value

INTAKE = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
                injury="minor", damage_level="moderate")
ENTRY = tools.find_fault_entry("후미추돌")  # 고객 과실 0~10%


def mod(bound, adj, mid="M"):
    return FaultModifier(id=mid, description="테스트", bound=bound, adjustment=adj)


def pr(lo, hi):
    return PercentRange(min=lo, max=hi)


class AdjustFaultTest(unittest.TestCase):
    """확인 기준 ①~④: 과실 조정 규칙. 수정 요소는 가능성이라 범위를 넓히기만 한다 (+는 상한, −는 하한)."""

    def test_1_single_upper_plus_10(self):
        self.assertEqual(tools.adjust_fault(pr(0, 10), [mod("max", 10)]), pr(0, 20))

    def test_2_same_bound_capped_at_20(self):
        # +15와 +10 → 합계 25 → 20으로 제한
        self.assertEqual(tools.adjust_fault(pr(20, 30), [mod("max", 15), mod("max", 10)]), pr(20, 50))

    def test_3_lower_minus_30_clipped_to_zero(self):
        # −30 → −20으로 제한 → 10 − 20 = −10 → 0으로 자름
        self.assertEqual(tools.adjust_fault(pr(10, 30), [mod("min", -30)]), pr(0, 30))

    def test_4_never_narrows(self):
        # 예전 규칙이면 하한 +20으로 20~20%가 됐다. 지금은 +값이라 상한만 올라 0~30%
        self.assertEqual(tools.adjust_fault(pr(0, 10), [mod("min", 20)]), pr(0, 30))
        # 예전 규칙이면 상한 −10으로 0~0%가 됐다(실제 호출에서 생긴 문제). 지금은 하한을 내리므로 그대로 0~10%
        self.assertEqual(tools.adjust_fault(pr(0, 10), [mod("max", -10)]), pr(0, 10))

    def test_opposite_adjustments_widen_both_sides(self):
        # 급정거(+10)와 상대 과속(−10)이 둘 다 확인되면 양쪽으로 넓힌다
        self.assertEqual(tools.adjust_fault(pr(20, 30), [mod("max", 10), mod("max", -10)]), pr(10, 40))

    def test_upper_clipped_to_100(self):
        self.assertEqual(tools.adjust_fault(pr(90, 100), [mod("max", 20)]), pr(90, 100))


def base_state(claims, verdicts, customer_id="C001"):
    estimate = Estimate(fault_table_id=ENTRY.id, customer_fault=ENTRY.customer_fault,
                        coverages=tools.estimate_coverages(tools.get_customer(customer_id), ENTRY, INTAKE))
    return {"intake": INTAKE, "estimate": estimate, "customer_id": customer_id, "claims": claims,
            "verdicts": {v.claim_id: v for v in verdicts}, "contexts": [Context(id="FT-01", source="기준표", text="…")],
            "a2_evidence": None, "a3_info_needed": []}


def claim(cid, kind, coverage, basis):
    return Claim(id=cid, kind=kind, coverage=coverage, text=f"{cid} 주장", candidate_basis_ids=basis)


class SynthesizeTest(unittest.TestCase):
    def test_confirmed_fault_adjustment_recalculates(self):
        c = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])  # 상한 +10
        out = debate.synthesize(base_state([c], [Verdict(claim_id=c.id, verdict="확인", evidence_ids=["FT-01-M1"])]))
        self.assertEqual(out["estimate"].customer_fault, pr(0, 20))
        own = {x.coverage: x for x in out["estimate"].coverages}["자차"]
        # 보통 피해(150만~500만) × 고객 0~20% − 20만 → 0 ~ 80만 (조정 전 0 ~ 30만)
        self.assertEqual((own.amount.min, own.amount.max), (0, 800_000))
        self.assertEqual([k.id for k in out["synthesis"].kept_claims], ["A3-1-1"])
        self.assertIn("FT-01-M1", [x.id for x in out["contexts"]])

    def test_5_confirmed_exemption_keeps_amount_and_adds_note(self):
        c = claim("A3-1-1", "면책", "대물", ["ART-13"])
        state = base_state([c], [Verdict(claim_id=c.id, verdict="확인", evidence_ids=["ART-13"])])
        out = debate.synthesize(state)
        self.assertEqual(out["estimate"], state["estimate"])  # 금액 그대로
        self.assertEqual(out["synthesis"].review_notes[0].evidence_ids, ["ART-13"])
        self.assertIn("ART-13", [x.id for x in out["contexts"]])

    def test_6_rejected_and_no_basis_have_no_effect(self):
        c1 = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])
        c2 = claim("A3-1-2", "과실조정", "공통", ["FT-01-M1"])
        state = base_state([c1, c2], [Verdict(claim_id="A3-1-1", verdict="기각", evidence_ids=["FT-01-M1"]),
                                      Verdict(claim_id="A3-1-2", verdict="근거없음", evidence_ids=[])])
        out = debate.synthesize(state)
        self.assertEqual(out["estimate"], state["estimate"])
        self.assertEqual(out["synthesis"].kept_claims, [])
        self.assertEqual(len(out["synthesis"].dropped_claims), 2)
        self.assertEqual(out["synthesis"].unresolved_issues, ["A3-1-2 주장"])  # 근거없음만 '확인되지 않은 사항'

    def test_other_kind_confirmed_is_not_reflected(self):
        c = claim("A3-1-1", "기타", "공통", ["ART-14"])
        out = debate.synthesize(base_state([c], [Verdict(claim_id=c.id, verdict="확인", evidence_ids=["ART-14"])]))
        self.assertEqual(out["synthesis"].kept_claims, [])
        self.assertEqual(out["synthesis"].review_notes, [])

    def test_same_modifier_counted_once(self):
        c1 = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])
        c2 = claim("A3-2-1", "과실조정", "대물", ["FT-01-M1"])
        vs = [Verdict(claim_id=c.id, verdict="확인", evidence_ids=["FT-01-M1"]) for c in (c1, c2)]
        out = debate.synthesize(base_state([c1, c2], vs))
        self.assertEqual(out["estimate"].customer_fault, pr(0, 20))  # +10 한 번만


class SynthesisInAnswerTest(unittest.TestCase):
    """그래프 전체: 종합 결과가 답변·trace·contexts에 나온다."""

    def test_answer_trace_and_contexts(self):
        c_fault = Claim(kind="과실조정", coverage="공통", text="고객이 급정거했다면 과실이 늘 수 있다", candidate_basis_ids=["FT-01-M1"])
        c_exempt = Claim(kind="면책", coverage="대물", text="사고와 무관한 수리는 보상되지 않는다", candidate_basis_ids=["ART-13"])
        c_bad = Claim(kind="감액", coverage="대인", text="근거 없는 감액 주장", candidate_basis_ids=["ART-99"])

        def structured(schema, messages):
            if schema is Intake:
                return llm.LLMResult(value=INTAKE, model_id="f")
            if schema is A3Opinion:
                first = structured.calls == 0
                structured.calls += 1
                return llm.LLMResult(value=A3Opinion(claims=[c_fault, c_exempt, c_bad] if first else [],
                                                     no_new_issues=not first), model_id="f")
            if schema is A2Review:
                return llm.LLMResult(value=A2Review(verdicts=[
                    Verdict(claim_id="A3-1-1", verdict="확인", evidence_ids=["FT-01-M1"]),
                    Verdict(claim_id="A3-1-2", verdict="확인", evidence_ids=["ART-13"]),
                    Verdict(claim_id="A3-1-3", verdict="확인", evidence_ids=["ART-99"])]), model_id="f")
            return llm.LLMResult(value=default_value(schema), model_id="f")
        structured.calls = 0

        agent.reset_memory()
        with patch.object(agent, "call_structured", structured), \
                patch.object(agent, "call_text", lambda m: llm.LLMResult(value="안내", model_id="f")):
            resp = agent.run_query("어제 강남역에서 뒤차가 받았고 목이 아파요")
        QueryResponse.model_validate(resp)
        a, t = resp["answer"], resp["trace"]
        self.assertIn("고객 과실: 0~20% (상대방 80~100%) — 기본 0~10%에 보험사 반론의 가능성을 반영해 넓힘", a)
        self.assertIn("반영한 과실 수정 요소: FT-01-M1", a)
        self.assertIn("보험사가 검토할 수 있는 사항: 사고와 무관한 수리는 보상되지 않는다 (근거 ART-13)", a)
        self.assertIn("확인되지 않은 사항: 근거 없는 감액 주장", a)
        self.assertEqual({c["id"]: c["kept"] for c in t["claims"]}, {"A3-1-1": True, "A3-1-2": True, "A3-1-3": False})
        self.assertEqual(t["claims"][2]["verdict"], "근거없음")  # 없는 ID → 코드가 근거없음
        self.assertEqual(len(t["review_notes"]), 1)
        # 답변에 나오는 모든 ID가 contexts에 있다
        ids_in_answer = set(re.findall(r"(?:ART|SP|FT)-\d+(?:-M\d+)?", a))
        self.assertTrue(ids_in_answer <= {c["id"] for c in resp["contexts"]}, ids_in_answer)


if __name__ == "__main__":
    unittest.main()


class ClaimTextCleanTest(unittest.TestCase):
    """평가 D1: '확인되지 않은 사항'에 contexts에 없는 ID와 영어 문장이 그대로 나가지 않는다."""

    def test_drops_ids_not_in_contexts(self):
        text, gone = debate.clean_claim_text("안전띠를 매지 않았다면 ART-12에 따라 대인 보상을 감액할 수 있다", {"ART-02"})
        self.assertEqual(gone, ["ART-12"])
        self.assertNotIn("ART", text)
        self.assertIn("안전띠", text)

    def test_keeps_allowed_and_normalizes_hyphen(self):
        text, gone = debate.clean_claim_text("기존 손상은 ART‑13에 따라 제외된다", {"ART-13"})
        self.assertEqual(gone, [])
        self.assertIn("ART-13", text)

    def test_english_sentence_dropped(self):
        text, gone = debate.clean_claim_text("If the insured failed to notify, reduce under ART-14.", set())
        self.assertIsNone(text)
        self.assertEqual(gone, ["ART-14"])

    def test_synthesize_cleans_unresolved_and_records_dropped(self):
        c = Claim(id="A3-1-1", kind="감액", coverage="대인", text="통지가 늦었다면 ART-14에 따라 감액할 수 있다",
                  candidate_basis_ids=["ART-14"])
        state = {"intake": INTAKE, "estimate": agent.a1_estimate({"intake": INTAKE, "customer_id": "C001"})["estimate"]
                 if hasattr(agent, "a1_estimate") else None,
                 "customer_id": "C001", "claims": [c],
                 "verdicts": {"A3-1-1": Verdict(claim_id="A3-1-1", verdict="근거없음", evidence_ids=[], reason="")},
                 "contexts": [], "dropped_ids": ["SP-77"]}
        if state["estimate"] is None:
            self.skipTest("a1_estimate 없음")
        out = debate.synthesize(state)
        self.assertEqual(out["synthesis"].unresolved_issues, ["통지가 늦었다면 감액할 수 있다"])
        self.assertEqual(out["dropped_ids"], ["SP-77", "ART-14"])


class A3NoBasisFilterTest(unittest.TestCase):
    def test_claims_without_basis_ids_are_ignored(self):
        op = A3Opinion(claims=[Claim(kind="면책", coverage="공통", text="고의 사고라면 면책된다", candidate_basis_ids=[])],
                       no_new_issues=False)
        res = llm.LLMResult(value=op, model_id="fake") if hasattr(llm, "LLMResult") else None
        state = {"intake": INTAKE, "estimate": None, "claims": []}
        est = Estimate(fault_table_id=ENTRY.id, customer_fault=ENTRY.customer_fault, coverages=[])
        state["estimate"] = est
        out = debate.a3_opinion(state, lambda schema, msgs: res, 1)
        self.assertEqual(out["claims"], [])
        self.assertTrue(out["no_new_issues"])


class ClaimTextNumberTest(unittest.TestCase):
    def test_sentence_with_number_dropped(self):
        text, _ = debate.clean_claim_text("급정거했다면 과실 비율을 10% 증가시켜야 합니다", set())
        self.assertIsNone(text)
