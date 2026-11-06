"""A2 반론 검증 노드 테스트 (LLM은 가짜 응답)."""
import unittest

from src import debate, llm, tools
from src.schemas import A2Review, Claim, Intake, Verdict

INTAKE = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역", injury="minor")
ENTRY = tools.find_fault_entry("후미추돌")


def claim(cid, kind, coverage, basis):
    return Claim(id=cid, kind=kind, coverage=coverage, text="...", candidate_basis_ids=basis)


def review_call(verdicts):
    def call(schema, messages):
        assert schema is A2Review
        call.prompt = str(messages[-1].content)
        call.count = getattr(call, "count", 0) + 1
        return llm.LLMResult(value=A2Review(verdicts=verdicts), model_id="fake")
    return call


def run_review(claims, new_ids, verdicts, prev_verdicts=None):
    state = {"intake": INTAKE, "claims": claims, "new_claim_ids": new_ids,
             "verdicts": prev_verdicts or {}, "debate_log": [], "models_used": []}
    call = review_call(verdicts)
    return debate.a2_review(state, call, round_no=1), call


def v(cid, verdict, ids):
    return Verdict(claim_id=cid, verdict=verdict, evidence_ids=ids, reason="")


class CodeCheckTest(unittest.TestCase):
    """확인 기준: ① 없는 근거 ID ② 종류가 다른 근거 ③ 담보가 다른 조항 → 모두 근거없음."""

    def check(self, c, verdict):
        return debate.check_verdict(c, verdict, ENTRY).verdict

    def test_1_unknown_id(self):
        c = claim("A3-1-1", "과실조정", "공통", ["FT-01-M9"])
        self.assertEqual(self.check(c, v(c.id, "확인", ["FT-01-M9"])), "근거없음")

    def test_2_wrong_kind_fault_adjust_with_clause(self):
        c = claim("A3-1-1", "과실조정", "공통", ["ART-12"])
        self.assertEqual(self.check(c, v(c.id, "확인", ["ART-12"])), "근거없음")

    def test_2_wrong_kind_exemption_with_modifier(self):
        c = claim("A3-1-1", "면책", "대물", ["FT-01-M1"])
        self.assertEqual(self.check(c, v(c.id, "확인", ["FT-01-M1"])), "근거없음")

    def test_3_coverage_mismatch(self):
        # 대물 주장에 자차 조항(ART-11)을 근거로 대면 근거없음
        c = claim("A3-1-1", "면책", "대물", ["ART-11"])
        self.assertEqual(self.check(c, v(c.id, "확인", ["ART-11"])), "근거없음")

    def test_common_clause_matches_any_coverage(self):
        c = claim("A3-1-1", "감액", "대인", ["ART-14"])  # ART-14는 공통
        self.assertEqual(self.check(c, v(c.id, "확인", ["ART-14"])), "확인")

    def test_modifier_of_other_accident_type(self):
        c = claim("A3-1-1", "과실조정", "공통", ["FT-02-M1"])  # 신호위반의 수정 요소
        self.assertEqual(self.check(c, v(c.id, "확인", ["FT-02-M1"])), "근거없음")

    def test_valid_confirm_and_reject_pass_through(self):
        c = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])
        self.assertEqual(self.check(c, v(c.id, "확인", ["FT-01-M1"])), "확인")
        self.assertEqual(self.check(c, v(c.id, "기각", ["FT-01-M1"])), "기각")

    def test_missing_or_empty_evidence(self):
        c = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])
        self.assertEqual(self.check(c, None), "근거없음")
        self.assertEqual(self.check(c, v(c.id, "확인", [])), "근거없음")


class A2ReviewNodeTest(unittest.TestCase):
    def test_reject_and_no_basis_kept_separately(self):
        c1 = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])
        c2 = claim("A3-1-2", "면책", "대물", ["ART-13"])
        c3 = claim("A3-1-3", "감액", "대인", ["ART-12"])
        out, _ = run_review([c1, c2, c3], ["A3-1-1", "A3-1-2", "A3-1-3"],
                            [v("A3-1-1", "확인", ["FT-01-M1"]), v("A3-1-2", "기각", ["ART-13"]), v("A3-1-3", "근거없음", [])])
        got = {k: x.verdict for k, x in out["verdicts"].items()}
        self.assertEqual(got, {"A3-1-1": "확인", "A3-1-2": "기각", "A3-1-3": "근거없음"})
        self.assertEqual(out["debate_log"][-1].agent, "A2-review")

    def test_previous_round_claims_are_not_reverified(self):
        old = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])
        new = claim("A3-2-1", "감액", "대인", ["ART-12"])
        prev = {"A3-1-1": v("A3-1-1", "기각", ["FT-01-M1"])}
        out, call = run_review([old, new], ["A3-2-1"], [v("A3-2-1", "확인", ["ART-12"]), v("A3-1-1", "확인", ["FT-01-M1"])], prev)
        self.assertEqual(out["verdicts"]["A3-1-1"].verdict, "기각")  # 이전 판정 유지
        self.assertEqual(out["verdicts"]["A3-2-1"].verdict, "확인")
        self.assertIn("A3-2-1", call.prompt)
        self.assertNotIn("A3-1-1 (", call.prompt)  # 이전 주장은 판정 대상으로 보내지 않는다

    def test_no_new_claims_skips_llm(self):
        out, call = run_review([claim("A3-1-1", "기타", "공통", [])], [], [])
        self.assertFalse(hasattr(call, "count"))

    def test_schema_failure_marks_all_new_claims_no_basis(self):
        c1 = claim("A3-1-1", "과실조정", "공통", ["FT-01-M1"])

        def boom(schema, messages):
            raise llm.LLMUnavailable([("m1", "ValidationError"), ("m2", "ValidationError")])
        state = {"intake": INTAKE, "claims": [c1], "new_claim_ids": ["A3-1-1"], "verdicts": {},
                 "debate_log": [], "models_used": []}
        out = debate.a2_review(state, boom, round_no=1)
        self.assertEqual(out["verdicts"]["A3-1-1"].verdict, "근거없음")
        self.assertEqual(out["error"], "agent_failed:A2-review")


if __name__ == "__main__":
    unittest.main()
