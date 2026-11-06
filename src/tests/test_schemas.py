"""src/schemas.py 검증 테스트."""
import unittest

from pydantic import ValidationError

from src import schemas as s


def valid_response() -> dict:
    return {
        "answer": "과실 0~10% … ※ 면책",
        "contexts": [{"id": "ART-12", "source": "약관", "text": "…"}],
        "trace": {
            "thread_id": "t-1",
            "intent": "accident",
            "route": "debate",
            "rounds": 2,
            "debate_log": [{"round": 1, "agent": "A1", "content": "…"}],
            "claims": [{"id": "A3-1-1", "kind": "과실조정", "coverage": "대물", "text": "…",
                        "verdict": "확인", "evidence_ids": ["FT-01-M1"], "kept": True, "round": 1}],
            "followup_used": True,
            "followup_reason": "intake",
            "claim_no": "CLM-20261104-001",
            "models_used": ["global.anthropic.claude-sonnet-4-5-20250929-v1:0"],
        },
    }


class ValidExamplesTest(unittest.TestCase):
    def test_response(self):
        resp = s.QueryResponse.model_validate(valid_response())
        self.assertEqual(resp.trace.rounds, 2)
        self.assertEqual(set(resp.model_dump()), {"answer", "contexts", "trace"})

    def test_error_response_without_route(self):
        resp = s.QueryResponse(answer="오류", trace=s.Trace(thread_id="t-1", error="bad_request"))
        self.assertIsNone(resp.trace.route)

    def test_agent_schemas(self):
        s.Intake(intent="accident", accident_type="후미추돌", injury=None)
        s.Estimate(fault_table_id="FT-01", customer_fault={"min": 0, "max": 10},
                   coverages=[{"coverage": "대물", "covered": True, "amount": {"min": 1, "max": 2}}])
        s.A2Evidence(coverage_citations=[{"coverage": "대물", "clause_ids": ["ART-1"]}])
        s.A2Review(verdicts=[{"claim_id": "A3-1-1", "verdict": "근거없음"}])
        s.A3Opinion(claims=[{"kind": "면책", "coverage": "공통", "text": "…"}], no_new_issues=False)
        s.FaultModifier(id="FT-01-M1", description="급정거", bound="max", adjustment=10)
        s.FaultModifier(id="FT-01-M2", description="전방주시", bound="min", adjustment=-10)

    def test_null_strings_become_none(self):
        # 일부 모델은 값이 없을 때 문자열 "null"을 보낸다
        intake = s.Intake.model_validate({"intent": "accident", "accident_type": "후미추돌",
                                          "injury": "null", "damage_level": "NULL", "location": ""})
        self.assertIsNone(intake.injury)
        self.assertIsNone(intake.damage_level)
        self.assertIsNone(intake.location)

    def test_empty_string_kept_for_required_str_fields(self):
        # 빈 문자열은 Optional 필드에서만 None이 된다. reason·text 같은 일반 문자열은 그대로 둔다
        self.assertEqual(s.Verdict.model_validate({"claim_id": "x", "verdict": "확인", "reason": ""}).reason, "")
        self.assertEqual(s.Claim.model_validate({"kind": "기타", "coverage": "공통", "text": ""}).text, "")

    def test_injury_none_value_is_kept(self):
        # "none"은 '부상 없음'이라는 정상 값이므로 None으로 바꾸면 안 된다
        intake = s.Intake.model_validate({"intent": "accident", "injury": "none"})
        self.assertEqual(intake.injury, "none")

    def test_range_equal_bounds(self):
        self.assertEqual(s.Range(min=5, max=5).min, 5)


class InvalidExamplesTest(unittest.TestCase):
    def assertInvalid(self, fn):
        with self.assertRaises(ValidationError):
            fn()

    def test_range_min_gt_max(self):
        self.assertInvalid(lambda: s.Range(min=10, max=5))
        self.assertInvalid(lambda: s.PercentRange(min=30, max=20))

    def test_percent_out_of_bounds(self):
        self.assertInvalid(lambda: s.PercentRange(min=0, max=120))
        self.assertInvalid(lambda: s.PercentRange(min=-5, max=10))

    def test_bad_route(self):
        data = valid_response()
        data["trace"]["route"] = "supervisor"
        self.assertInvalid(lambda: s.QueryResponse.model_validate(data))

    def test_bad_verdict(self):
        self.assertInvalid(lambda: s.Verdict(claim_id="A3-1-1", verdict="보류"))

    def test_bad_bound(self):
        self.assertInvalid(lambda: s.FaultModifier(id="M", description="x", bound="upper", adjustment=10))

    def test_rounds_over_three(self):
        data = valid_response()
        data["trace"]["rounds"] = 4
        self.assertInvalid(lambda: s.QueryResponse.model_validate(data))

    def test_extra_top_level_key(self):
        data = valid_response()
        data["debug"] = {}
        self.assertInvalid(lambda: s.QueryResponse.model_validate(data))

    def test_empty_question(self):
        self.assertInvalid(lambda: s.QueryRequest(question=""))


if __name__ == "__main__":
    unittest.main()
