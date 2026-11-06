"""A3 노드와 주장 ID·중복 제거 테스트 (LLM은 가짜 응답)."""
import unittest

from src import debate, llm, tools
from src.schemas import A3Opinion, Claim, Estimate, Intake

INTAKE = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
                injury="minor", damage_level="moderate")
ENTRY = tools.find_fault_entry("후미추돌")
ESTIMATE = Estimate(fault_table_id=ENTRY.id, customer_fault=ENTRY.customer_fault,
                    coverages=tools.estimate_coverages(tools.get_customer("C001"), ENTRY, INTAKE))

SUDDEN_STOP = Claim(kind="과실조정", coverage="공통", text="고객이 급정거했다면 고객 과실이 늘 수 있다",
                    candidate_basis_ids=["FT-01-M1"])
SEATBELT = Claim(kind="감액", coverage="대인", text="안전띠를 매지 않았다면 대인 보상이 줄 수 있다",
                 candidate_basis_ids=["ART-12"])


def fake_call(opinion: A3Opinion):
    def call(schema, messages):
        assert schema is A3Opinion
        call.prompt = str(messages[-1].content)
        return llm.LLMResult(value=opinion, model_id="fake")
    return call


def base_state(**kw):
    return {"intake": INTAKE, "estimate": ESTIMATE, "claims": [], "debate_log": [], "models_used": [], **kw}


class RegisterClaimsTest(unittest.TestCase):
    def test_ids_are_assigned_by_code(self):
        llm_ids = [SUDDEN_STOP.model_copy(update={"id": "whatever"}), SEATBELT]
        claims, new_ids = debate.register_claims([], llm_ids, round_no=1)
        self.assertEqual([c.id for c in claims], ["A3-1-1", "A3-1-2"])
        self.assertEqual(new_ids, ["A3-1-1", "A3-1-2"])

    def test_same_claim_three_times_does_not_grow(self):
        claims, _ = debate.register_claims([], [SUDDEN_STOP], round_no=1)
        for r in (2, 3):
            # 표현이 조금 달라도 (종류, 담보, 근거 ID)가 같으면 같은 주장이다
            again = SUDDEN_STOP.model_copy(update={"text": f"{r}라운드: 급정거 가능성"})
            claims, new_ids = debate.register_claims(claims, [again], round_no=r)
            self.assertEqual(new_ids, [])
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].id, "A3-1-1")

    def test_basis_order_does_not_matter(self):
        a = Claim(kind="면책", coverage="자차", text="x", candidate_basis_ids=["ART-11", "ART-13"])
        b = Claim(kind="면책", coverage="자차", text="y", candidate_basis_ids=["ART-13", "ART-11", "ART-11"])
        claims, new_ids = debate.register_claims([], [a, b], round_no=1)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].candidate_basis_ids, ["ART-11", "ART-13"])

    def test_different_basis_is_new_claim(self):
        other = SUDDEN_STOP.model_copy(update={"candidate_basis_ids": ["FT-01-M2"]})
        claims, new_ids = debate.register_claims([], [SUDDEN_STOP, other], round_no=1)
        self.assertEqual(new_ids, ["A3-1-1", "A3-1-2"])

    def test_new_claims_capped_per_round(self):
        many = [Claim(kind="기타", coverage="공통", text=f"주장 {i}", candidate_basis_ids=[f"ART-0{i}"]) for i in range(1, 6)]
        claims, new_ids = debate.register_claims([], many, round_no=1)
        self.assertEqual(new_ids, ["A3-1-1", "A3-1-2", "A3-1-3"])
        self.assertEqual(len(claims), debate.MAX_NEW_CLAIMS_PER_ROUND)

    def test_numbering_continues_within_round(self):
        claims, _ = debate.register_claims([], [SUDDEN_STOP], round_no=2)
        claims, new_ids = debate.register_claims(claims, [SEATBELT], round_no=2)
        self.assertEqual(new_ids, ["A3-2-2"])


class A3NodeTest(unittest.TestCase):
    def test_output_schema_and_unique_ids(self):
        out = debate.a3_opinion(base_state(), fake_call(A3Opinion(claims=[SUDDEN_STOP, SEATBELT], no_new_issues=False)), 1)
        ids = [c.id for c in out["claims"]]
        self.assertEqual(len(ids), len(set(ids)))
        for c in out["claims"]:
            Claim.model_validate(c.model_dump())
        self.assertFalse(out["no_new_issues"])
        self.assertEqual(out["debate_log"][-1].agent, "A3")
        self.assertEqual(out["models_used"], ["fake"])

    def test_repeat_only_means_no_new_issues(self):
        first = debate.a3_opinion(base_state(), fake_call(A3Opinion(claims=[SUDDEN_STOP], no_new_issues=False)), 1)
        # LLM이 같은 주장을 다시 내며 no_new_issues=false라고 해도, 새 주장이 없으면 '추가 쟁점 없음'이다
        second = debate.a3_opinion(base_state(claims=first["claims"]),
                                   fake_call(A3Opinion(claims=[SUDDEN_STOP], no_new_issues=False)), 2)
        self.assertEqual(len(second["claims"]), 1)
        self.assertEqual(second["new_claim_ids"], [])
        self.assertTrue(second["no_new_issues"])
        self.assertEqual(second["debate_log"][-1].content, "추가 쟁점 없음")

    def test_prompt_lists_only_entry_modifiers_and_policy_clauses(self):
        call = fake_call(A3Opinion(claims=[], no_new_issues=True))
        debate.a3_opinion(base_state(), call, 1)
        self.assertIn("[FT-01-M1]", call.prompt)
        self.assertNotIn("[FT-02-M1]", call.prompt)  # 다른 사고 유형의 수정 요소는 주지 않는다
        self.assertIn("[ART-12]", call.prompt)
        self.assertNotIn("[ART-04]", call.prompt)  # 보상 조항은 반론 근거 후보가 아니다


if __name__ == "__main__":
    unittest.main()
