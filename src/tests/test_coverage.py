"""자차·대인 계산과 담보별 보장 판정 테스트."""
import unittest

from src import tools
from src.schemas import Intake, PercentRange, Range

REAR_END = tools.find_fault_entry("후미추돌")  # 고객 과실 0~10%
DMG = Range(min=1_000_000, max=2_000_000)


def by_cov(estimates):
    return {c.coverage: c for c in estimates}


def intake(**kw):
    base = dict(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역")
    return Intake(**{**base, **kw})


class CalcOwnDamageTest(unittest.TestCase):
    def test_basic(self):
        # 고객 과실 20~30%: 100만×20% − 20만 = 0, 200만×30% − 20만 = 40만
        r = tools.calc_own_damage(DMG, PercentRange(min=20, max=30), 200_000, None)
        self.assertEqual((r.min, r.max), (0, 400_000))

    def test_deductible_larger_than_loss_is_zero(self):
        r = tools.calc_own_damage(DMG, PercentRange(min=0, max=10), 500_000, None)
        self.assertEqual((r.min, r.max), (0, 0))

    def test_limit(self):
        r = tools.calc_own_damage(Range(min=50_000_000, max=60_000_000), PercentRange(min=100, max=100), 0, 30_000_000)
        self.assertEqual((r.min, r.max), (30_000_000, 30_000_000))


class CalcBodilyTest(unittest.TestCase):
    def test_basic_and_limit(self):
        r = tools.calc_bodily(Range(min=500_000, max=3_000_000), PercentRange(min=0, max=10), None)
        self.assertEqual((r.min, r.max), (450_000, 3_000_000))
        r = tools.calc_bodily(Range(min=500_000, max=3_000_000), PercentRange(min=0, max=0), 1_000_000)
        self.assertEqual((r.min, r.max), (500_000, 1_000_000))


class EstimateCoveragesTest(unittest.TestCase):
    def test_three_customers_differ(self):
        # 같은 후미추돌(경상, 보통 피해)이 고객 3명마다 다른 담보 결과
        i = intake(injury="minor", damage_level="moderate")
        results = {cid: by_cov(tools.estimate_coverages(tools.get_customer(cid), REAR_END, i))
                   for cid in ("C001", "C002", "C003")}
        self.assertTrue(results["C001"]["자차"].covered)
        self.assertFalse(results["C002"]["자차"].covered)
        self.assertFalse(results["C003"]["자차"].covered)
        self.assertIn("미가입", results["C002"]["자차"].condition)
        # 대물·대인은 상대방 보험에서 받으므로 고객 담보와 관계없이 같다
        for cid in ("C002", "C003"):
            self.assertEqual(results[cid]["대물"].amount, results["C001"]["대물"].amount)
            self.assertEqual(results[cid]["대인"].amount, results["C001"]["대인"].amount)
        # C001 자차: 보통(150만~500만) × 고객 0~10% − 20만 → 0 ~ 30만
        self.assertEqual((results["C001"]["자차"].amount.min, results["C001"]["자차"].amount.max), (0, 300_000))



class ThreeCustomersAnswerTest(unittest.TestCase):
    """확인 기준: 같은 후미추돌이 고객 3명마다 다른 담보 결과로 나온다 (숫자 블록 기준)."""

    def test_number_blocks_all_differ(self):
        from src import agent
        i = intake(injury="minor", damage_level="moderate")
        est = {cid: tools.estimate_coverages(tools.get_customer(cid), REAR_END, i) for cid in ("C001", "C002", "C003")}
        blocks = {}
        for cid, covs in est.items():
            e = agent.Estimate(fault_table_id=REAR_END.id, customer_fault=REAR_END.customer_fault, coverages=covs)
            blocks[cid] = agent.number_block(e, REAR_END, [], tools.get_customer(cid))
        self.assertEqual(len(set(blocks.values())), 3)
        self.assertIn("자차(내 보험에서 받는 내 과실분 수리비, 자기부담금 공제): 0원 ~ 30만 원", blocks["C001"])
        self.assertIn("자차 미가입", blocks["C002"])
        self.assertIn("가입 특약: 대차비용 지원 특약 (SP-01)", blocks["C003"])
        self.assertNotIn("가입 특약", blocks["C002"])

    def test_rider_clause_is_in_contexts(self):
        # 답변에 나오는 특약 ID(SP-01)는 contexts에도 있어야 한다 (평가의 '근거 ID 0건' 조건)
        import re
        from unittest.mock import patch
        from src import agent
        from src.tests.test_a2_evidence import FakeA2, cite
        fake = FakeA2(cite(대물=["ART-04"]))
        agent.reset_memory()
        with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
            resp = agent.run_query("어제 강남역에서 뒤차가 받았고 목이 아파요", customer_id="C003")
        ids_in_answer = set(re.findall(r"(?:ART|SP|FT)-\d+", resp["answer"]))
        self.assertIn("SP-01", ids_in_answer)
        self.assertTrue(ids_in_answer <= {c["id"] for c in resp["contexts"]})

    def test_deductible_makes_own_damage_zero(self):
        # 경미 피해(50만~150만) × 고객 0~10% = 최대 15만 < 자기부담금 20만 → 0
        i = intake(injury="none", damage_level="minor")
        own = by_cov(tools.estimate_coverages(tools.get_customer("C001"), REAR_END, i))["자차"]
        self.assertTrue(own.covered)
        self.assertEqual((own.amount.min, own.amount.max), (0, 0))

    def test_injury_unknown_is_conditional(self):
        i = intake(injury=None, damage_level="minor")
        bodily = by_cov(tools.estimate_coverages(tools.get_customer("C001"), REAR_END, i))["대인"]
        self.assertTrue(bodily.covered)
        self.assertIsNone(bodily.amount)
        self.assertTrue(bodily.condition.startswith("부상이 있다면"))
        self.assertIn("경상 45만 원 ~ 300만 원", bodily.condition)
        self.assertIn("중상", bodily.condition)

    def test_no_injury_is_not_applicable(self):
        bodily = by_cov(tools.estimate_coverages(tools.get_customer("C001"), REAR_END, intake(injury="none")))["대인"]
        self.assertFalse(bodily.covered)

    def test_damage_unknown_is_conditional(self):
        prop = by_cov(tools.estimate_coverages(tools.get_customer("C001"), REAR_END, intake(injury="none")))["대물"]
        self.assertIsNone(prop.amount)
        self.assertEqual(prop.condition,
                         "피해 정도에 따라 경미 45만 원 ~ 150만 원 / 보통 135만 원 ~ 500만 원 / 심각 450만 원 ~ 2,000만 원")

    def test_customer_fully_at_fault(self):
        # 고객 과실 100%: 상대방에게 받을 대물·대인은 없다
        i = intake(accident_type="앞차 추돌", injury="minor", damage_level="minor")
        r = by_cov(tools.estimate_coverages(tools.get_customer("C001"), tools.find_fault_entry("앞차 추돌"), i,
                                            customer_fault=PercentRange(min=100, max=100)))
        self.assertFalse(r["대물"].covered)
        self.assertFalse(r["대인"].covered)
        self.assertTrue(r["자차"].covered)  # 내 과실분은 자차로 받는다

    def test_customer_no_fault_own_damage_zero(self):
        i = intake(injury="none", damage_level="moderate")
        r = by_cov(tools.estimate_coverages(tools.get_customer("C001"), REAR_END, i,
                                            customer_fault=PercentRange(min=0, max=0)))
        self.assertEqual((r["자차"].amount.min, r["자차"].amount.max), (0, 0))
        self.assertEqual((r["대물"].amount.min, r["대물"].amount.max), (1_500_000, 5_000_000))


if __name__ == "__main__":
    unittest.main()
