"""과실 조회와 대물 계산 테스트 (경계값 포함)."""
import unittest

from src import tools
from src.schemas import PercentRange, Range

DMG = Range(min=1_000_000, max=2_000_000)


class FindFaultEntryTest(unittest.TestCase):
    def test_exact_and_alias(self):
        for name in ["후미추돌", "후미 추돌", "신호 대기 중 추돌", "후미추돌 사고"]:
            with self.subTest(name=name):
                self.assertEqual(tools.find_fault_entry(name).id, "FT-01")

    def test_unknown(self):
        self.assertIsNone(tools.find_fault_entry("자전거와 충돌"))
        self.assertIsNone(tools.find_fault_entry(None))


class OpponentFaultTest(unittest.TestCase):
    def test_flip(self):
        opp = tools.opponent_fault(PercentRange(min=20, max=30))
        self.assertEqual((opp.min, opp.max), (70, 80))


class CalcPropertyTest(unittest.TestCase):
    def calc(self, fmin, fmax, limit=None, dmg=DMG):
        r = tools.calc_property(dmg, PercentRange(min=fmin, max=fmax), limit)
        return r.min, r.max

    def test_lower_bound_is_min_damage_times_min_ratio(self):
        # ① 고객 과실 20~30% → 상대방 70~80%. 하한 = 100만 × 70%
        self.assertEqual(self.calc(20, 30)[0], 700_000)

    def test_upper_bound_is_max_damage_times_max_ratio(self):
        # ② 상한 = 200만 × 80%
        self.assertEqual(self.calc(20, 30)[1], 1_600_000)

    def test_customer_fault_zero(self):
        # ③ 고객 과실 0% → 상대방 100%
        self.assertEqual(self.calc(0, 0), (1_000_000, 2_000_000))

    def test_customer_fault_hundred(self):
        # ③ 고객 과실 100% → 상대방 0%
        self.assertEqual(self.calc(100, 100), (0, 0))

    def test_rear_end_default(self):
        # 후미추돌 기본값(고객 0~10%) + 경미 손해(50만~150만): 45만~150만
        entry = tools.find_fault_entry("후미추돌")
        dmg = tools.load_damage_table().vehicle_damage["minor"]
        r = tools.calc_property(dmg, entry.customer_fault, tools.load_damage_table().opponent_limits["대물"])
        self.assertEqual((r.min, r.max), (450_000, 1_500_000))

    def test_over_limit_is_capped(self):
        # ④ 한도 초과: 상한 200만 → 한도 150만, 하한 100만은 그대로
        self.assertEqual(self.calc(0, 0, limit=1_500_000), (1_000_000, 1_500_000))

    def test_both_bounds_over_limit(self):
        self.assertEqual(self.calc(0, 0, limit=500_000), (500_000, 500_000))

    def test_exactly_at_limit(self):
        # ⑤ 한도와 정확히 같은 경우
        self.assertEqual(self.calc(0, 0, limit=2_000_000), (1_000_000, 2_000_000))

    def test_rounds_down_to_won(self):
        # 1,234,567원 × 70% = 864,196.9 → 864,196원
        r = tools.calc_property(Range(min=1_234_567, max=1_234_567), PercentRange(min=30, max=30), None)
        self.assertEqual(r.min, 864_196)

    def test_result_is_valid_range(self):
        for fmin, fmax in [(0, 10), (10, 90), (50, 50)]:
            lo, hi = self.calc(fmin, fmax)
            self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
