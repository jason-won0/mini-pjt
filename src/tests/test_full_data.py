"""전체 가상 데이터(T3-1) 검증 테스트."""
import json
import unittest

from src import tools


class FullDataTest(unittest.TestCase):
    def test_validate_data_passes(self):
        summary = tools.validate_data()
        self.assertGreaterEqual(summary["약관 조항"], 15)
        self.assertGreaterEqual(summary["기준표 항목"], 10)
        self.assertEqual(summary["고객"], 3)

    def test_three_customers_differ(self):
        c1, c2, c3 = (tools.get_customer(i) for i in ("C001", "C002", "C003"))
        self.assertTrue(c1.coverages["자차"].joined)  # ① 종합 가입
        self.assertFalse(c2.coverages["자차"].joined)  # ② 의무보험만
        self.assertLess(c2.coverages["대물"].limit, c1.coverages["대물"].limit)
        self.assertFalse(c3.coverages["자차"].joined)  # ③ 자차 없음 + 특약
        self.assertEqual(c3.riders, ["SP-01"])

    def test_every_fault_entry_has_range_and_valid_modifiers(self):
        for e in tools.load_fault_table():
            with self.subTest(e=e.id):
                self.assertLessEqual(e.customer_fault.min, e.customer_fault.max)
                for m in e.modifiers:
                    self.assertTrue(m.id.startswith(e.id + "-M"))

    def test_policy_has_each_kind_and_coverage(self):
        clauses = tools.load_policy()
        self.assertEqual({c.kind for c in clauses}, {"정의", "보상", "면책", "감액", "절차"})
        self.assertEqual({c.coverage for c in clauses}, {"대물", "자차", "대인", "공통"})
        self.assertEqual(tools.get_clause("ART-04").coverage, "대물")
        self.assertIsNone(tools.get_clause("ART-99"))

    def test_bodily_fixed(self):
        d = tools.load_damage_table()
        self.assertEqual(set(d.bodily_fixed), {"minor", "severe"})

    def test_rear_end_victim_vs_at_fault_are_separate(self):
        # 고객이 피해(FT-01)인 후미추돌과 고객이 가해(FT-10)인 앞차 추돌이 섞이지 않아야 한다
        self.assertEqual(tools.find_fault_entry("후미추돌").id, "FT-01")
        self.assertEqual(tools.find_fault_entry("앞차 추돌").id, "FT-10")
        self.assertEqual(tools.find_fault_entry("앞차를 추돌").id, "FT-10")

    def test_parked_car_vs_parking_lot_moving(self):
        # 주차된 차 접촉(FT-11, 100%)과 주차장 통로에서 둘 다 움직이다 접촉(FT-08, 40~60%)을 나눈다
        self.assertEqual(tools.find_fault_entry("주차 차량 접촉").id, "FT-11")
        self.assertEqual(tools.find_fault_entry("주차된 차를 긁음").id, "FT-11")
        self.assertEqual(tools.find_fault_entry("주차장 접촉").id, "FT-08")
        ft11 = tools.find_fault_entry("주차 차량 접촉")
        self.assertEqual((ft11.customer_fault.min, ft11.customer_fault.max), (100, 100))
        # 장소만 말한 모호한 표현은 어느 유형에도 붙이지 않는다 (예전 FT-08 별칭 '주차장에서 긁힘' 삭제)
        self.assertIsNone(tools.find_fault_entry("주차장에서 긁힘"))
        self.assertIsNone(tools.find_fault_entry("주차장 사고"))

    def test_parked_victim_is_zero_fault(self):
        ft12 = tools.find_fault_entry("주차 중 피해")
        self.assertEqual((ft12.id, ft12.customer_role), ("FT-12", "피해"))
        self.assertEqual((ft12.customer_fault.min, ft12.customer_fault.max), (0, 0))

    def test_every_entry_has_role(self):
        roles = {e.id: e.customer_role for e in tools.load_fault_table()}
        self.assertEqual(roles["FT-01"], "피해")
        self.assertEqual(roles["FT-08"], "쌍방")
        self.assertEqual(roles["FT-10"], "가해")
        self.assertEqual(roles["FT-11"], "가해")

    def test_role_check(self):
        from src.schemas import Intake
        def entry(t, role):
            e = tools.fault_entry_for(Intake(intent="accident", accident_type=t, customer_role=role))
            return e.id if e else None
        # 주차해 둔 내 차가 긁혔는데(피해) LLM이 FT-11(가해, 100%)을 골라도 코드가 쓰지 않는다
        self.assertIsNone(entry("주차 차량 접촉", "피해"))
        self.assertEqual(entry("주차 차량 접촉", "가해"), "FT-11")
        self.assertIsNone(entry("주차 중 피해", "가해"))
        self.assertIsNone(entry("후미추돌", "가해"))
        self.assertEqual(entry("주차장 접촉", "피해"), "FT-08")  # 쌍방 유형은 어느 방향이든 받는다
        self.assertEqual(entry("주차장 접촉", "가해"), "FT-08")
        self.assertEqual(entry("주차 차량 접촉", None), "FT-11")  # 방향을 모르면 검사하지 않는다
        self.assertIsNone(entry("주차 차량 접촉", "쌍방"))  # 서 있던 차와는 쌍방일 수 없다

    def test_substring_match_prefers_most_specific_name(self):
        # 여러 이름이 포함되면 가장 길게 일치한 이름을 고른다 (기준표 순서에 기대지 않음)
        self.assertEqual(tools.find_fault_entry("주차장에서 주차된 차를 긁음").id, "FT-11")

    def test_no_real_company_names(self):
        # 실제 회사명·브랜드를 쓰지 않는다 (대표적인 표현만 검사)
        text = "".join((tools.DATA_DIR / f).read_text(encoding="utf-8")
                       for f in ["policy.json", "fault_table.json", "customers.json", "documents.json"])
        for word in ["삼성", "현대해상", "DB손해", "KB손해", "메리츠", "한화", "롯데손해", "흥국"]:
            self.assertNotIn(word, text)

    def test_get_modifier(self):
        self.assertEqual(tools.get_modifier("FT-03-M1").adjustment, -10)
        self.assertIsNone(tools.get_modifier("FT-99-M1"))


if __name__ == "__main__":
    unittest.main()
