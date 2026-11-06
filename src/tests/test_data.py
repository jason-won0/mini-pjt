"""data/*.json 로드와 스키마 검증 테스트."""
import unittest

from src import tools


class DataLoadTest(unittest.TestCase):
    def test_customers(self):
        customers = tools.load_customers()
        self.assertGreaterEqual(len(customers), 1)
        c = tools.get_customer(None)
        self.assertEqual(c.id, "C001")
        self.assertTrue(c.coverages["자차"].joined)
        self.assertGreater(c.coverages["자차"].deductible, 0)

    def test_get_customer_unknown(self):
        with self.assertRaises(KeyError):
            tools.get_customer("C999")

    def test_fault_table(self):
        entry = tools.load_fault_table()[0]
        self.assertEqual(entry.id, "FT-01")
        self.assertEqual(entry.accident_type, "후미추돌")
        self.assertEqual(len(entry.modifiers), 2)
        self.assertEqual({m.bound for m in entry.modifiers}, {"max"})
        self.assertEqual(sorted(m.adjustment for m in entry.modifiers), [-10, 10])

    def test_damage_table(self):
        d = tools.load_damage_table()
        self.assertEqual(set(d.vehicle_damage), {"minor", "moderate", "severe"})
        self.assertIn("대물", d.opponent_limits)

    def test_ids_unique(self):
        ids = [e.id for e in tools.load_fault_table()]
        ids += [m.id for e in tools.load_fault_table() for m in e.modifiers]
        ids += [c.id for c in tools.load_customers()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({"FT-01", "FT-01-M1", "FT-01-M2", "ART-04", "SP-01"} <= tools.all_data_ids())


if __name__ == "__main__":
    unittest.main()
