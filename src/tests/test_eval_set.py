"""평가셋(evaluation/test_queries.csv) 형식 테스트."""
import tempfile
import unittest
from pathlib import Path

from src import evaluate, guards


class EvalSetTest(unittest.TestCase):
    def setUp(self):
        self.queries = evaluate.load_queries()

    def test_columns_and_counts(self):
        counts = evaluate.validate(self.queries)
        self.assertEqual(len(self.queries), 10)
        self.assertEqual(counts, {"debate": 4, "direct": 2, "refuse": 3, "followup": 1})

    def test_lists_are_split(self):
        d1 = next(q for q in self.queries if q.id == "D1")
        self.assertEqual(d1.expected_context_ids, ["FT-01", "ART-04"])
        self.assertIn("[필요 서류]", d1.must_include)

    def test_refuse_items_match_guard_rules(self):
        # 거절 문항 중 사기·무관은 코드 규칙으로 걸리고, 개인정보 문항은 거절이 아니라 마스킹을 본다
        by_id = {q.id: q for q in self.queries}
        self.assertEqual(guards.refusal_kind(by_id["R2"].question), "fraud")
        self.assertEqual(guards.refusal_kind(by_id["R3"].question), "off_topic")
        self.assertIsNone(guards.refusal_kind(by_id["R1"].question))
        self.assertTrue(guards.find_pii(by_id["R1"].question))

    def test_wrong_columns_rejected(self):
        bad = Path(tempfile.mkdtemp()) / "bad.csv"
        bad.write_text("id,type,question\nX,debate,q\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate.load_queries(bad)


if __name__ == "__main__":
    unittest.main()
