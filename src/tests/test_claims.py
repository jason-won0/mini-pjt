"""청구 저장소 테스트 (임시 파일 사용, 실제 data/claims.json은 건드리지 않는다)."""
import json
import shutil
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src import guards, tools
from src.schemas import ClaimRecord

DAY = datetime(2026, 11, 5, 10, 0, 0)


class ClaimStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "claims.json"
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def save(self, customer="C001", thread="t-1", **kw):
        return tools.save_claim(customer, thread, "뒤차 추돌", now=kw.pop("now", DAY), path=self.path, **kw)

    def test_numbers_issued_in_order_per_day(self):
        a, new_a = self.save(thread="t-1")
        b, new_b = self.save(thread="t-2")
        self.assertEqual((a.claim_no, b.claim_no), ("CLM-20261105-001", "CLM-20261105-002"))
        self.assertTrue(new_a and new_b)
        c, _ = self.save(thread="t-3", now=datetime(2026, 11, 6, 9, 0))
        self.assertEqual(c.claim_no, "CLM-20261106-001")  # 날짜가 바뀌면 001부터

    def test_new_claim_status_is_received(self):
        rec, _ = self.save()
        self.assertEqual(rec.status, "접수")

    def test_same_thread_returns_existing(self):
        first, _ = self.save(thread="t-same")
        again, created = self.save(thread="t-same")
        self.assertFalse(created)
        self.assertEqual(again.claim_no, first.claim_no)
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 1)

    def test_only_own_claims_are_found(self):
        rec, _ = self.save(customer="C001")
        self.assertEqual(tools.get_claim(rec.claim_no, "C001", path=self.path).claim_no, rec.claim_no)
        self.assertEqual(tools.get_claim(rec.claim_no.lower(), "C001", path=self.path).claim_no, rec.claim_no)
        # 다른 고객의 번호와 없는 번호는 똑같이 '찾을 수 없음'
        self.assertIsNone(tools.get_claim(rec.claim_no, "C002", path=self.path))
        self.assertIsNone(tools.get_claim("CLM-20991231-999", "C001", path=self.path))
        self.assertEqual(guards.CLAIM_NOT_FOUND, "해당 청구를 찾을 수 없습니다.")

    def test_write_failure_keeps_existing_file(self):
        self.save(thread="t-1")
        before = self.path.read_bytes()
        with patch.object(tools.json, "dump", side_effect=OSError("disk full")):
            with self.assertRaises(tools.StoreError):
                self.save(thread="t-2")
        self.assertEqual(self.path.read_bytes(), before)  # 기존 파일 그대로
        self.assertEqual(list(self.dir.glob("*.tmp")), [])  # 임시 파일도 남지 않는다
        self.assertFalse((self.dir / "claims.json.lock").exists())  # 잠금도 풀린다

    def test_concurrent_saves_get_unique_numbers(self):
        results = []
        threads = [threading.Thread(target=lambda i=i: results.append(self.save(thread=f"t-{i}")[0].claim_no))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(results)), 8)

    def test_corrupt_file_raises_store_error(self):
        self.path.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(tools.StoreError):
            tools.get_claim("CLM-20261105-001", "C001", path=self.path)


class SampleClaimsTest(unittest.TestCase):
    """data/claims.json의 가상 샘플 청구 (읽기만 한다)."""

    def test_samples_cover_other_statuses(self):
        raw = json.loads(tools.CLAIMS_PATH.read_text(encoding="utf-8"))
        samples = [ClaimRecord.model_validate(r) for r in raw if r["thread_id"] is None]
        self.assertTrue({"서류보완", "심사중", "지급완료", "부지급"} <= {s.status for s in samples})
        customer_ids = {c.id for c in tools.load_customers()}
        self.assertTrue(all(s.customer_id in customer_ids for s in samples))

    def test_sample_lookup(self):
        self.assertEqual(tools.get_claim("CLM-20261101-002", "C001").status, "심사중")
        self.assertIsNone(tools.get_claim("CLM-20261101-002", "C003"))


if __name__ == "__main__":
    unittest.main()
