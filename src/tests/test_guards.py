"""src/guards.py 개인정보 마스킹과 고정 문구 테스트."""
import unittest

from src import guards
from src.guards import mask_pii, find_pii


class MaskRRNTest(unittest.TestCase):
    def test_formats(self):
        for raw in ["900101-1234567", "9001011234567", "900101 - 2234567", "850505-5123456"]:
            with self.subTest(raw=raw):
                out = mask_pii(f"제 주민번호는 {raw}입니다")
                self.assertEqual(out, "제 주민번호는 [주민번호]입니다")


class MaskPhoneTest(unittest.TestCase):
    def test_formats(self):
        for raw in ["010-1234-5678", "01012345678", "010 1234 5678", "010.123.4567",
                    "02-123-4567", "031-1234-5678", "+82-10-1234-5678", "1588-1234"]:
            with self.subTest(raw=raw):
                out = mask_pii(f"연락처 {raw} 로 주세요")
                self.assertEqual(out, "연락처 [전화번호] 로 주세요")


class MaskAccountTest(unittest.TestCase):
    def test_grouped(self):
        for raw in ["110-123-456789", "1002-123-456789", "123-456789-01-001"]:
            with self.subTest(raw=raw):
                self.assertEqual(mask_pii(f"입금 {raw} 부탁"), "입금 [계좌번호] 부탁")

    def test_plain_after_keyword(self):
        self.assertEqual(mask_pii("계좌 1234567890123 으로"), "계좌 [계좌번호] 으로")
        self.assertEqual(mask_pii("통장번호는 12345678901"), "통장번호는 [계좌번호]")


class MaskPlateTest(unittest.TestCase):
    def test_formats(self):
        for raw in ["12가3456", "123가 4567", "12 가 3456", "서울12가3456", "경기 123호4567"]:
            with self.subTest(raw=raw):
                self.assertEqual(mask_pii(f"상대 차량 {raw} 입니다"), "상대 차량 [차량번호] 입니다")


class NoFalsePositiveTest(unittest.TestCase):
    def test_accident_sentences_unchanged(self):
        sentences = [
            "신호 대기 중에 뒤에서 차가 들이받았어요",
            "2026-11-04 오후 3시 30분, 강남역 사거리에서 사고가 났어요",
            "수리비가 200만 원쯤 나왔고 목이 좀 아파요",
            "CLM-20261104-001 진행 상황 알려줘",
            "시속 30km로 가다가 2차로에서 부딪혔어요",
            "어제 12시 30분쯤 3중 추돌이 있었습니다",
            "과실이 20~30% 정도라고 들었어요",
        ]
        for s in sentences:
            with self.subTest(s=s):
                self.assertEqual(mask_pii(s), s)
                self.assertEqual(find_pii(s), [])

    def test_empty(self):
        self.assertEqual(mask_pii(""), "")


class MixedTest(unittest.TestCase):
    def test_all_four_in_one(self):
        raw = "저는 900101-1234567, 010-1234-5678, 계좌 110-123-456789, 차량 12가3456 입니다"
        out = mask_pii(raw)
        self.assertEqual(out, "저는 [주민번호], [전화번호], 계좌 [계좌번호], 차량 [차량번호] 입니다")
        self.assertEqual(find_pii(out), [])
        self.assertEqual(sorted(find_pii(raw)), ["account", "phone", "plate", "rrn"])


class DisclaimerTest(unittest.TestCase):
    def test_append_once(self):
        once = guards.with_disclaimer("과실 20~30%")
        self.assertTrue(once.endswith(guards.DISCLAIMER))
        self.assertEqual(guards.with_disclaimer(once), once)


if __name__ == "__main__":
    unittest.main()
