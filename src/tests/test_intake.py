"""필수 접수 항목 누락 판정 테스트."""
import unittest

from src import tools
from src.schemas import Intake

FULL = dict(intent="accident", accident_type="후미추돌", occurred_at="어제 오후 3시",
            location="강남역 사거리", injury="minor")


class MissingFieldsTest(unittest.TestCase):
    def test_all_present(self):
        self.assertEqual(tools.missing_fields(Intake(**FULL)), [])

    def test_case1_only_accident_type(self):
        # "신호 대기 중에 뒤에서 차가 들이받았어요" → 사고 유형만 있음
        intake = Intake(intent="accident", accident_type="후미추돌")
        self.assertEqual(tools.missing_fields(intake), ["occurred_at", "location", "injury"])

    def test_case2_missing_injury_only(self):
        intake = Intake(**{**FULL, "injury": None})
        self.assertEqual(tools.missing_fields(intake), ["injury"])

    def test_case3_missing_type_and_location(self):
        intake = Intake(**{**FULL, "accident_type": None, "location": "  "})
        self.assertEqual(tools.missing_fields(intake), ["accident_type", "location"])

    def test_no_injury_counts_as_answered(self):
        # 부상 없음(none)은 답한 것이다
        self.assertEqual(tools.missing_fields(Intake(**{**FULL, "injury": "none"})), [])

    def test_null_string_from_llm_is_missing(self):
        intake = Intake.model_validate({**FULL, "occurred_at": "null"})
        self.assertEqual(tools.missing_fields(intake), ["occurred_at"])

    def test_optional_fields_do_not_count(self):
        # 상대 차량, 피해 정도는 선택 항목이라 비어 있어도 누락이 아니다
        intake = Intake(**FULL, other_vehicle=None, damage_level=None)
        self.assertEqual(tools.missing_fields(intake), [])

    def test_labels(self):
        self.assertEqual(tools.missing_labels(["occurred_at", "injury"]), ["사고 일시", "부상 여부"])


if __name__ == "__main__":
    unittest.main()
