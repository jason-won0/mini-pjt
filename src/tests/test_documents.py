"""서류 체크리스트 테스트."""
import unittest

from src import tools

BASE = ["사고 경위서", "사고 현장·차량 파손 사진", "차량 수리 견적서"]
INJURY = ["진단서", "치료비 영수증", "교통사고 사실확인원"]
BLACKBOX = "블랙박스 영상 또는 목격자 진술서"


class RequiredDocumentsTest(unittest.TestCase):
    def test_no_injury(self):
        required, conditional = tools.required_documents("FT-01", "none")
        self.assertEqual(required, BASE)
        self.assertEqual(conditional, [])

    def test_injury_adds_medical_documents(self):
        for injury in ("minor", "severe"):
            with self.subTest(injury=injury):
                required, conditional = tools.required_documents("FT-01", injury)
                self.assertEqual(required, BASE + INJURY)
                self.assertEqual(conditional, [])

    def test_injury_unknown_is_conditional(self):
        required, conditional = tools.required_documents("FT-01", None)
        self.assertEqual(required, BASE)
        self.assertEqual(conditional, INJURY)

    def test_disputed_accident_type_needs_blackbox(self):
        # 신호위반(FT-02)은 과실 다툼이 잦아 블랙박스 영상이 필요하다. 후미추돌(FT-01)은 아니다
        self.assertIn(BLACKBOX, tools.required_documents("FT-02", "none")[0])
        self.assertNotIn(BLACKBOX, tools.required_documents("FT-01", "none")[0])

    def test_unknown_accident_type(self):
        required, _ = tools.required_documents(None, "none")
        self.assertEqual(required, BASE)

    def test_order_follows_data(self):
        required, _ = tools.required_documents("FT-03", "minor")
        self.assertEqual(required, [d.name for d in tools.load_documents()])  # 모든 규칙이 해당하면 데이터 순서 그대로


if __name__ == "__main__":
    unittest.main()
