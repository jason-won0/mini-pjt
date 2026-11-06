"""약관 하이브리드 검색 테스트.

- 오프라인 테스트: RRF 결합, 토크나이저, BM25, 실패 처리 (항상 실행)
- 실제 검색 5종: Bedrock 임베딩을 부르므로 RUN_LIVE=1 일 때만 실행
  실행: set RUN_LIVE=1 && python -m unittest src.tests.test_retriever   (PowerShell은 $env:RUN_LIVE=1)
"""
import os
import unittest

from src import retriever, tools

LIVE = os.getenv("RUN_LIVE") == "1"


def clause(cid):
    return tools.get_clause(cid)


class RRFTest(unittest.TestCase):
    def test_both_sources_rank_higher(self):
        v = [(clause("ART-04"), 0.5), (clause("ART-05"), 0.4)]
        b = [(clause("ART-05"), 3.0), (clause("ART-13"), 2.0)]
        hits = retriever.rrf_fuse(v, b)
        self.assertEqual([h.id for h in hits], ["ART-05", "ART-04", "ART-13"])
        self.assertEqual(hits[0].sources, ["vector", "bm25"])
        self.assertAlmostEqual(hits[0].rrf, 1 / 62 + 1 / 61)

    def test_top_n_and_empty(self):
        v = [(clause(f"ART-0{i}"), 0.9 - i / 10) for i in range(1, 8)]
        self.assertEqual(len(retriever.rrf_fuse(v, [])), 4)
        self.assertEqual(retriever.rrf_fuse([], []), [])


class TokenizerBM25Test(unittest.TestCase):
    def test_stop_forms_removed(self):
        tokens = retriever.korean_tokenizer("목을 다쳤는데 치료비 받을 수 있나요")
        self.assertIn("치료비", tokens)
        for stop in ["수", "있", "받", "데"]:
            self.assertNotIn(stop, tokens)

    def test_bm25_unrelated_is_empty(self):
        for q in ["주식 투자 어떻게 해요", "날씨 알려줘", "오늘 저녁 메뉴 추천해 주세요"]:
            with self.subTest(q=q):
                self.assertEqual(retriever.bm25_search(q), [])

    def test_bm25_keyword(self):
        self.assertEqual(retriever.bm25_search("자기부담금")[0][0].id, "ART-07")


class FailureTest(unittest.TestCase):
    def test_vector_failure_raises(self):
        def boom(q):
            raise RuntimeError("embedding down")
        with self.assertRaises(retriever.RetrievalFailed):
            retriever.search("자기부담금", vector_fn=boom)

    def test_vector_empty_falls_back_to_bm25(self):
        hits = retriever.search("자기부담금", vector_fn=lambda q: [])
        self.assertEqual(hits[0].id, "ART-07")


@unittest.skipUnless(LIVE, "RUN_LIVE=1 일 때만 실행 (Bedrock 임베딩 호출)")
class LiveSearchTest(unittest.TestCase):
    """확인 기준: 검색 테스트 5종."""

    def ids(self, q):
        return [h.id for h in retriever.search(q)]

    def test_1_exact_term(self):
        self.assertEqual(self.ids("자기부담금")[0], "ART-07")
        self.assertIn("ART-04", self.ids("후미추돌 대물 배상")[:3])

    def test_2_synonym(self):
        self.assertEqual(self.ids("내 차 수리비를 내 보험으로 받을 수 있나요")[0], "ART-06")
        self.assertEqual(self.ids("목을 다쳤는데 치료비 받을 수 있나요")[0], "ART-03")

    def test_3_typo_and_spacing(self):
        self.assertEqual(self.ids("자기부담금이 얼마나 공재되나요")[0], "ART-07")
        self.assertEqual(self.ids("자기 부담 금은 얼마")[0], "ART-07")

    def test_4_unrelated_is_empty(self):
        for q in ["오늘 저녁 메뉴 추천해 주세요", "주식 투자 어떻게 해요", "날씨 알려줘"]:
            with self.subTest(q=q):
                self.assertEqual(self.ids(q), [])

    def test_5_similar_but_different_clause(self):
        # 대물의 '사고와 관계없는 수리'(ART-13)와 자차의 '보상하지 않는 손해'(ART-11)는 비슷하지만 다르다
        self.assertEqual(self.ids("사고와 관계없는 수리는 대물배상으로 보상되나요")[0], "ART-13")
        self.assertEqual(self.ids("타이어만 찢어졌는데 자차로 보상되나요")[0], "ART-11")


if __name__ == "__main__":
    unittest.main()
