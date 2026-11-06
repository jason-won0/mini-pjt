"""약관 RAG 파이프라인: 조항 단위 인덱싱과 하이브리드 검색.

검색 순서 (CLAUDE.md "RAG 설정"):
  ① Chroma 벡터 검색 상위 8개, 유사도(0~1)가 MIN_SIMILARITY 미만이면 제거
  ② BM25(kiwipiepy 형태소) 상위 8개, 0점 제거
  ③ RRF(k=60)로 합쳐 상위 4개 (RRF 점수 하한 없음)
  ④ 결과가 없으면 빈 목록 → 호출한 쪽에서 "근거 없음"으로 처리
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from src import llm, tools  # llm을 먼저 불러 .env(자격 증명)를 읽는다
from src.schemas import PolicyClause

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
CHROMA_DIR = tools.DATA_DIR / "chroma"
COLLECTION = "policy_clauses"
VECTOR_K = 8
BM25_K = 8
TOP_N = 4
RRF_K = 60
# T3-3 calibrate 결과: 관련 질문의 최저 유사도 0.247(치료비 → ART-03), 무관한 질문의 최고 유사도 0.163(주식 투자).
MIN_SIMILARITY = 0.2

logging.getLogger("langchain_aws").setLevel(logging.WARNING)  # 임베딩 호출마다 남는 INFO 로그를 끈다


class RetrievalFailed(Exception):
    code = "retrieval_failed"


@dataclass
class Hit:
    clause: PolicyClause
    rrf: float = 0.0
    vector_sim: float | None = None
    bm25: float | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.clause.id


# ---------------------------------------------------------------- 토크나이저 (교육 코드 day2_practice/hybrid.py)
@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


# 교육 코드의 품사 목록에서 의존명사(NNB)·대명사(NP)를 뺐다. '수', '데', '내' 같은 말이 무관한 질문을 약관에 묶기 때문이다.
KEEP_TAGS = {"NNG", "NNP", "NR", "VV", "VA", "SL", "SN"}
# 어디에나 나오는 어간·일반 명사. 이 말만 겹쳐서 무관한 질문에 BM25 점수가 생기지 않게 한다.
STOP_FORMS = {"하", "되", "있", "없", "받", "주", "알리", "보", "이", "같", "경우", "때", "등"}


def korean_tokenizer(text: str) -> list[str]:
    """명사, 어간, 숫자, 외래어만 남기고 불용어는 뺀다."""
    return [t.form for t in _kiwi().tokenize(text) if t.tag in KEEP_TAGS and t.form not in STOP_FORMS]


def clause_text(c: PolicyClause) -> str:
    return f"{c.title}. {c.text}"


# ---------------------------------------------------------------- BM25
@lru_cache(maxsize=1)
def _bm25() -> tuple[BM25Okapi, tuple[PolicyClause, ...]]:
    clauses = tools.load_policy()
    return BM25Okapi([korean_tokenizer(clause_text(c)) for c in clauses]), clauses


def bm25_search(query: str, k: int = BM25_K) -> list[tuple[PolicyClause, float]]:
    index, clauses = _bm25()
    tokens = korean_tokenizer(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(zip(clauses, scores), key=lambda x: x[1], reverse=True)
    return [(c, float(s)) for c, s in ranked[:k] if s > 0]


# ---------------------------------------------------------------- 벡터 (Chroma)
def _policy_hash() -> str:
    raw = json.dumps([c.model_dump() for c in tools.load_policy()], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def _vectorstore():
    from langchain_aws import BedrockEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL, region_name=llm.REGION)
    store = Chroma(
        collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    hash_file = CHROMA_DIR / "policy_hash.txt"
    current = _policy_hash()
    if not hash_file.exists() or hash_file.read_text(encoding="utf-8") != current:
        logger.info("약관 인덱스를 다시 만듭니다 (조항 %d개)", len(tools.load_policy()))
        existing = store.get()["ids"]
        if existing:
            store.delete(ids=existing)
        clauses = tools.load_policy()
        store.add_documents(
            [Document(page_content=clause_text(c), metadata={"id": c.id, "coverage": c.coverage, "kind": c.kind})
             for c in clauses],
            ids=[c.id for c in clauses],
        )
        hash_file.write_text(current, encoding="utf-8")
    return store


def vector_search(query: str, k: int = VECTOR_K, min_similarity: float = MIN_SIMILARITY
                  ) -> list[tuple[PolicyClause, float]]:
    """유사도(1 − cosine 거리)가 min_similarity 이상인 조항만 돌려준다."""
    results = _vectorstore().similarity_search_with_relevance_scores(query, k=k)
    out = []
    for doc, sim in results:
        clause = tools.get_clause(doc.metadata["id"])
        if clause and sim >= min_similarity:
            out.append((clause, float(sim)))
    return out


# ---------------------------------------------------------------- 결합
def rrf_fuse(vector_hits: list[tuple[PolicyClause, float]], bm25_hits: list[tuple[PolicyClause, float]],
             top_n: int = TOP_N, k: int = RRF_K) -> list[Hit]:
    """RRF: 각 검색 결과의 순위 r에 대해 1/(k + r)을 더한다. 점수 하한은 두지 않는다."""
    hits: dict[str, Hit] = {}
    for source, results in (("vector", vector_hits), ("bm25", bm25_hits)):
        for rank, (clause, score) in enumerate(results, start=1):
            hit = hits.setdefault(clause.id, Hit(clause=clause))
            hit.rrf += 1.0 / (k + rank)
            hit.sources.append(source)
            if source == "vector":
                hit.vector_sim = score
            else:
                hit.bm25 = score
    return sorted(hits.values(), key=lambda h: h.rrf, reverse=True)[:top_n]


def search(query: str, top_n: int = TOP_N,
           vector_fn: Callable[[str], list] | None = None, bm25_fn: Callable[[str], list] | None = None) -> list[Hit]:
    """하이브리드 검색. 결과가 없으면 빈 목록. 검색 자체가 실패하면 RetrievalFailed."""
    try:
        v = (vector_fn or vector_search)(query)
        b = (bm25_fn or bm25_search)(query)
    except Exception as exc:  # 임베딩 호출·인덱스 오류
        logger.warning("검색 실패: error=%s", type(exc).__name__)
        raise RetrievalFailed() from exc
    return rrf_fuse(v, b, top_n=top_n)


def calibrate(queries: list[str]) -> None:
    """MIN_SIMILARITY를 정할 때 쓴다: 질문별 벡터 유사도 상위 3개를 출력한다."""
    for q in queries:
        raw = _vectorstore().similarity_search_with_relevance_scores(q, k=3)
        print(q, "→", [(d.metadata["id"], round(s, 3)) for d, s in raw])


if __name__ == "__main__":
    # 확인 스크립트: python -m src.retriever "질문" ...
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for q in sys.argv[1:] or ["후미추돌 대물 배상"]:
        print(f"\n질문: {q}")
        for h in search(q):
            print(f"  {h.id} {h.clause.title} | rrf={h.rrf:.4f} vec={h.vector_sim} bm25={h.bm25} {h.sources}")
