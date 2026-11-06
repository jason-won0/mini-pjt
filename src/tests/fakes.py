"""테스트용 가짜 LLM 응답."""
from src.schemas import A1Opinion, A2Evidence, A2Review, A3Opinion, CoverageCitation


def fake_a2_evidence() -> A2Evidence:
    """담보별 대표 조항을 인용한다. 검색되지 않은 ID는 코드가 지운다."""
    return A2Evidence(coverage_citations=[
        CoverageCitation(coverage="대물", clause_ids=["ART-04"]),
        CoverageCitation(coverage="자차", clause_ids=["ART-06"]),
        CoverageCitation(coverage="대인", clause_ids=["ART-02"]),
    ])


def default_value(schema):
    """접수(Intake) 외의 에이전트 응답 기본값. A3는 '추가 쟁점 없음'이라 토론이 1라운드로 끝난다."""
    if schema is A1Opinion:
        return A1Opinion(explanation="가상 기준을 적용했습니다.")
    if schema is A2Evidence:
        return fake_a2_evidence()
    if schema is A3Opinion:
        return A3Opinion(claims=[], no_new_issues=True)
    if schema is A2Review:
        return A2Review(verdicts=[])
    raise AssertionError(f"예상하지 못한 스키마: {schema}")
