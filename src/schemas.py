"""에이전트 입출력 스키마와 응답 스키마 (CLAUDE.md "에이전트 입출력 스키마", "주고받는 형식").

금액은 원 단위 정수, 비율은 0~100 정수(%)다. 범위는 항상 {"min", "max"}이고 min <= max다.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------- 공통 타입
Intent = Literal["accident", "claim_status", "documents", "refuse"]
Route = Literal["debate", "direct", "followup", "refused"]
Coverage = Literal["대물", "자차", "대인"]
ClaimCoverage = Literal["대물", "자차", "대인", "공통"]
ClaimKind = Literal["과실조정", "면책", "감액", "기타"]
VerdictValue = Literal["확인", "기각", "근거없음"]
Injury = Literal["none", "minor", "severe"]
DamageLevel = Literal["minor", "moderate", "severe"]
CustomerRole = Literal["가해", "피해", "쌍방"]  # 사고 방향: 고객 차가 부딪힘 / 당함 / 둘 다 움직임


# "none"은 넣지 않는다. Intake.injury의 정상 값("부상 없음")이기 때문이다.
_NULL_STRINGS = {"null", "NULL", "Null", ""}


class LLMModel(BaseModel):
    """LLM이 채우는 스키마의 기반. 일부 모델은 값이 없을 때 null 대신 문자열 "null"을 보내므로 None으로 바꾼다.

    None을 허용하는 필드(Optional)에서만 바꾼다. 일반 문자열 필드(예: reason, text)의 빈 문자열은 그대로 둔다.
    """

    @model_validator(mode="before")
    @classmethod
    def _null_strings_to_none(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key, value in data.items():
            field = cls.model_fields.get(key)
            if field is None or not isinstance(value, str) or value.strip() not in _NULL_STRINGS:
                continue
            if field.default is None:  # Optional 필드(기본값 None)만 None으로 바꾼다
                out[key] = None
        return out


class Range(BaseModel):
    """금액 범위(원). min <= max."""

    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> "Range":
        if self.min > self.max:
            raise ValueError(f"범위 하한({self.min})이 상한({self.max})보다 큽니다")
        return self


class PercentRange(Range):
    """비율 범위(%). 0~100, min <= max."""

    min: int = Field(ge=0, le=100)
    max: int = Field(ge=0, le=100)


# ---------------------------------------------------------------- Supervisor
class Intake(LLMModel):
    """고객 입력에서 뽑은 요청 분류와 접수 정보. 입력에 없는 값은 null로 둔다."""

    intent: Intent = Field(description=(
        "가장 우선하는 요청: accident(사고 평가: 사고 설명, 보장·보상 여부, 과실·보험금 질문) > "
        "claim_status(이미 낸 청구의 진행 상태 조회, 보통 청구번호가 있음) > documents(필요 서류 안내). "
        "보험사기 유도나 무관한 질문은 refuse"))
    refuse_kind: Optional[Literal["fraud", "off_topic"]] = Field(default=None, description=(
        "intent가 refuse일 때만: fraud(사고·손해를 꾸미거나 부풀리려는 보험사기 유도) / off_topic(서비스와 무관한 질문)"))
    deferred_intents: list[Intent] = Field(default_factory=list, description=(
        "같은 질문에 섞여 있지만 이번에 처리하지 않는 나머지 요청. 고객이 실제로 함께 요청한 것만 넣고, 없으면 빈 목록"))
    accident_type: Optional[str] = Field(default=None, description="사고 유형 (예: 후미추돌, 신호위반, 차로변경)")
    customer_role: Optional[CustomerRole] = Field(default=None, description=(
        "사고 방향. 과실 판단이 아니라 누가 부딪혔는지만 본다: 가해(고객 차가 상대 차·물체를 부딪히거나 긁음), "
        "피해(상대가 고객 차를 부딪히거나 긁음, 주차해 둔 고객 차가 당한 경우 포함), "
        "쌍방(둘 다 움직이다 서로 부딪혀 한쪽으로 말하기 어려움). 입력으로 알 수 없으면 null"))
    occurred_at: Optional[str] = Field(default=None, description="사고 일시 (고객이 말한 그대로)")
    location: Optional[str] = Field(default=None, description="사고 장소")
    injury: Optional[Injury] = Field(default=None, description="부상 정도: none(없음), minor(경상), severe(중상). 언급이 없으면 null")
    other_vehicle: Optional[str] = Field(default=None, description="상대 차량 정보")
    damage_level: Optional[DamageLevel] = Field(default=None, description="차량 피해 정도: minor(경미), moderate(보통), severe(심각). 언급이 없으면 null")
    claim_no: Optional[str] = Field(default=None, description="질문에 나온 청구번호 (CLM-YYYYMMDD-NNN)")
    facts: list[str] = Field(default_factory=list, description=(
        "고객이 말한 사고 경위 중 과실·보장 판단에 쓸 만한 사실을 짧은 문장으로 (예: '급정거하지 않았음', '안전띠를 맸음'). "
        "고객이 말하지 않은 것은 넣지 않는다"))


# ---------------------------------------------------------------- A1 (계산은 코드)
class CoverageEstimate(BaseModel):
    coverage: Coverage
    covered: bool
    amount: Optional[Range] = None  # 보장되지 않거나 정보가 부족하면 None
    condition: Optional[str] = None  # 정보 부족 시 조건 (예: "부상이 있다면 …")
    basis_ids: list[str] = Field(default_factory=list)


class Estimate(BaseModel):
    fault_table_id: str
    customer_fault: PercentRange
    coverages: list[CoverageEstimate] = Field(default_factory=list)


class A1Opinion(LLMModel):
    """A1의 설명. 숫자는 주어진 계산 결과를 그대로 인용하고 새로 만들지 않는다."""

    explanation: str = Field(description="계산 결과와 보장 판단에 대한 설명. 숫자를 새로 만들지 말 것")


# ---------------------------------------------------------------- A2
class CoverageCitation(LLMModel):
    coverage: Coverage
    clause_ids: list[str] = Field(default_factory=list, description="검색 결과에 있는 약관 조항 ID만")


class A2Evidence(LLMModel):
    """A2 근거 검색 결과."""

    coverage_citations: list[CoverageCitation] = Field(default_factory=list)
    unconfirmed_topics: list[str] = Field(default_factory=list, description="고객이 물었지만 주어진 약관 조항에서 근거를 찾지 못한 보장이나 주제")
    general_info: list[str] = Field(default_factory=list, description="실제 법령의 일반 원칙 설명. 조문 번호와 판례는 쓰지 않는다")
    customer_info_needed: list[str] = Field(default_factory=list, description="고객만 답할 수 있어 되물어야 할 정보")


class Verdict(LLMModel):
    claim_id: str
    verdict: VerdictValue = Field(description="확인: 근거가 있고 주장이 타당 / 기각: 근거상 주장이 틀림 / 근거없음: 관련 근거를 찾지 못함")
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class A2Review(LLMModel):
    """A2 반론 검증 결과. A3 주장마다 하나씩."""

    verdicts: list[Verdict] = Field(default_factory=list)


# ---------------------------------------------------------------- A3
class Claim(LLMModel):
    id: str = ""  # 코드가 A3-{라운드}-{순번}으로 붙인다
    kind: ClaimKind = Field(description="과실조정 / 면책 / 감액 / 기타")
    coverage: ClaimCoverage = Field(description="주장이 해당하는 담보. 담보와 무관하면 공통")
    text: str = Field(description="주장 내용")
    candidate_basis_ids: list[str] = Field(default_factory=list, description="근거 후보 ID (과실조정은 기준표 수정 요소 ID, 면책·감액은 약관 조항 ID)")


class A3Opinion(LLMModel):
    """A3(보험사 관점) 반론."""

    claims: list[Claim] = Field(default_factory=list)
    no_new_issues: bool = Field(description="이번 라운드에 새로 낼 주장이 없으면 true")
    customer_info_needed: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 데이터 (data/*.json)
class FaultModifier(BaseModel):
    id: str
    description: str
    bound: Literal["min", "max"]
    adjustment: int = Field(ge=-100, le=100)  # ±%p


class FaultTableEntry(BaseModel):
    """과실비율 기준표 항목. 과실은 모두 고객 과실 기준이다."""

    id: str
    accident_type: str
    aliases: list[str] = Field(default_factory=list)  # 같은 사고 유형의 다른 표현
    description: str
    customer_role: CustomerRole  # 이 유형에서 고객의 방향. 접수 정보의 방향과 반대면 이 유형을 쓰지 않는다
    customer_fault: PercentRange
    modifiers: list[FaultModifier] = Field(default_factory=list)


class CoverageTerm(BaseModel):
    joined: bool
    limit: Optional[int] = Field(default=None, ge=0)  # None이면 한도 없음
    deductible: int = Field(default=0, ge=0)  # 자기부담금(원)


class Customer(BaseModel):
    id: str
    name: str  # 가상 이름
    plan: str  # 가입 형태 설명
    coverages: dict[Coverage, CoverageTerm]
    riders: list[str] = Field(default_factory=list)  # 특약


class DamageTable(BaseModel):
    vehicle_damage: dict[DamageLevel, Range]  # 피해 정도별 차량 손해액 범위
    bodily_fixed: dict[Literal["minor", "severe"], Range]  # 부상 정도별 대인 가상 정액 범위
    opponent_limits: dict[Coverage, int]  # 가상 상대방 보험의 담보 한도


ClaimStatus = Literal["접수", "서류보완", "심사중", "지급완료", "부지급"]


class CoverageAmount(BaseModel):
    """청구 기록에 남기는 담보별 금액 범위 (보장되지 않거나 조건부면 amount는 None)."""

    coverage: Coverage
    covered: bool
    amount: Optional[Range] = None


class ClaimRecord(BaseModel):
    """청구 저장소(data/claims.json)의 청구 한 건 (SERVICE.md 3장 "청구")."""

    claim_no: str = Field(pattern=r"^CLM-\d{8}-\d{3}$")
    customer_id: str
    thread_id: Optional[str] = None  # 샘플 청구는 None
    created_at: str  # ISO 날짜·시각
    status: ClaimStatus = "접수"
    accident_summary: str
    missing_fields: list[str] = Field(default_factory=list)
    customer_fault: Optional[PercentRange] = None
    coverages: list[CoverageAmount] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)


class PolicyClause(BaseModel):
    """가상 약관 조항. RAG 인덱스의 단위다."""

    id: str
    part: Literal["보통약관", "특약"]
    coverage: ClaimCoverage  # 관련 담보 (담보와 무관하면 공통)
    kind: Literal["정의", "보상", "면책", "감액", "절차"]
    title: str
    text: str


class DocumentRule(BaseModel):
    """필요 서류 규칙. always이거나, 부상이 있거나, 해당 기준표 항목일 때 필요하다."""

    id: str
    name: str
    always: bool = False
    when_injury: bool = False
    fault_table_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 종합 (코드)
class ReviewNote(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    estimate: Estimate
    kept_claims: list[Claim] = Field(default_factory=list)
    dropped_claims: list[Claim] = Field(default_factory=list)
    review_notes: list[ReviewNote] = Field(default_factory=list)
    customer_info_needed: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    context_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 요청·응답 (제출 규약)
class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    customer_id: Optional[str] = None
    thread_id: Optional[str] = None


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: Literal["약관", "기준표"]
    text: str


class DebateEntry(BaseModel):
    round: int = Field(ge=1)
    agent: Literal["A1", "A2-evidence", "A3", "A2-review"]
    content: str


class TraceClaim(BaseModel):
    id: str
    kind: ClaimKind
    coverage: ClaimCoverage
    text: str
    verdict: Optional[VerdictValue] = None
    evidence_ids: list[str] = Field(default_factory=list)
    kept: bool = False
    round: int = Field(ge=1)


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    intent: Optional[Intent] = None  # 분류 전 오류면 None
    route: Optional[Route] = None  # 분류 전 오류면 None
    rounds: int = Field(default=0, ge=0, le=3)
    debate_log: list[DebateEntry] = Field(default_factory=list)
    claims: list[TraceClaim] = Field(default_factory=list)
    review_notes: list[ReviewNote] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    followup_used: bool = False
    followup_reason: Optional[Literal["intake", "debate"]] = None
    missing_fields: list[str] = Field(default_factory=list)
    deferred_intents: list[Intent] = Field(default_factory=list)
    dropped_ids: list[str] = Field(default_factory=list)
    claim_no: Optional[str] = None
    models_used: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class QueryResponse(BaseModel):
    """응답은 answer, contexts, trace 세 키만 최상위에 둔다."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    contexts: list[Context] = Field(default_factory=list)
    trace: Trace
