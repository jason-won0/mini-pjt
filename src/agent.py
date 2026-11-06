"""메인 에이전트 그래프와 진입 함수 run_query().

현재 구조:
  마스킹 → Supervisor(접수) ─┬─ 필수 항목 누락 + 아직 안 물음 → 되묻기 (T2-2)
                             └─ 그 외 → A1(계산 + 설명) → 답변 작성 → 면책 문구
멀티턴은 checkpointer로 thread_id 단위로 이어 간다. API와 Streamlit 모두 run_query() 하나만 부른다.
"""
from __future__ import annotations

import asyncio
import json
import re
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from src import debate, guards, llm, retriever, schemas, tools
from src.schemas import (
    A1Opinion,
    A2Evidence,
    Claim,
    DebateEntry,
    Synthesis,
    TraceClaim,
    Verdict,
    Context,
    CoverageEstimate,
    Estimate,
    Intake,
    QueryResponse,
    Trace,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- LLM 호출 (테스트에서 교체)
def call_structured(schema, messages) -> llm.LLMResult:
    return llm.structured(schema, messages)


def call_text(messages) -> llm.LLMResult:
    """고객 안내문 작성 (역할별 모델: llm.ANSWER_MODELS)."""
    return llm.chat(messages, models=llm.ANSWER_MODELS)


# 안내문에서 코드가 지우는 문장: 숫자·% (숫자는 코드 블록만 쓴다), 근거 ID, 코드 블록 기호
ANSWER_FORBIDDEN_RE = re.compile(r"\d|%|\b(?:ART|FT|SP|DOC|CLM)-|```")


def clean_answer_text(text: str) -> tuple[str, list[str]]:
    """LLM이 쓴 안내문에서 숫자·근거 ID가 든 문장을 지운다. (남은 글, 지운 문장)을 돌려준다.
    숫자와 근거는 코드가 만든 블록에만 나와야 한다(CLAUDE.md '하지 말 것')."""
    kept, dropped = [], []
    for line in text.replace("```", "\n```\n").splitlines():
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", line.strip()) if s]
        good = [s for s in sentences if not ANSWER_FORBIDDEN_RE.search(s)]
        dropped += [s for s in sentences if ANSWER_FORBIDDEN_RE.search(s)]
        if good:
            kept.append(" ".join(good))
    return ("\n".join(kept).strip() or guards.ANSWER_FALLBACK), dropped


retrieval_search = retriever.search  # 약관 검색 (테스트에서 교체)


# ---------------------------------------------------------------- A4: MCP 클라이언트 (T6-4)
MCP_SERVER = Path(__file__).resolve().parent / "mcp_server.py"  # 테스트에서 틀린 경로로 바꿔 실패를 재현한다


class MCPFailed(Exception):
    """MCP 서버 연결·도구 호출 실패. trace.error는 mcp_failed."""

    code = "mcp_failed"


def _text_of(result) -> str:
    """도구 결과에서 텍스트를 꺼낸다 (어댑터 버전에 따라 문자열 또는 콘텐츠 블록 목록)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in result)
    return str(result)


def call_mcp(calls: list[tuple[str, Any]]) -> list[dict]:
    """A4의 MCP 도구 호출. 요청 하나의 호출들을 세션 하나로 묶어 서버 프로세스를 한 번만 띄운다.

    calls의 인자는 dict이거나, 앞 호출 결과 목록을 받아 dict를 돌려주는 함수다 (예: 서류 결과로 청구 저장).
    CLAUDE.md "MCP 연결": A4는 langchain-mcp-adapters로 src/mcp_server.py를 stdio로 띄워 호출하고,
    tools.py의 청구 함수를 직접 import하지 않는다. (테스트에서는 같은 프로세스 호출로 교체한다)
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    async def run() -> list[dict]:
        client = MultiServerMCPClient({"claims": {
            "command": sys.executable, "args": [str(MCP_SERVER)], "transport": "stdio", "env": dict(os.environ),
        }})
        results: list[dict] = []
        async with client.session("claims") as session:
            mcp_tools = {t.name: t for t in await load_mcp_tools(session)}
            for name, args in calls:
                args = args(results) if callable(args) else args
                results.append(json.loads(_text_of(await mcp_tools[name].ainvoke(args))))
        return results

    try:
        return asyncio.run(run())
    except Exception as exc:  # 서버 기동 실패, 연결 끊김, 도구 오류
        logger.warning("MCP 호출 실패: error=%s", type(exc).__name__)
        raise MCPFailed() from exc


# ---------------------------------------------------------------- 상태
class State(TypedDict, total=False):
    # 대화(thread)를 넘어 유지하는 값
    intake: Intake  # 지금까지 합친 접수 정보
    followup_used: bool  # 이 사고에서 되묻기를 썼는지 (한 사고에 1번)
    followup_reason: Optional[str]  # intake / debate
    awaiting_followup: bool  # 직전 응답이 되묻기였는지 (다음 요청은 그 답으로 본다)
    refuse_kind: Optional[str]  # 거절 사유 종류 fraud / off_topic (T7-1)
    # 요청마다 run_query()가 초기화하는 값
    question: str  # 마스킹된 질문
    customer_id: Optional[str]
    thread_id: str  # 대화 ID (A4가 청구 중복 방지에 쓴다)
    route: Optional[str]
    missing: list[str]
    estimate: Optional[Estimate]
    contexts: list[Context]
    a1_explanation: str
    answer: str
    has_estimate: bool
    a2_evidence: Optional[A2Evidence]
    dropped_ids: list[str]
    error: Optional[str]  # 처리를 멈추지 않는 오류 (예: retrieval_failed)
    # 토론 (T4-3): 주장·판정은 주장 ID별로 라운드를 넘어 누적한다
    round: int
    claims: list[Claim]
    verdicts: dict[str, Verdict]
    new_claim_ids: list[str]
    no_new_issues: bool
    a3_info_needed: list[str]
    debate_log: list[DebateEntry]
    synthesis: Optional[Synthesis]  # 종합 (T4-4)
    applied_modifiers: list[str]
    documents: Optional[dict]  # A4 서류 체크리스트 {"required": [...], "conditional": [...]}
    claim_no: Optional[str]  # A4가 발급·조회한 청구번호
    models_used: list[str]


PER_REQUEST_RESET = {
    "route": None, "refuse_kind": None, "missing": [], "estimate": None, "contexts": [], "a1_explanation": "",
    "answer": "", "has_estimate": False, "a2_evidence": None, "dropped_ids": [], "error": None,
    "round": 0, "claims": [], "verdicts": {}, "new_claim_ids": [], "no_new_issues": False, "a3_info_needed": [],
    "debate_log": [], "synthesis": None, "applied_modifiers": [], "documents": None, "claim_no": None,
    "models_used": [],
}


# ---------------------------------------------------------------- 표시용 코드
won = tools.won
money_range = tools.money_range

COVERAGE_LABELS = {
    "대물": "대물(상대방 보험에서 받는 내 차 수리비)",
    "자차": "자차(내 보험에서 받는 내 과실분 수리비, 자기부담금 공제)",
    "대인": "대인(상대방 보험에서 받는 치료 관련 보상, 가상 정액 기준)",
}


def coverage_line(c: CoverageEstimate) -> str:
    label = COVERAGE_LABELS[c.coverage]
    if not c.covered:
        return f"- {label}: 보장되지 않음 ({c.condition})"
    if c.amount and not c.condition:
        return f"- {label}: {money_range(c.amount.min, c.amount.max)}"
    if c.amount:
        return f"- {label}: {money_range(c.amount.min, c.amount.max)} ({c.condition})"
    return f"- {label}: {c.condition}"


def rider_contexts(customer) -> list[Context]:
    """답변에 쓰는 가입 특약 조항. 기준표 항목처럼 코드가 직접 쓰는 근거이므로 contexts에 넣는다."""
    out = []
    for rider_id in customer.riders:
        clause = tools.get_clause(rider_id)
        if clause:
            out.append(Context(id=clause.id, source="약관", text=f"{clause.title}: {clause.text}"))
    return out


def rider_lines(customer) -> list[str]:
    """가입한 특약. 금액은 계산하지 않고 가입 사실과 내용만 안내한다."""
    lines = []
    for rider_id in customer.riders:
        clause = tools.get_clause(rider_id)
        if clause:
            lines.append(f"- 가입 특약: {clause.title} ({rider_id}). 해당 여부는 보험사 확인이 필요합니다")
    return lines


def evidence_lines(estimate: Estimate, evidence: A2Evidence | None) -> list[str]:
    """담보별 약관 근거 조항 ID. 검증을 통과한(검색된) ID만 들어온다. 없으면 '확인되지 않음'."""
    if evidence is None:
        return []
    cited = {c.coverage: c.clause_ids for c in evidence.coverage_citations}
    parts = [f"{c.coverage} {', '.join(cited[c.coverage]) if cited.get(c.coverage) else guards.NOT_CONFIRMED}"
             for c in estimate.coverages]
    lines = [f"- 약관 근거(가상 약관): {' / '.join(parts)}"]
    if evidence.unconfirmed_topics:
        lines.append(f"- 약관에서 {guards.NOT_CONFIRMED}: {', '.join(evidence.unconfirmed_topics)}")
    return lines


def general_info_lines(evidence: A2Evidence | None) -> list[str]:
    if evidence is None or not evidence.general_info:
        return []
    return ["[일반 정보] " + g for g in evidence.general_info]


def pct(r) -> str:
    """과실 범위 표시. 최소와 최대가 같으면 한 값으로 쓴다 (예: 100~100% → 100%)."""
    return f"{r.min}%" if r.min == r.max else f"{r.min}~{r.max}%"


def fault_line(estimate: Estimate, entry, applied: list[str]) -> list[str]:
    f = estimate.customer_fault
    opp = tools.opponent_fault(f)
    base = entry.customer_fault
    if not applied or (f.min, f.max) == (base.min, base.max):
        return [f"- 고객 과실: {pct(f)} (상대방 {pct(opp)})"]
    mods = ", ".join(f"{m.id} {m.description}({m.adjustment:+d}%p)"
                     for m in (tools.get_modifier(i) for i in applied) if m)
    return [f"- 고객 과실: {pct(f)} (상대방 {pct(opp)}) — 기본 {pct(base)}에 보험사 반론의 가능성을 반영해 넓힘",
            f"- 반영한 과실 수정 요소: {mods}"]


def synthesis_lines(synthesis: Synthesis | None) -> list[str]:
    """종합 결과 중 금액을 바꾸지 않는 안내: 보험사 검토 사항(면책·감액), 확인되지 않은 사항(근거없음)."""
    if synthesis is None:
        return []
    lines = [f"- 보험사가 검토할 수 있는 사항: {n.text} (근거 {', '.join(n.evidence_ids)})" for n in synthesis.review_notes]
    lines += [f"- 확인되지 않은 사항: {u}" for u in synthesis.unresolved_issues]
    if synthesis.customer_info_needed:  # 되묻기를 이미 써서 다시 묻지 않은 경우 (조건부 안내)
        lines.append(f"- 추가로 확인이 필요한 정보: {', '.join(synthesis.customer_info_needed)} "
                     "(이 정보에 따라 결과가 달라질 수 있습니다)")
    return lines


def number_block(estimate: Estimate, entry, missing: list[str], customer=None,
                 evidence: A2Evidence | None = None, synthesis: Synthesis | None = None,
                 applied: list[str] | None = None) -> str:
    """숫자와 조건부 안내, 근거 ID는 코드가 만든 이 블록에만 들어간다."""
    lines = [
        "[추정 결과]",
        f"- 적용 기준: {entry.id} {entry.accident_type} (가상 과실비율 기준표)",
        *fault_line(estimate, entry, applied or []),
    ]
    lines += [coverage_line(c) for c in estimate.coverages]
    lines += evidence_lines(estimate, evidence)
    lines += synthesis_lines(synthesis)
    if customer is not None:
        lines += rider_lines(customer)
    lines += conditional_lines(missing)
    return "\n".join(lines)


def conditional_lines(missing: list[str]) -> list[str]:
    """되묻기 뒤에도 비어 있는 필수 항목에 대한 조건부 안내 (부상 여부는 대인 줄에서 조건으로 안내한다)."""
    if not missing:
        return []
    return [f"- 확인되지 않은 항목: {', '.join(tools.missing_labels(missing))} (이 정보에 따라 결과가 달라질 수 있습니다)"]


FOLLOWUP_HINTS = {
    "accident_type": "어떤 상황에서 사고가 났는지",
    "occurred_at": "언제 사고가 났는지",
    "location": "어디서 사고가 났는지",
    "injury": "다치신 곳이 있는지",
}


def followup_question(missing: list[str]) -> str:
    """되묻기 문장 (고정 형식). LLM을 쓰지 않는다."""
    items = "\n".join(f"- {tools.REQUIRED_FIELDS[m]}: {FOLLOWUP_HINTS[m]}" for m in missing)
    return f"정확하게 안내해 드리기 위해 몇 가지 여쭙겠습니다. 아래 내용을 알려 주세요.\n{items}"


def merge_intake(prev: Intake | None, new: Intake, answering_followup: bool) -> Intake:
    """이전 접수 정보에 새로 들어온 값만 덮어쓴다. 되묻기에 답하는 중이면 요청 종류는 유지한다."""
    if prev is None:
        return new
    data = prev.model_dump()
    for key, value in new.model_dump().items():
        if key in ("intent", "deferred_intents"):
            continue
        if key == "facts":
            data["facts"] = list(dict.fromkeys([*prev.facts, *new.facts]))  # 고객이 말한 사실은 쌓는다
        elif value is not None:
            data[key] = value
    if not answering_followup:
        data["intent"] = new.intent
        data["deferred_intents"] = new.deferred_intents
    return Intake.model_validate(data)


# ---------------------------------------------------------------- 라우팅 (코드 규칙 먼저, LLM 분류로 보완)
INTENT_PRIORITY = ["accident", "claim_status", "documents"]  # 사고 평가 > 청구 조회 > 서류 안내
CLAIM_NO_RE = re.compile(r"CLM-\d{8}-\d{3}", re.IGNORECASE)
STATUS_KEYWORDS = ["진행 상황", "진행상황", "처리 상태", "청구 상태", "어디까지 진행"]
DOC_KEYWORDS = ["서류", "준비물", "제출", "구비", "뭘 준비", "무엇을 준비", "뭐 준비"]
SCHEMA_ERRORS = {"ValidationError", "OutputParserException"}


class AgentFailed(Exception):
    """에이전트의 구조화 출력이 모든 모델에서 형식 검증에 실패했다. trace.error는 agent_failed:<이름>."""

    def __init__(self, agent_name: str):
        self.code = f"agent_failed:{agent_name}"
        super().__init__(self.code)


def code_intents(question: str) -> list[str]:
    """코드 규칙으로 찾은 요청 종류 (청구번호 패턴, 조회·서류 키워드)."""
    found = []
    if CLAIM_NO_RE.search(question) or any(k in question for k in STATUS_KEYWORDS):
        found.append("claim_status")
    if any(k in question for k in DOC_KEYWORDS):
        found.append("documents")
    return found


def decide_intents(code: list[str], intake: Intake) -> tuple[str, list[str]]:
    """코드 규칙과 LLM 분류를 합쳐 이번에 처리할 요청 하나와 미룬 요청 목록을 정한다.

    LLM이 보험사기 유도(refuse, fraud)로 분류하면 다른 요청이 섞여 있어도 거절한다(안전 우선).
    무관한 질문(off_topic)은 코드 규칙에 걸린 서비스 요청이 없을 때만 거절한다.
    """
    if intake.intent == "refuse" and intake.refuse_kind == "fraud":
        return "refuse", []
    candidates = set(code) | {intake.intent} | set(intake.deferred_intents)
    serviceable = [i for i in INTENT_PRIORITY if i in candidates]
    if not serviceable:
        return "refuse", []
    return serviceable[0], serviceable[1:]


# ---------------------------------------------------------------- 노드
def supervisor_intake(state: State) -> dict:
    prev: Intake | None = state.get("intake")
    answering_followup = bool(prev) and bool(state.get("awaiting_followup"))
    # 사고 유형은 설명까지 보여 준다 (예: 후미추돌=고객이 피해, 앞차 추돌=고객이 가해)
    types = "; ".join(f"{e.accident_type}({e.description})" for e in tools.load_fault_table())
    context = (
        f"이전까지의 접수 정보: {prev.model_dump_json(exclude_none=True)}. 고객은 같은 사고에 대해 말하고 있다. "
        if prev else ""
    )
    messages = [
        SystemMessage(content=(
            "너는 자동차 사고 보험금 청구 도우미의 접수 담당이다. 고객 입력에서 요청 종류와 접수 정보를 뽑는다. "
            f"사고 유형 목록: {types}. "
            "사고 유형은 목록의 괄호 안 설명과 핵심 조건(어느 차가 움직이고 있었는지, 상대 차가 주차·정차해 있었는지, "
            "누가 어디를 받았는지)이 맞을 때만 그 유형 이름을 그대로 적는다. 장소(주차장 등)나 낱말만 같다고 고르지 않는다. "
            "특히 고객 차가 부딪힌 쪽인지 당한 쪽인지('긁었어요'와 '긁혔어요', '받았어요'와 '받혔어요')를 구분해 "
            "유형 설명의 방향과 같을 때만 고른다. 방향을 알 수 없으면 고르지 않는다. "
            "customer_role에는 유형과 따로 사고 방향만 적는다: 고객 차가 부딪히거나 긁었으면 가해, "
            "상대가 고객 차를 부딪히거나 긁었으면(주차해 둔 고객 차가 당한 경우 포함) 피해, 둘 다 움직이다 부딪혔으면 쌍방, "
            "모르면 null. 과실이 누구에게 있는지가 아니라 누가 부딪혔는지만 본다. "
            "맞는 유형이 없거나 조건이 애매하면 억지로 고르지 말고 고객 표현을 그대로 짧게 적는다. "
            "고객이 사고 상황을 아직 말하지 않았으면 null로 둔다. "
            "입력에 없는 정보는 추측하지 말고 null로 둔다. "
            "한 질문에 여러 요청이 섞이면 우선순위(accident > claim_status > documents)가 가장 높은 것을 intent에, "
            "나머지를 deferred_intents에 넣는다. "
            "사고나 손해를 꾸미거나 부풀리려는 요청은 사고 이야기가 섞여 있어도 intent를 refuse, refuse_kind를 fraud로 한다. "
            "자동차 사고·보험과 무관한 질문은 refuse, refuse_kind를 off_topic으로 한다. " + context
        )),
        HumanMessage(content=state["question"]),
    ]
    try:
        res = call_structured(Intake, messages)
    except llm.LLMUnavailable as exc:
        # 모든 모델이 형식 검증에서 실패했으면 분류 실패, 그 밖(네트워크·한도)은 모델 사용 불가로 본다.
        if exc.attempts and all(err in SCHEMA_ERRORS for _, err in exc.attempts):
            raise AgentFailed("supervisor") from exc
        raise
    intake = merge_intake(prev, res.value, answering_followup)
    intent, deferred = decide_intents(code_intents(state["question"]), intake)
    intake = intake.model_copy(update={"intent": intent, "deferred_intents": deferred})
    return {
        "intake": intake,
        "missing": tools.missing_fields(intake),
        "awaiting_followup": False,
        "models_used": state.get("models_used", []) + [res.model_id],
    }


def route_after_intake(state: State) -> str:
    intent = state["intake"].intent
    if intent in ("claim_status", "documents"):
        return "direct"
    if intent == "refuse":
        return "refused"
    if state.get("missing") and not state.get("followup_used"):
        return "ask_followup"
    return "debate"


def with_deferred(answer: str, state: State) -> str:
    note = guards.deferred_note(state["intake"].deferred_intents)
    return f"{answer}\n\n{note}" if note else answer


def accident_summary(intake: Intake) -> str:
    """청구 기록에 남길 사고 요약 (마스킹된 접수 정보로 코드가 만든다)."""
    parts = [intake.accident_type, intake.occurred_at, intake.location]
    return " / ".join(p for p in parts if p) or "사고 정보 미상"


def documents_lines(docs: dict | None) -> list[str]:
    if not docs:
        return []
    lines = ["[필요 서류]", *[f"- {d}" for d in docs.get("required", [])]]
    if docs.get("conditional"):
        lines.append(f"- 부상이 있다면 추가로: {', '.join(docs['conditional'])}")
    return lines


def claim_lines(claim_no: str | None, error: str | None) -> list[str]:
    if claim_no:
        return ["[청구 접수]", f"- 청구번호: {claim_no} (상태: 접수)",
                "- 새 대화에서 이 청구번호로 진행 상황을 조회할 수 있습니다."]
    if error in (MCPFailed.code, tools.StoreError.code):
        return ["[청구 접수]", f"- {guards.RETRY_LATER}"]
    return []


def a4_after_debate(state: State) -> dict:
    """A4 (토론 뒤): 필요 서류를 안내하고 청구를 저장해 청구번호를 받는다. MCP로만 호출한다."""
    if not state.get("has_estimate"):
        return {}
    intake: Intake = state["intake"]
    estimate: Estimate = state["estimate"]
    save_args = {
        "customer_id": tools.get_customer(state.get("customer_id")).id, "thread_id": state["thread_id"],
        "accident_summary": accident_summary(intake), "missing_fields": state.get("missing", []),
        "customer_fault": estimate.customer_fault.model_dump(),
        "coverages": [{"coverage": c.coverage, "covered": c.covered,
                       "amount": c.amount.model_dump() if c.amount else None} for c in estimate.coverages],
    }
    docs_args = {"fault_table_id": estimate.fault_table_id, "injury": intake.injury}
    try:
        # 서류 조회와 청구 저장을 한 세션에서 이어 한다 (저장할 때 서류 조회 결과를 쓴다)
        docs, saved = call_mcp([
            ("document_checklist", docs_args),
            ("save_claim", lambda prev: {**save_args, "documents": prev[0].get("required", [])}),
        ])
    except MCPFailed as exc:
        return {"error": exc.code}
    if "error" in saved:
        return {"documents": docs, "error": saved["error"]}
    return {"documents": docs, "claim_no": saved["claim"]["claim_no"]}


def claim_status_answer(claim: dict) -> str:
    status = claim["status"]
    lines = [f"청구번호 {claim['claim_no']}의 진행 상태는 **{status}**입니다. {guards.CLAIM_STATUS_TEXT.get(status, '')}",
             "", "[청구 정보]",
             f"- 사고 요약: {claim['accident_summary']}",
             f"- 접수 일시: {claim['created_at'].replace('T', ' ')}"]
    if claim.get("documents"):
        lines.append(f"- 필요 서류: {', '.join(claim['documents'])}")
    return "\n".join(lines)


def direct(state: State) -> dict:
    """A4 (직접 처리): 청구 조회와 서류 안내. 토론 없이 MCP 도구만 쓴다."""
    intake: Intake = state["intake"]
    customer_id = tools.get_customer(state.get("customer_id")).id
    out: dict = {"route": "direct"}
    try:
        if intake.intent == "claim_status":
            m = CLAIM_NO_RE.search(state["question"])
            claim_no = (m.group(0) if m else intake.claim_no or "").upper()
            if not claim_no:
                answer = guards.ASK_CLAIM_NO
            else:
                [res] = call_mcp([("get_claim_status", {"claim_no": claim_no, "customer_id": customer_id})])
                if "error" in res:
                    out["error"], answer = res["error"], guards.ERROR_MESSAGE
                elif res.get("found"):
                    answer, out["claim_no"] = claim_status_answer(res["claim"]), res["claim"]["claim_no"]
                else:
                    answer = guards.CLAIM_NOT_FOUND
        else:  # documents: 같은 대화에 사고 정보가 있으면 그 사고 기준으로 안내한다
            entry = tools.fault_entry_for(intake)
            [docs] = call_mcp([("document_checklist", {"fault_table_id": entry.id if entry else None,
                                                        "injury": intake.injury})])
            head = (f"말씀하신 사고({entry.accident_type}) 기준으로 필요한 서류입니다." if entry
                    else "일반적으로 필요한 서류입니다. 사고 유형에 따라 블랙박스 영상 등이 더 필요할 수 있습니다.")
            answer = head + "\n\n" + "\n".join(documents_lines(docs))
            out["documents"] = docs
    except MCPFailed as exc:
        out["error"], answer = exc.code, guards.ERROR_MESSAGE
    out["answer"] = with_deferred(answer, state)
    return out


def refused(state: State) -> dict:
    """LLM 분류로 거절할 때 (코드 규칙에 걸리지 않은 표현). 고정 문구로 답한다."""
    kind = state["intake"].refuse_kind or "off_topic"
    logger.info("거절(LLM 분류): kind=%s", kind)  # 사유 종류만 남긴다
    return {"answer": guards.REFUSAL_MESSAGES[kind], "route": "refused", "refuse_kind": kind}


def guard_check(state: State) -> dict:
    """거절 판정 (코드 규칙, LLM 호출 전). 마스킹된 질문만 본다."""
    kind = guards.refusal_kind(state["question"])
    if kind is None:
        return {}
    logger.info("거절(코드 규칙): kind=%s", kind)
    out = {"answer": guards.REFUSAL_MESSAGES[kind], "route": "refused", "refuse_kind": kind}
    if not state.get("intake"):  # 대화 중간이면 앞서 받은 사고 정보는 지우지 않는다
        out["intake"] = Intake(intent="refuse", refuse_kind=kind)
    return out


def route_after_guard(state: State) -> str:
    return END if state.get("route") == "refused" else "supervisor_intake"


def ask_followup(state: State) -> dict:
    return {
        "answer": with_deferred(followup_question(state["missing"]), state),
        "route": "followup",
        "followup_used": True,
        "followup_reason": "intake",
        "awaiting_followup": True,
    }


def route_after_synthesis(state: State) -> str:
    """토론 뒤 되묻기 (T4-5): 고객만 답할 수 있는 정보가 필요하고 아직 되묻지 않았으면 묻는다.
    해결되지 않은 쟁점(unresolved_issues)만 있으면 묻지 않는다."""
    syn: Synthesis | None = state.get("synthesis")
    if syn and syn.customer_info_needed and not state.get("followup_used"):
        return "ask_debate_followup"
    return "a4_after_debate"


def debate_followup_question(info: list[str]) -> str:
    """토론 뒤 되묻기 문장 (고정 형식). 항목은 A2·A3가 낸 '고객만 답할 수 있는 정보'다."""
    items = "\n".join(f"- {i}" for i in info)
    return f"보장과 과실을 더 정확히 살펴보기 위해 몇 가지 여쭙겠습니다. 아래 내용을 알려 주세요.\n{items}"


def ask_debate_followup(state: State) -> dict:
    return {
        "answer": with_deferred(debate_followup_question(state["synthesis"].customer_info_needed), state),
        "route": "followup",
        "followup_used": True,
        "followup_reason": "debate",
        "awaiting_followup": True,
    }


def a1_estimate(state: State) -> dict:
    intake = state["intake"]
    entry = tools.fault_entry_for(intake)
    if entry is None:
        return {"estimate": None, "contexts": [], "has_estimate": False, "a1_explanation": ""}

    estimate = Estimate(
        fault_table_id=entry.id,
        customer_fault=entry.customer_fault,
        coverages=tools.estimate_coverages(tools.get_customer(state.get("customer_id")), entry, intake),
    )
    ctx = Context(
        id=entry.id, source="기준표",
        text=f"{entry.accident_type}: {entry.description} 고객 과실 {entry.customer_fault.min}~{entry.customer_fault.max}%",
    )
    messages = [
        SystemMessage(content=(
            "너는 보장·과실·보험금 추정 담당(A1)이다. 주어진 계산 결과가 어떤 기준에서 나왔는지 2~3문장으로 설명한다. "
            "숫자를 새로 만들거나 바꾸지 말고, 단정하지 않는다."
        )),
        HumanMessage(content=(
            f"사고 접수 정보: {intake.model_dump_json(exclude_none=True)}\n"
            f"적용 기준: {ctx.text}\n계산 결과: {estimate.model_dump_json()}"
        )),
    ]
    res = call_structured(A1Opinion, messages)
    return {
        "estimate": estimate, "contexts": [ctx, *rider_contexts(tools.get_customer(state.get("customer_id")))],
        "has_estimate": True,
        "a1_explanation": res.value.explanation,
        "debate_log": state.get("debate_log", []) + [
            DebateEntry(round=max(state.get("round", 1), 1), agent="A1", content=res.value.explanation)],
        "models_used": state.get("models_used", []) + [res.model_id],
    }


COVERAGE_MEANING = {
    "대물": "상대방 보험의 대물배상이 고객 차량 수리비 중 상대방 과실분을 보상한다",
    "자차": "고객 자신의 자차 보험이 고객 차량 수리비 중 고객 과실분을 보상한다(자기부담금 공제)",
    "대인": "상대방 보험의 대인배상이 고객의 치료 관련 손해를 상대방 과실분만큼 보상한다",
}


def coverage_meaning(estimate: Estimate) -> str:
    """답변을 쓰는 LLM에게 담보 방향과 보장 여부를 코드로 정리해 준다 (T8-2 평가에서 방향 오류가 드러남)."""
    lines = []
    for c in estimate.coverages:
        status = "보장 대상" if c.covered else f"보장되지 않음({c.condition})"
        lines.append(f"- {c.coverage}: {COVERAGE_MEANING[c.coverage]} → {status}")
    return "\n".join(lines)


def cited_coverages(evidence: A2Evidence | None) -> str:
    if evidence is None:
        return ""
    return ", ".join(c.coverage for c in evidence.coverage_citations if c.clause_ids)


def a2_evidence(state: State) -> dict:
    # call_structured·retrieval_search는 호출할 때 찾으므로 테스트에서 교체할 수 있다
    out = debate.a2_evidence(state, call=call_structured, search=retrieval_search)
    ev = out.get("a2_evidence")
    summary = " / ".join(f"{c.coverage} {', '.join(c.clause_ids) or guards.NOT_CONFIRMED}"
                         for c in (ev.coverage_citations if ev else [])) or guards.NOT_CONFIRMED
    out["debate_log"] = state.get("debate_log", []) + [
        DebateEntry(round=state.get("round", 1), agent="A2-evidence", content=f"약관 근거: {summary}")]
    return out


# ---------------------------------------------------------------- 토론 서브그래프 (T4-3)
def start_round(state: State) -> dict:
    return {"round": state.get("round", 0) + 1, "new_claim_ids": []}


def a3_opinion(state: State) -> dict:
    try:
        return debate.a3_opinion(state, call=call_structured, round_no=state["round"])
    except llm.LLMUnavailable as exc:
        if exc.attempts and all(err in SCHEMA_ERRORS for _, err in exc.attempts):
            # A3 실패: 주장 없이 토론을 끝낸다 (CLAUDE.md 예외 처리)
            return {"new_claim_ids": [], "no_new_issues": True, "error": "agent_failed:A3"}
        raise


def a2_review(state: State) -> dict:
    return debate.a2_review(state, call=call_structured, round_no=state["round"])


def route_after_a1(state: State) -> str:
    return "a2_evidence" if state.get("has_estimate") else END


def route_after_round(state: State) -> str:
    """라운드 종료 판정 (코드). 추가 쟁점이 없거나 최대 라운드에 도달하면 끝낸다."""
    if state.get("no_new_issues") or state.get("round", 0) >= debate.MAX_ROUNDS:
        return END
    return "start_round"


def synthesis(state: State) -> dict:
    """종합 노드 (코드). 추정이 없으면(사고 유형 미확인) 건너뛴다."""
    if not state.get("has_estimate"):
        return {}
    return debate.synthesize(state)


def route_round_start(state: State) -> str:
    """2라운드부터는 A1·A2 검색을 다시 돌리지 않는다. 과실 조정은 토론이 끝난 뒤 종합 단계에서 하므로
    토론 중에는 두 에이전트의 입력이 바뀌지 않아 결과가 1라운드와 같기 때문이다 (T4-6 결정)."""
    return "a1_estimate" if state.get("round", 1) == 1 else "a3_opinion"


def build_debate_graph():
    """1라운드: A1 → A2 검색 → A3 → A2 검증. 2라운드부터: A3 → A2 검증 (A1·A2 검색 결과 재사용).
    라운드마다 끝에서 종료를 판정한다. 최대 debate.MAX_ROUNDS 라운드."""
    d = StateGraph(State)
    d.add_node("start_round", start_round)
    d.add_node("a1_estimate", a1_estimate)
    d.add_node("a2_evidence", a2_evidence)
    d.add_node("a3_opinion", a3_opinion)
    d.add_node("a2_review", a2_review)
    d.add_edge(START, "start_round")
    d.add_conditional_edges("start_round", route_round_start, ["a1_estimate", "a3_opinion"])
    d.add_conditional_edges("a1_estimate", route_after_a1, ["a2_evidence", END])
    d.add_edge("a2_evidence", "a3_opinion")
    d.add_edge("a3_opinion", "a2_review")
    d.add_conditional_edges("a2_review", route_after_round, ["start_round", END])
    return d.compile()


def unmatched_final(intake: Intake) -> str:
    """기준표에 맞는 유형이 끝내 없을 때의 마무리 안내 (고정 문구 + 사고 방향이 맞는 기준표 유형 목록, 코드가 만든다)."""
    similar = [e for e in tools.load_fault_table() if tools.role_matches(e.customer_role, intake.customer_role)]
    lines = "\n".join(f"- {e.accident_type}: {e.description}" for e in similar)
    return f"{guards.UNMATCHED_FINAL}\n\n{guards.UNMATCHED_TYPES_HEAD}\n{lines}"


def write_answer(state: State) -> dict:
    intake = state["intake"]
    if not state.get("has_estimate"):
        # 기준표에 맞는 유형이 없다. 상황 묻기는 되묻기 1번에 포함해 한 번만 하고, 그래도 없으면 안내로 마무리한다.
        if not state.get("followup_used"):
            return {"answer": with_deferred(guards.UNMATCHED_ASK, state), "route": "followup",
                    "followup_used": True, "followup_reason": "intake", "awaiting_followup": True}
        return {"answer": with_deferred(unmatched_final(intake), state), "route": "debate"}

    entry = tools.fault_entry_for(intake)
    messages = [
        SystemMessage(content=(
            "너는 자동차 사고 고객을 돕는 상담 담당이다. 아래 설명을 바탕으로 고객에게 보낼 안내를 2~4문장으로 쓴다. "
            "고객 편도 보험사 편도 아닌 중립적인 말투로 쓴다. 숫자(과실 비율, 금액)는 쓰지 않는다. 숫자는 별도 블록으로 붙는다. "
            "면책 문구도 쓰지 않는다. '접수', '확인되었다', '발급', '확정', '진행 가능'처럼 처리 결과로 들리는 표현을 쓰지 않는다. "
            "'유리하다', '불리하다'처럼 한쪽 편을 드는 평가도 쓰지 않는다. 예상과 안내로만 쓴다. "
            "A1 설명과 계산 결과에 없는 담보나 보장은 언급하지 않는다. 약관 근거가 확인된 담보만 '약관상 보장 대상으로 보인다'고 "
            "안내하고, 조항 ID는 쓰지 않는다(별도 블록에 붙는다). "
            "이 안내는 고객이 받는 보상만 다룬다. 아래 '담보별 의미와 보장 여부'를 그대로 따르고, 상대 차량의 수리비나 "
            "상대방의 치료비를 고객이 받는 보장처럼 말하지 않는다."
        )),
        HumanMessage(content=(
            f"사고 정보: {intake.model_dump_json(exclude_none=True)}\nA1 설명: {state['a1_explanation']}\n"
            f"약관 근거가 확인된 담보: {cited_coverages(state.get('a2_evidence')) or '없음'}\n"
            f"담보별 의미와 보장 여부(코드 판정):\n{coverage_meaning(state['estimate'])}"
        )),
    ]
    res = call_text(messages)
    text, removed = clean_answer_text(res.value)
    if removed:
        logger.info("안내문에서 숫자·근거 ID가 든 문장 %d개를 지움", len(removed))
    block = number_block(state["estimate"], entry, state.get("missing", []),
                         tools.get_customer(state.get("customer_id")), state.get("a2_evidence"),
                         state.get("synthesis"), state.get("applied_modifiers"))
    info = "\n".join(general_info_lines(state.get("a2_evidence")))
    a4 = "\n".join(documents_lines(state.get("documents")) + [""] + claim_lines(state.get("claim_no"), state.get("error")))
    answer = f"{text}\n\n{block}" + (f"\n\n{a4.strip()}" if a4.strip() else "") + (f"\n\n{info}" if info else "")
    return {
        "answer": guards.with_disclaimer(with_deferred(answer, state)),  # 면책 문구는 맨 끝
        "route": "debate",  # 사고 평가 경로 (토론은 묶음 4에서 추가)
        "models_used": state.get("models_used", []) + [res.model_id],
    }


# ---------------------------------------------------------------- 그래프
def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("supervisor_intake", supervisor_intake)
    g.add_node("ask_followup", ask_followup)
    g.add_node("direct", direct)
    g.add_node("refused", refused)
    g.add_node("debate", build_debate_graph())
    g.add_node("write_answer", write_answer)
    g.add_node("guard_check", guard_check)
    g.add_edge(START, "guard_check")
    g.add_conditional_edges("guard_check", route_after_guard, ["supervisor_intake", END])
    g.add_conditional_edges(
        "supervisor_intake", route_after_intake, ["ask_followup", "direct", "refused", "debate"]
    )
    g.add_edge("ask_followup", END)
    g.add_edge("direct", END)
    g.add_edge("refused", END)
    g.add_node("synthesis", synthesis)
    g.add_edge("debate", "synthesis")
    g.add_node("ask_debate_followup", ask_debate_followup)
    g.add_node("a4_after_debate", a4_after_debate)
    g.add_conditional_edges("synthesis", route_after_synthesis, ["ask_debate_followup", "a4_after_debate"])
    g.add_edge("a4_after_debate", "write_answer")
    g.add_edge("ask_debate_followup", END)
    g.add_edge("write_answer", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(serde=_serializer()))


def _serializer() -> JsonPlusSerializer:
    """checkpointer가 상태에 담긴 우리 스키마 객체를 다시 읽을 수 있도록 허용 목록에 등록한다."""
    names = [
        name for name, obj in vars(schemas).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=[(schemas.__name__, n) for n in names])


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def reset_memory() -> None:
    """대화 기억을 모두 지운다 (테스트용)."""
    global _GRAPH
    _GRAPH = None


# ---------------------------------------------------------------- 진입 함수
EventFn = Callable[[dict], None]


def _emit(on_event: EventFn, event: dict) -> None:
    """진행 알림. 화면 쪽 오류가 처리 흐름을 끊지 않게 한다."""
    try:
        on_event(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning("진행 알림 실패: %s", type(exc).__name__)


STREAM_TOKEN_NODES = {"write_answer"}  # 토큰을 그대로 흘려보내는 노드 (일반 텍스트 생성만. 구조화 출력은 JSON이라 제외)


def _run_graph(inputs: dict, config: dict, on_event: EventFn | None) -> dict:
    """그래프를 실행해 최종 상태를 돌려준다. on_event가 있으면 다음을 알린다.
    - {"type": "node", "node", "round"}: 노드가 끝남
    - {"type": "debate", "round", "agent", "content"}: 토론 발언이 새로 생김
    - {"type": "token", "node", "text"}: 고객 안내문을 LLM이 쓰는 중인 토큰 (실제 스트리밍)"""
    graph = _graph()
    if on_event is None:
        return graph.invoke(inputs, config=config)
    seen = 0  # 이미 알린 토론 발언 수 (토론 서브그래프와 상위 그래프가 같은 목록을 다시 보내므로 개수로 거른다)
    latest: dict = {}  # 주장 판정을 만들 때 쓸 최신 값 (claims, verdicts)
    for _ns, mode, data in graph.stream(inputs, config=config, stream_mode=["updates", "messages"], subgraphs=True):
        if mode == "messages":
            chunk, meta = data
            text = llm.get_text(chunk)  # gpt-oss의 추론(reasoning) 블록은 빼고 본문 텍스트만
            if text and meta.get("langgraph_node") in STREAM_TOKEN_NODES:
                _emit(on_event, {"type": "token", "node": meta["langgraph_node"], "text": text})
            continue
        for node, update in (data or {}).items():
            update = update if isinstance(update, dict) else {}
            _emit(on_event, {"type": "node", "node": node, "round": update.get("round")})
            latest.update({k: update[k] for k in ("claims", "verdicts") if k in update})
            if node == "synthesis" and update.get("synthesis"):
                # 종합이 끝나면 최종 반영 여부(kept)가 정해진 A3 주장 판정을 알린다 (trace.claims와 같은 형식)
                claims = trace_claims({**latest, "synthesis": update["synthesis"]})
                _emit(on_event, {"type": "claims", "claims": [c.model_dump() for c in claims]})
            log = update.get("debate_log") or []
            for entry in log[seen:]:
                _emit(on_event, {"type": "debate", **entry.model_dump()})
            seen = max(seen, len(log))
    return graph.get_state(config).values


def run_query(question: str, customer_id: str | None = None, thread_id: str | None = None,
              on_event: EventFn | None = None) -> dict[str, Any]:
    """질문 하나를 처리해 {answer, contexts, trace}를 돌려준다. 어떤 오류에서도 세 키 형식을 지킨다.
    on_event(선택)를 주면 처리 중 진행 상황(노드 완료, 토론 발언)을 알린다. 화면의 스트리밍 표시에 쓴다."""
    thread_id = thread_id or f"t-{uuid.uuid4().hex[:12]}"
    masked = guards.mask_pii(question or "").strip()  # 가장 먼저 마스킹한다

    if not masked:
        return _response(guards.BAD_REQUEST_MESSAGE, [], Trace(thread_id=thread_id, error="bad_request"))

    try:
        tools.get_customer(customer_id)
        state = _run_graph(
            {**PER_REQUEST_RESET, "question": masked, "customer_id": customer_id, "thread_id": thread_id},
            {"configurable": {"thread_id": thread_id}}, on_event,
        )
    except (llm.LLMUnavailable, AgentFailed) as exc:
        logger.warning("run_query 실패: thread=%s error=%s", thread_id, exc.code)
        return _response(guards.ERROR_MESSAGE, [], Trace(thread_id=thread_id, error=exc.code))
    except KeyError:
        return _response(guards.BAD_REQUEST_MESSAGE, [], Trace(thread_id=thread_id, error="bad_request"))

    intake: Intake = state["intake"]
    trace = Trace(
        thread_id=thread_id,
        intent="refuse" if state.get("route") == "refused" else intake.intent,
        route=state.get("route"),
        followup_used=bool(state.get("followup_used")),
        followup_reason=state.get("followup_reason"),
        missing_fields=state.get("missing", []),
        deferred_intents=intake.deferred_intents,
        rounds=state.get("round", 0),
        debate_log=state.get("debate_log", []),
        claims=trace_claims(state),
        review_notes=state["synthesis"].review_notes if state.get("synthesis") else [],
        unresolved_issues=state["synthesis"].unresolved_issues if state.get("synthesis") else [],
        dropped_ids=state.get("dropped_ids", []),
        claim_no=state.get("claim_no"),
        error=state.get("error"),
        models_used=state.get("models_used", []),
    )
    logger.info("run_query 완료: thread=%s intent=%s route=%s", thread_id, intake.intent, trace.route)
    return _response(state["answer"], state.get("contexts", []), trace)


def trace_claims(state: State) -> list[TraceClaim]:
    """A3 주장과 판정. kept는 종합 노드가 최종 반영한 주장(확인된 과실조정·면책·감액)이면 True다."""
    verdicts = state.get("verdicts", {})
    syn: Synthesis | None = state.get("synthesis")
    kept_ids = {c.id for c in syn.kept_claims} if syn else set()
    out = []
    for c in state.get("claims", []):
        v = verdicts.get(c.id)
        out.append(TraceClaim(
            id=c.id, kind=c.kind, coverage=c.coverage, text=c.text,
            verdict=v.verdict if v else None, evidence_ids=v.evidence_ids if v else [],
            kept=c.id in kept_ids, round=int(c.id.split("-")[1]),
        ))
    return out


def error_response(error: str, thread_id: str | None = None) -> dict[str, Any]:
    """오류 응답. 세 키 형식을 지키고, 고정 문구를 쓴다."""
    message = guards.BAD_REQUEST_MESSAGE if error == "bad_request" else guards.ERROR_MESSAGE
    return _response(message, [], Trace(thread_id=thread_id or f"t-{uuid.uuid4().hex[:12]}", error=error))


def _response(answer: str, contexts: list[Context], trace: Trace) -> dict[str, Any]:
    # 응답에도 원문 개인정보가 남지 않도록 한 번 더 마스킹한다.
    return QueryResponse(answer=guards.mask_pii(answer), contexts=contexts, trace=trace).model_dump()


if __name__ == "__main__":
    # 확인 스크립트: python -m src.agent "질문1" ["질문2" ...]  (같은 thread_id로 이어서 보낸다)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    questions = sys.argv[1:] or ["신호 대기 중에 뒤에서 차가 들이받았어요"]
    tid = None
    for q in questions:
        out = run_query(q, thread_id=tid)
        tid = out["trace"]["thread_id"]
        print(f"\n>>> {q}")
        print(json.dumps(out, ensure_ascii=False, indent=2))
