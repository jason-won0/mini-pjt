"""토론 노드.

- A2(약관·법률 자문가) 근거 검색 (T3-4)
  ① 코드: 담보별 검색어와 고객 질문으로 약관을 검색한다
  ② LLM: 검색된 조항만 보고 담보별 근거 조항 ID를 고른다 (A2Evidence)
  ③ 코드: 검색 결과에 없는 ID를 지우고(dropped_ids), 조문 번호가 들어간 일반 정보 문장을 지운다
- A3(기업 CEO) 반론 (T4-1)
  ① LLM: 보험사 비용을 줄이는 방향의 반론을 근거 후보 ID와 함께 공격적으로 낸다 (A3Opinion)
  ② 코드: 주장 ID를 붙이고, (종류, 담보, 근거 ID)가 같은 주장은 기존 주장으로 합친다
- A2 반론 검증 (T4-2)
  ① LLM: 이번 라운드의 새 주장마다 확인/기각/근거없음과 근거 ID를 낸다 (A2Review)
  ② 코드: 근거 ID 존재 → 근거 종류 → 담보 일치를 검사하고, 통과한 경우에만 A2 판정을 쓴다
- 종합 (T4-4): 코드로 확인된 주장만 남기고 과실 조정·재계산, 면책·감액은 검토 사항으로 정리한다
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src import guards, llm, retriever, tools
from src.schemas import (
    A2Evidence, A2Review, A3Opinion, Claim, Context, CoverageCitation, DebateEntry, Estimate, Intake, ReviewNote,
    Synthesis, Verdict,
)

logger = logging.getLogger(__name__)

SCHEMA_ERRORS = {"ValidationError", "OutputParserException"}
# 일반 정보에서 버릴 문장: 조문 번호·판례(실제 법령 인용 금지)와 숫자·비율(추정은 코드가 범위로만 낸다)
LAW_ARTICLE_RE = re.compile(r"제\s*\d+\s*조|\d+\s*조\s*\d*\s*항|판례|대법원|선고|\d|퍼센트|%")
COVERAGE_QUERY = {"대물": "대물배상 상대방 차량 수리비", "자차": "자기차량손해 자차 자기부담금", "대인": "대인배상 치료비 부상"}


def build_queries(intake: Intake, estimate: Estimate, question: str) -> list[str]:
    queries = [question]
    for c in estimate.coverages:
        queries.append(f"{intake.accident_type or ''} {COVERAGE_QUERY[c.coverage]}".strip())
    return queries


def retrieve(queries: list[str], search: Callable) -> dict[str, retriever.Hit]:
    """검색어마다 찾은 조항을 합친다. 같은 조항은 RRF 점수가 높은 쪽을 남긴다."""
    found: dict[str, retriever.Hit] = {}
    for q in queries:
        for hit in search(q):
            if hit.id not in found or hit.rrf > found[hit.id].rrf:
                found[hit.id] = hit
    return found


def validate_evidence(evidence: A2Evidence, retrieved_ids: set[str]) -> tuple[A2Evidence, list[str]]:
    """검색 결과에 없는 조항 ID를 지우고 지운 ID를 돌려준다. 조문 번호·판례가 들어간 일반 정보는 버린다."""
    dropped: list[str] = []
    citations = []
    for cit in evidence.coverage_citations:
        kept = [cid for cid in cit.clause_ids if cid in retrieved_ids]
        dropped += [cid for cid in cit.clause_ids if cid not in retrieved_ids]
        citations.append(CoverageCitation(coverage=cit.coverage, clause_ids=list(dict.fromkeys(kept))))
    general = [g for g in evidence.general_info if not LAW_ARTICLE_RE.search(g)]
    cleaned = evidence.model_copy(update={"coverage_citations": citations, "general_info": general})
    return cleaned, list(dict.fromkeys(dropped))


def a2_evidence(state: dict, call: Callable, search: Callable) -> dict:
    """A2 근거 검색 노드. call은 구조화 출력 호출, search는 약관 검색 함수다(테스트에서 교체)."""
    intake: Intake = state["intake"]
    estimate: Estimate = state["estimate"]
    models = list(state.get("models_used", []))
    contexts = list(state.get("contexts", []))

    try:
        found = retrieve(build_queries(intake, estimate, state["question"]), search)
    except retriever.RetrievalFailed:
        return {"a2_evidence": A2Evidence(), "error": "retrieval_failed", "dropped_ids": []}
    if not found:
        return {"a2_evidence": A2Evidence(), "dropped_ids": []}

    clauses_text = "\n".join(f"[{h.id}] ({h.clause.coverage}/{h.clause.kind}) {h.clause.title}: {h.clause.text}"
                             for h in sorted(found.values(), key=lambda h: h.id))
    coverage_summary = ", ".join(f"{c.coverage}({'보장' if c.covered else '미보장'})" for c in estimate.coverages)
    messages = [
        SystemMessage(content=(
            "너는 약관·법률 자문가(A2)다. 아래에 검색된 가상 약관 조항만 근거로 쓴다. "
            "담보(대물·자차·대인)마다 근거가 되는 조항 ID를 고른다. 목록에 없는 조항 ID는 절대 쓰지 않는다. "
            "고객이 물었는데 조항에서 근거를 찾을 수 없는 보장이나 주제는 unconfirmed_topics에 적는다. "
            "실제 법령은 일반 원칙만 general_info에 한두 문장으로 쓰고, 조문 번호와 판례, 과실 비율이나 금액 같은 숫자는 쓰지 않는다. "
            "근거를 찾지 못한 보장은 general_info에서 설명하지 않는다. "
            "고객만 답할 수 있는 정보가 더 필요하면 customer_info_needed에 적는다."
        )),
        HumanMessage(content=(
            f"고객 질문: {state['question']}\n사고 정보: {intake.model_dump_json(exclude_none=True)}\n"
            f"담보 판정(코드): {coverage_summary}\n\n검색된 약관 조항:\n{clauses_text}"
        )),
    ]
    try:
        res = call(A2Evidence, messages)
    except llm.LLMUnavailable as exc:
        if exc.attempts and all(err in SCHEMA_ERRORS for _, err in exc.attempts):
            return {"a2_evidence": A2Evidence(), "error": "agent_failed:A2", "dropped_ids": []}
        raise
    evidence, dropped = validate_evidence(res.value, set(found))
    if dropped:
        logger.info("A2가 검색되지 않은 조항을 인용해 지웠습니다: %s", dropped)

    cited = [cid for cit in evidence.coverage_citations for cid in cit.clause_ids]
    for cid in dict.fromkeys(cited):
        clause = found[cid].clause
        contexts.append(Context(id=cid, source="약관", text=f"{clause.title}: {clause.text}"))
    return {
        "a2_evidence": evidence, "contexts": contexts, "dropped_ids": dropped,
        "models_used": models + [res.model_id],
    }


# ---------------------------------------------------------------- A3 (기업 CEO) 반론
MAX_ROUNDS = 3  # 토론 최대 라운드 (코드로 건다)
MAX_FOLLOWUP_ITEMS = 3  # 토론 뒤 되묻기 항목 수 상한
MAX_NEW_CLAIMS_PER_ROUND = 3  # 라운드당 새 주장 수 상한 (A2 검증 호출과 답변 길이를 줄인다)


def claim_key(c: Claim) -> tuple:
    """같은 주장인지 판단하는 키. LLM이 ID를 다시 쓸 거라고 가정하지 않고 코드로 판단한다."""
    return (c.kind, c.coverage, tuple(sorted(set(c.candidate_basis_ids))))


def register_claims(existing: list[Claim], proposed: list[Claim], round_no: int,
                    max_new: int = MAX_NEW_CLAIMS_PER_ROUND) -> tuple[list[Claim], list[str]]:
    """새 주장에 ID(A3-{라운드}-{순번})를 붙이고, 기존 주장과 같은 키면 기존 주장으로 합친다.

    돌려주는 값: (전체 주장 목록, 이번 라운드에 새로 생긴 주장 ID 목록)
    """
    claims = list(existing)
    by_key = {claim_key(c): c for c in claims}
    new_ids: list[str] = []
    seq = sum(1 for c in claims if c.id.startswith(f"A3-{round_no}-"))
    for p in proposed:
        key = claim_key(p)
        if key in by_key:
            continue  # 반복 주장: 기존 ID 유지, 다시 검증하지 않는다
        if len(new_ids) >= max_new:
            break  # 상한을 넘는 새 주장은 버린다 (앞쪽이 더 중요한 주장이라고 본다)
        seq += 1
        claim = p.model_copy(update={"id": f"A3-{round_no}-{seq}",
                                     "candidate_basis_ids": sorted(set(p.candidate_basis_ids))})
        claims.append(claim)
        by_key[key] = claim
        new_ids.append(claim.id)
    return claims, new_ids


def a3_basis_candidates(entry) -> str:
    """A3가 고를 수 있는 근거 후보: 이 사고 유형의 기준표 수정 요소와 약관의 면책·감액 조항."""
    lines = [f"[{m.id}] 과실조정 근거(기준표 수정 요소): {m.description} ({m.bound} {m.adjustment:+d}%p)"
             for m in (entry.modifiers if entry else [])]
    lines += [f"[{c.id}] {c.kind} 근거({c.coverage}): {c.title} - {c.text}"
              for c in tools.load_policy() if c.kind in ("면책", "감액")]
    return "\n".join(lines)


def a3_opinion(state: dict, call: Callable, round_no: int) -> dict:
    """A3 노드. 보험사 관점의 반론을 내고, 코드가 ID와 중복을 처리한다."""
    intake: Intake = state["intake"]
    estimate: Estimate = state["estimate"]
    entry = tools.fault_entry_for(intake)
    existing: list[Claim] = list(state.get("claims", []))
    prev = "\n".join(f"- {c.id} ({c.kind}/{c.coverage}, 근거 {c.candidate_basis_ids}): {c.text}" for c in existing) or "없음"
    messages = [
        SystemMessage(content=(
            "너는 보험사 경영자(A3)다. 목표는 보험사의 손해배상 비용을 최대한 줄이는 것이다. "
            "고객 과실을 높이거나, 보상을 면책·감액할 수 있는 반론을 공격적으로 찾아 낸다. "
            "근거 후보 목록을 빠짐없이 훑어 이 사고에 걸 수 있는 수정 요소와 면책·감액 조항을 모두 검토하고, "
            "비용을 많이 줄이는 주장부터 낸다. 앞 라운드 주장이 기각돼도 남은 근거 후보로 다른 각도의 반론을 다시 시도한다. "
            "네 주장은 약관·법률 자문가(A2)가 근거로 검증하므로, 근거 후보가 받쳐 주는 한 주저하지 말고 제기한다. "
            "주장마다 종류(과실조정/면책/감액/기타), 해당 담보(대물/자차/대인/공통), 근거 후보 ID를 적는다. "
            "근거 후보 ID가 없는 주장은 받지 않는다. 주장 문장(text)은 한국어로 쓰고, ID는 문장에 넣지 말고 candidate_basis_ids에만 적는다. "
            "근거 후보 ID는 아래 목록에서만 고른다. 과실조정은 기준표 수정 요소를, 면책·감액은 약관 조항을 근거로 한다. "
            "사고 정보에 없는 사실을 단정하지 말고 '~했다면'처럼 조건으로 쓴다. 숫자(비율·금액)는 쓰지 않는다. "
            "이 사고의 사실관계와 직접 관련 있는 주장만 중요한 순서로 최대 3개 낸다. "
            "반론은 고객이 받는 보상(고객 차량 수리비, 고객의 치료 관련 보상, 고객의 자차 보험금)에 대한 것만 낸다. "
            "상대 차량의 손상이나 상대 차량 탑승자에 대한 주장은 내지 않는다. "
            "사고 정보와 고객 진술(facts)에 이미 반대되는 사실이 있으면 그 주장은 내지 않는다. "
            "음주·무면허·통지 지연처럼 사고 정보에 아무 단서가 없는 일반적인 주장은 내지 않는다. "
            "이미 낸 주장은 다시 내지 않는다. 근거 후보를 다 검토해 더 낼 주장이 정말 없을 때만 claims를 비우고 no_new_issues를 true로 한다. "
            "customer_info_needed에는 네 주장이 맞는지 가리는 데 필요하고 고객만 답할 수 있는 사실만 최대 2개 적는다. "
            "고객이 이미 말한 사실(사고 정보의 facts 포함), 계약·가입 정보, 견적 금액, 상대방 쪽 정보는 묻지 않는다."
        )),
        HumanMessage(content=(
            f"사고 정보: {intake.model_dump_json(exclude_none=True)}\n"
            f"A1 추정(코드 계산): {estimate.model_dump_json()}\n"
            f"이미 낸 주장:\n{prev}\n\n근거 후보 목록:\n{a3_basis_candidates(entry)}"
        )),
    ]
    res = call(A3Opinion, messages)
    opinion: A3Opinion = res.value
    # 근거 후보 ID가 하나도 없는 주장은 검증할 수 없어 받지 않는다 (막연한 주장으로 라운드가 늘어나는 것을 막는다)
    proposed = [p for p in opinion.claims if p.candidate_basis_ids]
    if len(proposed) < len(opinion.claims):
        logger.info("A3 근거 ID 없는 주장 %d개 제외", len(opinion.claims) - len(proposed))
    claims, new_ids = register_claims(existing, proposed, round_no)
    new_claims = [c for c in claims if c.id in new_ids]
    content = "; ".join(f"{c.id} {c.kind}({c.coverage}): {c.text}" for c in new_claims) or "추가 쟁점 없음"
    return {
        "claims": claims,
        "new_claim_ids": new_ids,
        # 새 주장이 하나도 없으면 LLM 플래그와 관계없이 '추가 쟁점 없음'으로 본다(반복 주장만 낸 경우)
        "no_new_issues": opinion.no_new_issues or not new_ids,
        # 고객에게 물을 정보도 라운드를 넘어 누적한다 (다음 라운드가 빈 목록을 내도 앞 라운드 값이 남아야 한다)
        "a3_info_needed": list(dict.fromkeys([*state.get("a3_info_needed", []), *opinion.customer_info_needed])),
        "debate_log": list(state.get("debate_log", [])) + [DebateEntry(round=round_no, agent="A3", content=content)],
        "models_used": list(state.get("models_used", [])) + [res.model_id],
    }


# ---------------------------------------------------------------- A2 반론 검증
def check_verdict(claim: Claim, verdict: Verdict | None, entry) -> Verdict:
    """코드 검증 단계 (CLAUDE.md "검증 단계"). 통과하지 못하면 근거없음으로 바꾼다.

    ① 근거 ID가 실제 데이터에 있는가  ② 근거 종류가 주장 종류와 맞는가(과실조정=수정 요소, 면책·감액=약관 조항)
    ③ 약관 조항의 담보가 주장 담보와 같거나 공통인가  ④ 통과하면 A2의 확인/기각을 그대로 쓴다
    추가 규칙: 수정 요소는 이 사고 유형의 것이어야 한다. 판정이 없거나, 확인인데 근거 ID가 없으면 근거없음이다.
    """
    def no_basis(reason: str) -> Verdict:
        return Verdict(claim_id=claim.id, verdict="근거없음", evidence_ids=[], reason=reason)

    if verdict is None:
        return no_basis("A2 판정이 없음")
    if verdict.verdict == "근거없음":
        return verdict.model_copy(update={"claim_id": claim.id})
    if not verdict.evidence_ids:
        return no_basis("근거 ID가 없음")
    entry_modifiers = {m.id for m in entry.modifiers} if entry else set()
    for eid in verdict.evidence_ids:
        if eid not in tools.all_data_ids():
            return no_basis(f"없는 근거 ID: {eid}")  # ①
        modifier, clause = tools.get_modifier(eid), tools.get_clause(eid)
        if claim.kind == "과실조정":
            if modifier is None:
                return no_basis(f"과실조정 주장의 근거가 수정 요소가 아님: {eid}")  # ②
            if eid not in entry_modifiers:
                return no_basis(f"이 사고 유형의 수정 요소가 아님: {eid}")
        elif claim.kind in ("면책", "감액"):
            if clause is None:
                return no_basis(f"{claim.kind} 주장의 근거가 약관 조항이 아님: {eid}")  # ②
            if clause.coverage not in (claim.coverage, "공통"):
                return no_basis(f"담보 불일치: 주장 {claim.coverage}, 조항 {clause.coverage}")  # ③
    return verdict.model_copy(update={"claim_id": claim.id})  # ④


def evidence_texts(claims: list[Claim], entry) -> str:
    """A2가 검증할 때 보는 근거 원문: 주장의 근거 후보와 이 사고 유형의 수정 요소, 약관 면책·감액 조항."""
    ids = {b for c in claims for b in c.candidate_basis_ids}
    ids |= {m.id for m in (entry.modifiers if entry else [])}
    ids |= {c.id for c in tools.load_policy() if c.kind in ("면책", "감액")}
    lines = []
    for eid in sorted(ids):
        m, c = tools.get_modifier(eid), tools.get_clause(eid)
        if m:
            lines.append(f"[{eid}] 기준표 수정 요소: {m.description} ({m.bound} {m.adjustment:+d}%p)")
        elif c:
            lines.append(f"[{eid}] 약관 {c.kind}({c.coverage}): {c.title} - {c.text}")
    return "\n".join(lines)


def a2_review(state: dict, call: Callable, round_no: int) -> dict:
    """A2 반론 검증 노드. 이번 라운드의 새 주장에만 판정을 돌린다(이전 판정은 다시 쓰지 않는다)."""
    claims: list[Claim] = list(state.get("claims", []))
    new_ids: list[str] = list(state.get("new_claim_ids", []))
    verdicts: dict[str, Verdict] = dict(state.get("verdicts", {}))
    models = list(state.get("models_used", []))
    targets = [c for c in claims if c.id in new_ids and c.id not in verdicts]
    if not targets:
        return {"verdicts": verdicts}

    intake: Intake = state["intake"]
    entry = tools.fault_entry_for(intake)
    listing = "\n".join(f"- {c.id} ({c.kind}/{c.coverage}, 근거 후보 {c.candidate_basis_ids}): {c.text}" for c in targets)
    messages = [
        SystemMessage(content=(
            "너는 약관·법률 자문가(A2)다. 보험사 측(A3)의 주장마다 아래 근거 원문만 보고 판정한다. "
            "확인: 근거가 있고, 근거 내용이 이 사고에 적용될 수 있어 주장이 타당함 / "
            "기각: 근거를 보면 주장이 틀렸거나 이 사고에 적용되지 않음 / 근거없음: 관련 근거를 찾지 못함. "
            "사고 정보와 고객 진술(facts)에 반대되는 조건을 가정한 주장은 기각한다 "
            "(예: 고객이 신호 대기로 정차해 있었다고 했는데 '급정거했다면'을 가정한 주장). "
            "상대 차량의 손상이나 상대 탑승자에 관한 주장은 고객이 받는 보상과 무관하므로 기각한다. "
            "evidence_ids에는 판정의 근거가 된 ID만 목록에서 고른다. 모든 주장에 하나씩 판정을 낸다. 숫자는 쓰지 않는다."
        )),
        HumanMessage(content=(
            f"사고 정보: {intake.model_dump_json(exclude_none=True)}\n\n판정할 주장:\n{listing}\n\n"
            f"근거 원문:\n{evidence_texts(targets, entry)}"
        )),
    ]
    error = state.get("error")
    try:
        res = call(A2Review, messages)
        by_claim = {v.claim_id: v for v in res.value.verdicts}
        models.append(res.model_id)
    except llm.LLMUnavailable as exc:
        if not (exc.attempts and all(err in SCHEMA_ERRORS for _, err in exc.attempts)):
            raise
        by_claim, error = {}, "agent_failed:A2-review"  # 검증 실패: 이번 라운드의 새 주장은 모두 근거없음

    log = []
    for c in targets:
        v = check_verdict(c, by_claim.get(c.id), entry)
        verdicts[c.id] = v
        log.append(f"{c.id} {v.verdict}" + (f"({', '.join(v.evidence_ids)})" if v.evidence_ids else "")
                   + (f" - {v.reason}" if v.reason else ""))
    return {
        "verdicts": verdicts, "error": error, "models_used": models,
        "debate_log": list(state.get("debate_log", [])) + [DebateEntry(round=round_no, agent="A2-review", content="; ".join(log))],
    }


# ---------------------------------------------------------------- 종합 (코드)
# 주장 문장 안에 LLM이 적은 근거 ID (‑ 같은 다른 하이픈도 잡는다)
TEXT_ID_RE = re.compile(r"(?<![A-Za-z])(?:(?:ART|SP|DOC)[-‐‑–]\d+|FT[-‐‑–]\d+(?:[-‐‑–]M\d+)?)(?!\d)")
# ID를 지운 뒤 남는 '(근거 )', '에 따라' 같은 군더더기
_ID_LEFTOVER_RE = re.compile(r"\(\s*(?:근거\s*)?[,\s]*\)|(?<![가-힣])(?:에\s*따라|에\s*의해|에\s*의거하여|상)(?=[\s,.])")
HANGUL_RE = re.compile(r"[가-힣]")


def clean_claim_text(text: str, allowed: set[str]) -> tuple[str | None, list[str]]:
    """고객에게 보일 주장 문장에서 contexts에 없는 ID를 지운다. 한국어가 아니거나 숫자가 든 문장은 버린다(None).

    돌려주는 값: (정리한 문장 또는 None, 지운 ID 목록)
    """
    dropped: list[str] = []

    def repl(m: re.Match) -> str:
        norm = re.sub(r"[‐‑–]", "-", m.group(0))
        if norm in allowed:
            return norm
        dropped.append(norm)
        return ""

    out = TEXT_ID_RE.sub(repl, text)
    if dropped:
        out = _ID_LEFTOVER_RE.sub("", out)
    out = re.sub(r"\s+([,.])", r"\1", re.sub(r"\s{2,}", " ", out)).strip(" ,")
    bare = re.sub(r"A3-\d+-\d+", "", TEXT_ID_RE.sub("", out))  # 남겨 둔 근거 ID와 주장 ID의 숫자는 빼고 본다
    if not HANGUL_RE.search(out) or re.search(r"\d|%", bare):  # 영어 문장, LLM이 쓴 숫자는 내보내지 않는다
        return None, dropped
    return out, dropped


def synthesize(state: dict) -> dict:
    """토론 결과를 종합한다. LLM을 쓰지 않는다 (CLAUDE.md "책임 구분").

    - 확인된 과실조정: 수정 요소로 과실 범위를 조정하고 담보별 보험금을 다시 계산한다
    - 확인된 면책·감액: 금액은 그대로 두고 review_notes에 넣는다
    - 확인된 기타: 반영하지 않는다(토론 기록에만 남음)
    - 근거없음: unresolved_issues('확인되지 않은 사항')에 넣는다. 기각은 반영하지 않는다
    """
    intake: Intake = state["intake"]
    base: Estimate = state["estimate"]
    entry = tools.fault_entry_for(intake)
    customer = tools.get_customer(state.get("customer_id"))
    claims: list[Claim] = state.get("claims", [])
    verdicts: dict[str, Verdict] = state.get("verdicts", {})

    kept, dropped, notes, unresolved, modifiers = [], [], [], [], []
    for c in claims:
        v = verdicts.get(c.id)
        verdict = v.verdict if v else "근거없음"
        if verdict == "확인" and c.kind == "과실조정":
            kept.append(c)
            for e in v.evidence_ids:
                m = tools.get_modifier(e)
                if m and m.id not in {x.id for x in modifiers}:  # 같은 수정 요소는 한 번만 더한다
                    modifiers.append(m)
        elif verdict == "확인" and c.kind in ("면책", "감액"):
            kept.append(c)
            notes.append(ReviewNote(claim_id=c.id, text=c.text, evidence_ids=v.evidence_ids))
        else:
            dropped.append(c)
            if verdict == "근거없음":
                unresolved.append(c.text)

    fault = tools.adjust_fault(entry.customer_fault, modifiers) if modifiers else base.customer_fault
    estimate = base
    if fault != base.customer_fault:
        estimate = base.model_copy(update={
            "customer_fault": fault,
            "coverages": tools.estimate_coverages(customer, entry, intake, customer_fault=fault),
        })

    # 답변에 새로 나오는 근거 ID(수정 요소, 검토 사항의 조항)를 contexts에 넣는다
    contexts = list(state.get("contexts", []))
    have = {c.id for c in contexts}
    for m in modifiers:
        if m.id not in have:
            contexts.append(Context(id=m.id, source="기준표",
                                    text=f"수정 요소: {m.description} ({m.adjustment:+d}%p)"))
            have.add(m.id)
    for n in notes:
        for eid in n.evidence_ids:
            clause = tools.get_clause(eid)
            if clause and eid not in have:
                contexts.append(Context(id=eid, source="약관", text=f"{clause.title}: {clause.text}"))
                have.add(eid)

    # 답변에 그대로 나가는 주장 문장에서 contexts에 없는 ID를 지우고 trace.dropped_ids에 남긴다
    dropped_ids = list(state.get("dropped_ids", []))
    cleaned_notes, cleaned_unresolved = [], []
    for n in notes:
        text, gone = clean_claim_text(n.text, have)
        dropped_ids += gone
        cleaned_notes.append(n.model_copy(update={"text": text or guards.REVIEW_NOTE_FALLBACK}))
    for u in unresolved:
        text, gone = clean_claim_text(u, have)
        dropped_ids += gone
        if text:
            cleaned_unresolved.append(text)
        else:
            logger.info("한국어가 아닌 확인되지 않은 사항 문장 제외")
    notes, unresolved = cleaned_notes, list(dict.fromkeys(cleaned_unresolved))
    dropped_ids = list(dict.fromkeys(dropped_ids))

    # 되물을 정보는 A3 주장을 판정하는 데 필요한 사실만 쓴다 (A2 검색 단계의 일반적인 요청은 쓰지 않는다)
    info = list(dict.fromkeys(state.get("a3_info_needed", [])))[:MAX_FOLLOWUP_ITEMS]
    synthesis = Synthesis(
        estimate=estimate, kept_claims=kept, dropped_claims=dropped, review_notes=notes,
        customer_info_needed=info, unresolved_issues=unresolved, context_ids=[c.id for c in contexts],
    )
    return {"synthesis": synthesis, "estimate": estimate, "contexts": contexts,
            "applied_modifiers": [m.id for m in modifiers], "dropped_ids": dropped_ids}
