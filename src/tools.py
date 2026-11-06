"""도메인 도구 (순수 파이썬). 계산과 규칙 판정은 모두 여기서 코드로 한다.

- 가상 데이터 로드 (T1-4)
- 과실 조회와 대물 계산 (T1-5). 계산식은 SERVICE.md 4장 "보험금 추정"의 교육용 가상 규칙이다.
- 필수 접수 항목 누락 판정 (T2-1)
- 자차·대인 계산과 담보별 보장 판정 (T3-2)
- 과실 조정 (T4-4)
- 청구 저장소: 청구번호 발급, 중복 방지, 본인 청구만 조회 (T6-1)
- 서류 체크리스트 (T6-2)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from pydantic import TypeAdapter

logger = logging.getLogger(__name__)

from src.schemas import (
    ClaimRecord,
    CoverageAmount,
    CoverageEstimate,
    Customer,
    DamageTable,
    DocumentRule,
    FaultModifier,
    FaultTableEntry,
    Intake,
    PercentRange,
    PolicyClause,
    Range,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _read_json(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 데이터 로드
@lru_cache(maxsize=None)
def load_customers() -> tuple[Customer, ...]:
    return tuple(TypeAdapter(list[Customer]).validate_python(_read_json("customers.json")))


@lru_cache(maxsize=None)
def load_fault_table() -> tuple[FaultTableEntry, ...]:
    return tuple(TypeAdapter(list[FaultTableEntry]).validate_python(_read_json("fault_table.json")))


@lru_cache(maxsize=None)
def load_damage_table() -> DamageTable:
    return DamageTable.model_validate(_read_json("damage_table.json"))


@lru_cache(maxsize=None)
def load_policy() -> tuple[PolicyClause, ...]:
    return tuple(TypeAdapter(list[PolicyClause]).validate_python(_read_json("policy.json")))


@lru_cache(maxsize=None)
def load_documents() -> tuple[DocumentRule, ...]:
    return tuple(TypeAdapter(list[DocumentRule]).validate_python(_read_json("documents.json")))


def get_clause(clause_id: str) -> PolicyClause | None:
    return next((c for c in load_policy() if c.id == clause_id), None)


def get_modifier(modifier_id: str):
    for entry in load_fault_table():
        for m in entry.modifiers:
            if m.id == modifier_id:
                return m
    return None


def get_customer(customer_id: str | None) -> Customer:
    """고객을 찾는다. customer_id가 없으면 기본 고객(첫 번째 고객)을 돌려준다."""
    customers = load_customers()
    if not customer_id:
        return customers[0]
    for c in customers:
        if c.id == customer_id:
            return c
    raise KeyError(f"고객을 찾을 수 없습니다: {customer_id}")


def all_data_ids() -> set[str]:
    """근거로 인정되는 데이터 ID 전체: 약관 조항, 기준표 항목, 수정 요소."""
    ids: set[str] = {c.id for c in load_policy()}
    for entry in load_fault_table():
        ids.add(entry.id)
        ids.update(m.id for m in entry.modifiers)
    return ids


def validate_data() -> dict:
    """데이터 무결성 검사. 문제가 있으면 ValueError를 낸다. 요약을 돌려준다."""
    policy, table, customers, docs = load_policy(), load_fault_table(), load_customers(), load_documents()
    all_ids = [c.id for c in policy] + [e.id for e in table] + [m.id for e in table for m in e.modifiers] \
        + [c.id for c in customers] + [d.id for d in docs]
    dup = {i for i in all_ids if all_ids.count(i) > 1}
    if dup:
        raise ValueError(f"중복 ID: {sorted(dup)}")
    names = [n for e in table for n in [e.accident_type, *e.aliases]]
    dup_names = {n for n in names if names.count(n) > 1}
    if dup_names:
        raise ValueError(f"기준표 유형·별칭 중복: {sorted(dup_names)}")
    clause_ids = {c.id for c in policy}
    for c in customers:
        missing = [r for r in c.riders if r not in clause_ids]
        if missing:
            raise ValueError(f"{c.id} 특약 조항 없음: {missing}")
    plans = {tuple(sorted((k, v.joined, v.limit) for k, v in c.coverages.items())) for c in customers}
    if len(plans) != len(customers):
        raise ValueError("고객 담보 구성이 서로 같은 고객이 있습니다")
    table_ids = {e.id for e in table}
    for d in docs:
        bad = [i for i in d.fault_table_ids if i not in table_ids]
        if bad:
            raise ValueError(f"{d.id} 서류 규칙의 기준표 ID 없음: {bad}")
    return {
        "약관 조항": len(policy), "기준표 항목": len(table),
        "수정 요소": sum(len(e.modifiers) for e in table), "고객": len(customers), "서류 규칙": len(docs),
    }


# ---------------------------------------------------------------- 접수 필수 항목
# SERVICE.md 4장 "접수와 되묻기": 필수 항목은 사고 유형, 일시, 장소, 부상 여부.
REQUIRED_FIELDS: dict[str, str] = {
    "accident_type": "사고 유형",
    "occurred_at": "사고 일시",
    "location": "사고 장소",
    "injury": "부상 여부",
}


def missing_fields(intake: Intake) -> list[str]:
    """비어 있는 필수 항목의 필드 이름을 REQUIRED_FIELDS 순서대로 돌려준다.

    부상 여부는 'none'(부상 없음)도 답한 것으로 본다. 값이 None이거나 공백뿐이면 누락이다.
    """
    missing = []
    for field in REQUIRED_FIELDS:
        value = getattr(intake, field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def missing_labels(fields: list[str]) -> list[str]:
    """필드 이름을 고객에게 보여줄 한국어 이름으로 바꾼다."""
    return [REQUIRED_FIELDS[f] for f in fields]


# ---------------------------------------------------------------- 청구 저장소 (T6-1)
CLAIMS_PATH = DATA_DIR / "claims.json"  # 테스트에서 임시 경로로 바꾼다
LOCK_TIMEOUT = 10.0  # 초
_process_lock = threading.Lock()


class StoreError(Exception):
    """청구 저장소 읽기·쓰기 실패. trace.error는 store_failed."""

    code = "store_failed"


@contextmanager
def _file_lock(path: Path):
    """잠금 파일(<저장소>.lock)을 O_EXCL로 만들어 다른 프로세스와 겹쳐 쓰지 않게 한다. 같은 프로세스 안은 threading.Lock."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _process_lock:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise StoreError("청구 저장소 잠금 대기 시간 초과")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            os.remove(lock_path)


def _read_claims(path: Path) -> list[ClaimRecord]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return TypeAdapter(list[ClaimRecord]).validate_python(json.load(f))


def _write_claims_atomic(path: Path, claims: list[ClaimRecord]) -> None:
    """임시 파일에 다 쓴 뒤 os.replace로 바꿔 넣는다. 쓰다가 실패해도 기존 파일은 그대로다."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([c.model_dump(mode="json") for c in claims], f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _next_claim_no(claims: list[ClaimRecord], day: str) -> str:
    prefix = f"CLM-{day}-"
    seqs = [int(c.claim_no[len(prefix):]) for c in claims if c.claim_no.startswith(prefix)]
    return f"{prefix}{max(seqs, default=0) + 1:03d}"


def save_claim(customer_id: str, thread_id: str, accident_summary: str, *, missing_fields: list[str] | None = None,
               customer_fault: PercentRange | None = None, coverages: list[CoverageAmount] | None = None,
               documents: list[str] | None = None, now: datetime | None = None,
               path: Path | None = None) -> tuple[ClaimRecord, bool]:
    """청구를 저장하고 (청구 기록, 새로 만들었는지)를 돌려준다.

    청구번호 발급과 저장은 같은 잠금 안에서 한다. 같은 고객·같은 thread_id의 청구가 있으면 새로 만들지 않는다.
    """
    path = path or CLAIMS_PATH
    now = now or datetime.now()
    try:
        with _file_lock(path):
            claims = _read_claims(path)
            for c in claims:
                if c.thread_id == thread_id and c.customer_id == customer_id:
                    return c, False  # 한 대화에서 청구는 1건 (SERVICE.md 4장 "중복 방지")
            record = ClaimRecord(
                claim_no=_next_claim_no(claims, now.strftime("%Y%m%d")), customer_id=customer_id, thread_id=thread_id,
                created_at=now.isoformat(timespec="seconds"), status="접수", accident_summary=accident_summary,
                missing_fields=missing_fields or [], customer_fault=customer_fault, coverages=coverages or [],
                documents=documents or [],
            )
            _write_claims_atomic(path, [*claims, record])
            return record, True
    except StoreError:
        raise
    except Exception as exc:
        raise StoreError(type(exc).__name__) from exc


def get_claim(claim_no: str | None, customer_id: str, path: Path | None = None) -> ClaimRecord | None:
    """본인 청구만 돌려준다. 다른 고객의 번호나 없는 번호는 똑같이 None이다(호출한 쪽이 고정 문구로 답한다)."""
    if not claim_no:
        return None
    try:
        claims = _read_claims(path or CLAIMS_PATH)
    except Exception as exc:
        raise StoreError(type(exc).__name__) from exc
    for c in claims:
        if c.claim_no == claim_no.strip().upper() and c.customer_id == customer_id:
            return c
    return None


# ---------------------------------------------------------------- 서류 체크리스트 (T6-2)
def required_documents(fault_table_id: str | None, injury: str | None) -> tuple[list[str], list[str]]:
    """사고 유형(기준표 ID)과 부상 여부로 필요 서류를 돌려준다: (필요 서류, 조건부 서류).

    - always: 항상 필요  - when_injury: 부상이 있으면 필요  - fault_table_ids: 해당 사고 유형이면 필요
    - 부상 여부를 모르면(None) 부상 서류는 '부상이 있다면 필요'한 조건부 서류로 따로 돌려준다
    """
    required, conditional = [], []
    for rule in load_documents():
        if rule.always or (fault_table_id and fault_table_id in rule.fault_table_ids):
            required.append(rule.name)
        elif rule.when_injury:
            if injury in ("minor", "severe"):
                required.append(rule.name)
            elif injury is None:
                conditional.append(rule.name)
    return required, conditional


# ---------------------------------------------------------------- 과실 조회
def _norm(text: str) -> str:
    return "".join(text.split())


def find_fault_entry(accident_type: str | None) -> FaultTableEntry | None:
    """사고 유형(또는 별칭)으로 기준표 항목을 찾는다. 공백은 무시한다. 없으면 None."""
    if not accident_type:
        return None
    key = _norm(accident_type)
    for entry in load_fault_table():
        names = [entry.accident_type, *entry.aliases]
        if any(_norm(n) == key for n in names):
            return entry
    # 정확히 같지 않으면 포함 관계로 한 번 더 찾는다 (예: "후미추돌 사고").
    # 여러 항목이 걸리면 가장 길게 일치한(더 구체적인) 이름의 항목을 고른다. 기준표 순서에 기대지 않는다.
    hits = [(len(_norm(n)), entry) for entry in load_fault_table()
            for n in [entry.accident_type, *entry.aliases] if _norm(n) in key]
    return max(hits, key=lambda h: h[0])[1] if hits else None


def role_matches(entry_role: str, customer_role: str | None) -> bool:
    """사고 방향 검사. 방향을 모르면(None) 검사하지 않고, 쌍방 유형은 어느 방향이든 받는다.
    가해·피해 유형은 접수 방향이 같을 때만 받는다 (예: 주차해 둔 내 차가 긁힘(피해)에 FT-11(가해, 100%)을 쓰지 않음)."""
    return customer_role is None or entry_role == "쌍방" or entry_role == customer_role


def fault_entry_for(intake) -> FaultTableEntry | None:
    """접수 정보로 기준표 항목을 찾는다. 유형 이름이 맞아도 사고 방향이 반대면 None(맞는 유형 없음)이다."""
    entry = find_fault_entry(intake.accident_type)
    if entry is None or role_matches(entry.customer_role, intake.customer_role):
        return entry
    logger.info("사고 방향 불일치로 기준표 항목을 쓰지 않음: %s(%s) / 접수 방향 %s",
                entry.id, entry.customer_role, intake.customer_role)
    return None


def opponent_fault(customer_fault: PercentRange) -> PercentRange:
    """상대방 과실 범위 = (100 − 고객 상한, 100 − 고객 하한)."""
    return PercentRange(min=100 - customer_fault.max, max=100 - customer_fault.min)


# ---------------------------------------------------------------- 담보별 계산
def _apply_ratio(amount: Range, ratio: PercentRange, limit: int | None) -> Range:
    """하한 = 최소 금액 × 최소 비율, 상한 = 최대 금액 × 최대 비율. 원 단위 내림, 한도 상한."""
    low = amount.min * ratio.min // 100
    high = amount.max * ratio.max // 100
    if limit is not None:
        low, high = min(low, limit), min(high, limit)
    return Range(min=low, max=high)


def calc_property(vehicle_damage: Range, customer_fault: PercentRange, opponent_limit: int | None) -> Range:
    """대물: 상대방 보험에서 받는 내 차 수리비 = 차량 손해액 × 상대방 과실. 상대방 대물 한도를 넘지 않는다."""
    return _apply_ratio(vehicle_damage, opponent_fault(customer_fault), opponent_limit)


def calc_own_damage(vehicle_damage: Range, customer_fault: PercentRange, deductible: int, limit: int | None) -> Range:
    """자차: 내 보험에서 받는 내 과실분 = 차량 손해액 × 고객 과실 − 자기부담금(0 미만이면 0). 자차 한도를 넘지 않는다."""
    share = _apply_ratio(vehicle_damage, customer_fault, None)
    low, high = max(0, share.min - deductible), max(0, share.max - deductible)
    if limit is not None:
        low, high = min(low, limit), min(high, limit)
    return Range(min=low, max=high)


def calc_bodily(bodily_fixed: Range, customer_fault: PercentRange, opponent_limit: int | None) -> Range:
    """대인: 상대방 보험에서 받는 보상 = 대인 가상 정액 × 상대방 과실. 상대방 대인 한도를 넘지 않는다."""
    return _apply_ratio(bodily_fixed, opponent_fault(customer_fault), opponent_limit)


# ---------------------------------------------------------------- 과실 조정 (T4-4)
MAX_ADJUST_PER_BOUND = 20  # 한쪽 끝의 조정 합계 상한(%p). SERVICE.md 4장 "과실 조정"


def adjust_fault(base: PercentRange, modifiers: list[FaultModifier]) -> PercentRange:
    """확인된 과실조정의 수정 요소를 '가능성'으로 보고 범위를 넓힌다 (SERVICE.md 4장).

    A3의 과실조정 주장은 "~했다면" 같은 조건부라서, 확인돼도 사실로 정해진 것이 아니다.
    그래서 범위를 좁히지 않고 넓히기만 한다 (bound 필드는 쓰지 않는다).
    ① +값은 모두 더해 상한을 올리고, −값은 모두 더해 하한을 내린다  ② 각 합계는 20%p로 제한한다
    ③ 0~100으로 자른다
    """
    up = min(MAX_ADJUST_PER_BOUND, sum(m.adjustment for m in modifiers if m.adjustment > 0))
    down = min(MAX_ADJUST_PER_BOUND, -sum(m.adjustment for m in modifiers if m.adjustment < 0))
    low = max(0, base.min - down)
    high = min(100, base.max + up)
    return PercentRange(min=low, max=high)


# ---------------------------------------------------------------- 금액 표시
def won(n: int) -> str:
    """금액 표시. 만 원 단위로 떨어지면 '45만 원', 아니면 '864,196원'."""
    return f"{n // 10000:,}만 원" if n % 10000 == 0 and n >= 10000 else f"{n:,}원"


def money_range(lo: int, hi: int) -> str:
    return won(lo) if lo == hi else f"{won(lo)} ~ {won(hi)}"


DAMAGE_LABELS = {"minor": "경미", "moderate": "보통", "severe": "심각"}
INJURY_LABELS = {"minor": "경상", "severe": "중상"}


# ---------------------------------------------------------------- 담보별 보장 판정
def _by_level(prefix: str, table: dict, labels: dict[str, str], calc) -> str:
    """정도를 모를 때 정도별 범위를 조건 문장으로 만든다. 피해 정도와 부상 정도는 키가 겹치므로 이름표를 따로 받는다."""
    parts = [f"{labels[level]} {money_range(r.min, r.max)}" for level, r in ((lv, calc(v)) for lv, v in table.items())]
    return prefix + " " + " / ".join(parts)


def estimate_coverages(customer: Customer, entry: FaultTableEntry, intake: Intake,
                       customer_fault: PercentRange | None = None) -> list[CoverageEstimate]:
    """대물·자차·대인의 보장 여부와 금액 범위를 판정한다. customer_fault를 주면 그 범위로 계산한다(과실 조정 뒤)."""
    fault = customer_fault or entry.customer_fault
    table = load_damage_table()
    opp = opponent_fault(fault)
    basis = [entry.id]
    out: list[CoverageEstimate] = []

    # 대물: 상대방 과실이 있으면 상대방 대물로 받는다
    limit = table.opponent_limits.get("대물")
    if opp.max == 0:
        out.append(CoverageEstimate(coverage="대물", covered=False, condition="상대방 과실이 없어 받을 금액이 없습니다", basis_ids=basis))
    elif intake.damage_level:
        out.append(CoverageEstimate(coverage="대물", covered=True, basis_ids=basis,
                                    amount=calc_property(table.vehicle_damage[intake.damage_level], fault, limit)))
    else:
        out.append(CoverageEstimate(coverage="대물", covered=True, basis_ids=basis, condition=_by_level(
            "피해 정도에 따라", table.vehicle_damage, DAMAGE_LABELS, lambda d: calc_property(d, fault, limit))))

    # 자차: 가입한 경우만, 고객 과실분에서 자기부담금을 뺀다
    term = customer.coverages.get("자차")
    if not term or not term.joined:
        out.append(CoverageEstimate(coverage="자차", covered=False, condition="자차 미가입으로 보장되지 않습니다", basis_ids=basis))
    elif fault.max == 0:
        out.append(CoverageEstimate(coverage="자차", covered=True, amount=Range(min=0, max=0), basis_ids=basis,
                                    condition="고객 과실이 없어 자차로 받을 금액이 없습니다"))
    elif intake.damage_level:
        out.append(CoverageEstimate(coverage="자차", covered=True, basis_ids=basis, amount=calc_own_damage(
            table.vehicle_damage[intake.damage_level], fault, term.deductible, term.limit)))
    else:
        out.append(CoverageEstimate(coverage="자차", covered=True, basis_ids=basis, condition=_by_level(
            "피해 정도에 따라", table.vehicle_damage, DAMAGE_LABELS, lambda d: calc_own_damage(d, fault, term.deductible, term.limit))))

    # 대인: 부상이 있는 경우만, 상대방 대인으로 받는다
    limit = table.opponent_limits.get("대인")
    if intake.injury == "none":
        out.append(CoverageEstimate(coverage="대인", covered=False, condition="부상이 없어 해당하지 않습니다", basis_ids=basis))
    elif opp.max == 0:
        out.append(CoverageEstimate(coverage="대인", covered=False, condition="상대방 과실이 없어 받을 금액이 없습니다", basis_ids=basis))
    elif intake.injury in table.bodily_fixed:
        out.append(CoverageEstimate(coverage="대인", covered=True, basis_ids=basis,
                                    amount=calc_bodily(table.bodily_fixed[intake.injury], fault, limit)))
    else:
        out.append(CoverageEstimate(coverage="대인", covered=True, basis_ids=basis, condition=_by_level(
            "부상이 있다면", table.bodily_fixed, INJURY_LABELS, lambda d: calc_bodily(d, fault, limit))))
    return out


if __name__ == "__main__":
    # 확인 스크립트: python -m src.tools
    for c in load_customers():
        print(f"고객 {c.id} {c.name} | {c.plan} | 자차 자기부담금 {c.coverages['자차'].deductible:,}원")
    for e in load_fault_table():
        f = e.customer_fault
        print(f"기준표 {e.id} {e.accident_type} | 고객 과실 {f.min}~{f.max}%")
        for m in e.modifiers:
            print(f"  수정 요소 {m.id} | {m.description} | {m.bound} {m.adjustment:+d}%p")
    d = load_damage_table()
    for level, r in d.vehicle_damage.items():
        print(f"차량 손해액 {level}: {r.min:,}~{r.max:,}원")
    print("상대방 한도:", {k: f"{v:,}원" for k, v in d.opponent_limits.items()})
