"""평가 실행기. evaluation/test_queries.csv를 읽어 evaluation/report.md를 쓴다.

실행: python -m src.evaluate          (전체 실행, 실제 Bedrock + 실제 MCP, 약 7분)
      python -m src.evaluate --check  (평가셋 형식 검증만)

채점 (SERVICE.md 5장, CLAUDE.md 평가셋 형식):
- 코드 채점: route, expected_context_ids, must_include / must_not_include, A3 주장 필터(kept는 확인 판정만)
- 필수 0건: 개인정보 노출, 데이터에 없거나 contexts에 없는 근거 ID, 추정 답변의 면책 문구 누락, 3라운드 초과
- LLM 심사(토론 문항만): 근거 인용의 적절성 1~5점, 3점 이상 통과 (교육 코드 day7_practice/llm_judge.py 방식)
- 합격: 전체 8/10 이상 + 토론 3/4·직접 처리 2/2·되묻기 1/1 + 필수 0건 위반 0
토론 문항의 첫 응답이 되묻기면 같은 대화에 CONTINUE_TEXT를 한 번 이어 보내 최종 답변으로 채점한다.
평가 중 생기는 청구는 임시 저장소(샘플 복사본)에 저장한다. 실제 data/claims.json은 바뀌지 않는다.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "evaluation" / "test_queries.csv"
REPORT = ROOT / "evaluation" / "report.md"

COLUMNS = ["id", "type", "question", "customer_id", "expected_route", "expected_context_ids",
           "must_include", "must_not_include"]
TYPE_COUNTS = {"debate": 4, "direct": 2, "refuse": 3, "followup": 1}  # SERVICE.md 5장 배분
TYPE_LABELS = {"debate": "사고 평가(토론)", "direct": "직접 처리", "refuse": "거절", "followup": "되묻기"}
TYPE_MIN_PASS = {"debate": 3, "direct": 2, "followup": 1}  # 유형별 합격 기준 (거절은 전체·필수 0건으로 본다)
TOTAL_MIN_PASS = 8
ROUTES = {"debate", "direct", "followup", "refused"}
CONTINUE_TEXT = "추가로 말씀드릴 내용은 없어요."
JUDGE_PASS = 3
ID_RE = re.compile(r"\b(?:ART|SP)-\d+\b|\bFT-\d+(?:-M\d+)?\b")


@dataclass
class Query:
    id: str
    type: str
    question: str
    customer_id: str
    expected_route: str
    expected_context_ids: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 평가셋 로드·검증 (T8-1)
def _split(value: str) -> list[str]:
    """목록 값은 ';'로 구분하고, 없으면 빈칸이다 (CLAUDE.md 평가셋 형식)."""
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def load_queries(path: Path = QUERIES) -> list[Query]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"컬럼이 규약과 다릅니다: {reader.fieldnames} (규약: {COLUMNS})")
        rows = list(reader)
    return [Query(id=r["id"], type=r["type"], question=r["question"], customer_id=r["customer_id"],
                  expected_route=r["expected_route"], expected_context_ids=_split(r["expected_context_ids"]),
                  must_include=_split(r["must_include"]), must_not_include=_split(r["must_not_include"]))
            for r in rows]


def validate(queries: list[Query]) -> dict[str, int]:
    """형식 검증. 문제가 있으면 ValueError. 유형별 문항 수를 돌려준다."""
    from src import tools

    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("문항 ID가 중복됩니다")
    customers = {c.id for c in tools.load_customers()}
    data_ids = tools.all_data_ids()
    for q in queries:
        if q.type not in TYPE_COUNTS:
            raise ValueError(f"{q.id}: 알 수 없는 type {q.type}")
        if q.expected_route not in ROUTES:
            raise ValueError(f"{q.id}: 알 수 없는 expected_route {q.expected_route}")
        if q.customer_id not in customers:
            raise ValueError(f"{q.id}: 없는 고객 {q.customer_id}")
        if not q.question.strip():
            raise ValueError(f"{q.id}: 질문이 비어 있습니다")
        bad = [i for i in q.expected_context_ids if i not in data_ids]
        if bad:
            raise ValueError(f"{q.id}: 데이터에 없는 근거 ID {bad}")
    counts = {t: sum(q.type == t for q in queries) for t in TYPE_COUNTS}
    if counts != TYPE_COUNTS:
        raise ValueError(f"유형별 문항 수가 배분과 다릅니다: {counts} (배분: {TYPE_COUNTS})")
    return counts


# ---------------------------------------------------------------- 채점 (코드)
def must_zero_violations(resp: dict) -> list[str]:
    """필수 0건 항목 위반 목록. 응답 하나(중간 턴 포함)에 대해 검사한다."""
    from src import guards, tools

    answer, trace = resp["answer"], resp["trace"]
    out = []
    if guards.find_pii(answer):
        out.append(f"개인정보 노출({', '.join(guards.find_pii(answer))})")
    ctx_ids = {c["id"] for c in resp["contexts"]}
    data_ids = tools.all_data_ids()
    for i in sorted(set(ID_RE.findall(answer))):
        if i not in data_ids:
            out.append(f"없는 근거 ID {i}")
        elif i not in ctx_ids:
            out.append(f"contexts에 없는 근거 ID {i}")
    if "[추정 결과]" in answer and not answer.rstrip().endswith(guards.DISCLAIMER):
        out.append("면책 문구 누락")
    if trace.get("rounds", 0) > 3:
        out.append(f"토론 {trace['rounds']}라운드(3 초과)")
    return out


def code_checks(q: Query, final: dict) -> dict[str, bool]:
    answer, trace = final["answer"], final["trace"]
    ctx_ids = {c["id"] for c in final["contexts"]}
    checks = {f"route={q.expected_route}": trace.get("route") == q.expected_route}
    for i in q.expected_context_ids:
        checks[f"근거 {i}"] = i in ctx_ids
    for s in q.must_include:
        checks[f"포함 '{s}'"] = s in answer
    for s in q.must_not_include:
        checks[f"미포함 '{s}'"] = s not in answer
    if trace.get("claims"):
        checks["A3 주장 필터(kept=확인만)"] = all(c["verdict"] == "확인" for c in trace["claims"] if c["kept"])
    return checks


# ---------------------------------------------------------------- LLM 심사 (토론 문항)
class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    problems: list[str] = Field(default_factory=list, description="근거와 맞지 않거나 근거 없는 보장 안내")
    reasoning: str = Field(default="", description="판단 근거 한두 문장")


RUBRIC = """자동차 사고 보험 상담 답변의 '근거 인용 적절성'을 평가하세요. 아래 인용된 근거(가상 약관·기준표)만 기준으로 봅니다.
5점: 보장·과실 안내가 인용된 근거와 모두 맞고, 근거 없는 보장을 말하지 않는다
4점: 대체로 맞지만 표현이 약간 부정확하다
3점: 근거와 맞지 않는 안내가 한 군데 있지만 핵심(보장 담보와 과실 기준)은 맞다
2점: 근거와 맞지 않거나 근거 없는 보장 안내가 여러 군데 있다
1점: 근거와 무관하거나 틀린 안내가 대부분이다
숫자 블록([추정 결과])의 금액·비율은 코드가 계산한 값이므로 맞다고 보고, 안내 문장과 근거 조항의 일치를 보세요.

이 서비스의 정책(SERVICE.md)을 기준으로 판단하세요. 아래 정책대로 한 안내는 감점하지 않습니다.
- 안내는 고객이 받는 보상만 다룬다. 대물 = 상대방 보험이 고객 차량 수리비(상대방 과실분)를 보상, 자차 = 고객 보험이 고객 과실분 수리비를 보상(자기부담금 공제), 대인 = 상대방 보험이 고객의 치료 관련 손해(상대방 과실분)를 보상
- 보험사 측 반론(A3)이 근거로 확인되면 그 가능성을 반영해 과실 범위를 넓힌다(좁히지 않는다). 이때 '기본 범위에 보험사 반론의 가능성을 반영해 넓힘'이라고 표시한다
- '보험사가 검토할 수 있는 사항'은 보험사가 제기할 수 있는 쟁점이고, '가입 특약'은 고객이 실제로 가입한 특약이다
- 실제 법령은 '[일반 정보]'로 일반 원칙만 설명한다"""
# T8-2 1차 평가(2026-11-05)에서 심사가 고객 가입 정보와 서비스 정책을 몰라 정책대로 한 안내를 감점했다.
# 그래서 심사에 고객 가입 정보와 위 정책을 함께 준다. 근거와 맞지 않는 안내를 찾는 기준 자체는 바꾸지 않았다.


def customer_profile(customer_id: str | None) -> str:
    from src import tools

    c = tools.get_customer(customer_id)
    joined = ", ".join(k for k, v in c.coverages.items() if v.joined) or "없음"
    not_joined = ", ".join(k for k, v in c.coverages.items() if not v.joined) or "없음"
    riders = ", ".join(c.riders) or "없음"
    return f"{c.id} ({c.plan}) · 가입 담보: {joined} · 미가입: {not_joined} · 가입 특약: {riders}"


def judge(question: str, final: dict, customer_id: str | None = None) -> JudgeResult:
    """고정 루브릭·고정 모델(llm.PRIMARY_MODEL 우선)로 심사한다."""
    from src import llm

    contexts = "\n".join(f"[{c['id']}] ({c['source']}) {c['text']}" for c in final["contexts"])
    prompt = (f"{RUBRIC}\n\n고객 가입 정보: {customer_profile(customer_id)}\n고객 질문: {question}\n\n"
              f"인용된 근거:\n{contexts or '(없음)'}\n\n답변:\n{final['answer']}")
    return llm.structured(JudgeResult, prompt).value


# ---------------------------------------------------------------- 실행
@dataclass
class ItemResult:
    query: Query
    responses: list[dict]
    checks: dict[str, bool]
    violations: list[str]
    judge: JudgeResult | None = None
    judge_error: str | None = None
    continued: bool = False
    seconds: float = 0.0

    @property
    def final(self) -> dict:
        return self.responses[-1]

    @property
    def passed(self) -> bool:
        ok = all(self.checks.values()) and not self.violations
        if self.query.type == "debate":
            ok = ok and self.judge is not None and self.judge.score >= JUDGE_PASS
        return ok


def run_item(q: Query, run_query: Callable, judge_fn: Callable = judge) -> ItemResult:
    t0 = time.time()
    responses = [run_query(q.question, customer_id=q.customer_id)]
    continued = False
    if q.expected_route == "debate" and responses[-1]["trace"].get("route") == "followup":
        responses.append(run_query(CONTINUE_TEXT, customer_id=q.customer_id,
                                   thread_id=responses[-1]["trace"]["thread_id"]))
        continued = True
    final = responses[-1]
    violations = [v for r in responses for v in must_zero_violations(r)]
    item = ItemResult(query=q, responses=responses, checks=code_checks(q, final), violations=violations,
                      continued=continued)
    if q.type == "debate":
        try:
            item.judge = judge_fn(q.question, final, q.customer_id)
        except Exception as exc:  # 심사 실패도 결과에 드러낸다
            item.judge_error = type(exc).__name__
    item.seconds = time.time() - t0
    return item


def verdict(items: list[ItemResult]) -> dict:
    total = sum(i.passed for i in items)
    by_type = {t: (sum(i.passed for i in items if i.query.type == t), sum(i.query.type == t for i in items))
               for t in TYPE_COUNTS}
    zero = sum(len(i.violations) for i in items)
    conditions = {
        f"전체 {TOTAL_MIN_PASS}/{len(items)} 이상": total >= TOTAL_MIN_PASS,
        **{f"{TYPE_LABELS[t]} {n}/{TYPE_COUNTS[t]} 이상": by_type[t][0] >= n for t, n in TYPE_MIN_PASS.items()},
        "필수 0건 위반 0": zero == 0,
    }
    return {"total": total, "by_type": by_type, "zero": zero, "conditions": conditions,
            "passed": all(conditions.values())}


def write_report(items: list[ItemResult], v: dict, path: Path = REPORT, elapsed: float = 0.0) -> None:
    from src import llm

    L = ["# 평가 리포트", "",
         f"- 실행: {time.strftime('%Y-%m-%d %H:%M')} · 문항 {len(items)}개 · 소요 {elapsed / 60:.1f}분",
         f"- 모델: 기본 `{llm.PRIMARY_MODEL}` (대체 {', '.join('`' + m + '`' for m in llm.FALLBACK_MODELS)}) · 심사 루브릭 고정, 3점 이상 통과",
         "- 심사에는 고객 가입 정보와 서비스 정책(담보 방향, 확인된 반론은 범위를 넓힘)을 함께 준다. "
         "1차 평가(2026-11-05 10:55)에서 심사가 이 정보를 몰라 정책대로 한 안내를 감점해 추가했다",
         f"- **최종: {'합격' if v['passed'] else '불합격'}** (전체 {v['total']}/{len(items)} 통과, 필수 0건 위반 {v['zero']}건)",
         "", "## 합격 조건 (SERVICE.md 5장)", "", "| 조건 | 결과 |", "|---|---|"]
    L += [f"| {k} | {'✅' if ok else '❌'} |" for k, ok in v["conditions"].items()]
    L += ["", "## 유형별 결과", "", "| 유형 | 통과 |", "|---|---|"]
    L += [f"| {TYPE_LABELS[t]} | {p}/{n} |" for t, (p, n) in v["by_type"].items()]
    L += ["", "## 문항별 결과", "", "| ID | 유형 | 결과 | route | rounds | 코드 채점 | 필수 0건 | LLM 심사 | 비고 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i in items:
        t = i.final["trace"]
        failed = [k for k, ok in i.checks.items() if not ok]
        code = "✅" if not failed else "❌ " + ", ".join(failed)
        zero = "✅" if not i.violations else "❌ " + ", ".join(i.violations)
        jd = (f"{i.judge.score}점" if i.judge else (f"실패({i.judge_error})" if i.judge_error else "-"))
        note = ("이어 보냄 · " if i.continued else "") + f"{i.seconds:.0f}초"
        L.append(f"| {i.query.id} | {TYPE_LABELS[i.query.type]} | {'PASS' if i.passed else 'FAIL'} | {t.get('route')} | "
                 f"{t.get('rounds', 0)} | {code} | {zero} | {jd} | {note} |")
    L += ["", "## LLM 심사 근거 (토론 문항)", ""]
    for i in items:
        if i.judge:
            probs = f" 지적: {'; '.join(i.judge.problems)}" if i.judge.problems else ""
            L.append(f"- **{i.query.id}** {i.judge.score}점: {i.judge.reasoning}{probs}")
    L += ["", "## 문항별 최종 답변", ""]
    for i in items:
        L += [f"### {i.query.id} ({TYPE_LABELS[i.query.type]}, 고객 {i.query.customer_id})", "",
              f"질문: {i.query.question}", ""]
        if i.continued:
            L += [f"(첫 응답이 되묻기여서 \"{CONTINUE_TEXT}\"를 이어 보냄)", ""]
        L += ["```", i.final["answer"], "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def run_all(queries: list[Query], run_query: Callable, judge_fn: Callable = judge,
            progress: Callable[[str], None] = print) -> list[ItemResult]:
    items = []
    for q in queries:
        item = run_item(q, run_query, judge_fn)
        progress(f"[{'PASS' if item.passed else 'FAIL'}] {q.id} ({q.type}) {item.seconds:.0f}초"
                 + (f" 실패: {[k for k, ok in item.checks.items() if not ok] + item.violations}" if not item.passed else ""))
        items.append(item)
    return items


def main(argv: list[str]) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    queries = load_queries()
    counts = validate(queries)
    print(f"평가셋 {QUERIES.relative_to(ROOT)}: {len(queries)}문항, 컬럼 규약 일치")
    print("유형별 문항 수: " + " · ".join(f"{t} {n}" for t, n in counts.items()))
    if "--check" in argv:
        return 0

    # 평가 중 생기는 청구는 임시 저장소에 저장한다 (MCP 서버는 자식 프로세스라 환경 변수로 넘긴다)
    tmp = Path(tempfile.mkdtemp(prefix="eval-claims-"))
    store = tmp / "claims.json"
    shutil.copy(ROOT / "data" / "claims.json", store)
    os.environ["MINI_PJT_CLAIMS_PATH"] = str(store)
    from src import agent, tools
    tools.CLAIMS_PATH = store
    try:
        t0 = time.time()
        items = run_all(queries, agent.run_query)
        v = verdict(items)
        write_report(items, v, elapsed=time.time() - t0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n최종: {'합격' if v['passed'] else '불합격'} · 전체 {v['total']}/{len(items)} · 필수 0건 위반 {v['zero']}건")
    for k, ok in v["conditions"].items():
        print(f"  [{'OK' if ok else 'NG'}] {k}")
    print(f"리포트: {REPORT.relative_to(ROOT)}")
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
