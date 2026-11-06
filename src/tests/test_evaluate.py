"""평가 실행기의 채점 로직 테스트 (가짜 응답, 네트워크 없음)."""
import tempfile
import unittest
from pathlib import Path

from src import evaluate, guards
from src.evaluate import JudgeResult, Query


def resp(answer, route="debate", rounds=1, contexts=("FT-01", "ART-04"), claims=(), thread="t-1"):
    return {"answer": answer, "contexts": [{"id": i, "source": "약관", "text": "…"} for i in contexts],
            "trace": {"route": route, "rounds": rounds, "claims": list(claims), "thread_id": thread}}


GOOD = f"안내\n\n[추정 결과]\n- 적용 기준: FT-01\n- 약관 근거(가상 약관): 대물 ART-04\n\n{guards.DISCLAIMER}"
DEBATE_Q = Query(id="D", type="debate", question="뒤에서 받혔어요", customer_id="C001", expected_route="debate",
                 expected_context_ids=["FT-01", "ART-04"], must_include=["보험사 심사 후 확정됩니다"])


class MustZeroTest(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(evaluate.must_zero_violations(resp(GOOD)), [])

    def test_pii(self):
        # 전화번호 하나를 전화번호·계좌번호로 두 번 세지 않는다
        self.assertIn("개인정보 노출(phone)", evaluate.must_zero_violations(resp("번호 010-1234-5678")))

    def test_unknown_and_unretrieved_ids(self):
        v = evaluate.must_zero_violations(resp(GOOD.replace("ART-04", "ART-04, ART-99, ART-13")))
        self.assertIn("없는 근거 ID ART-99", v)
        self.assertIn("contexts에 없는 근거 ID ART-13", v)

    def test_missing_disclaimer(self):
        self.assertIn("면책 문구 누락", evaluate.must_zero_violations(resp(GOOD.replace(guards.DISCLAIMER, ""))))

    def test_rounds_over_three(self):
        self.assertIn("토론 4라운드(3 초과)", evaluate.must_zero_violations(resp(GOOD, rounds=4)))

    def test_claim_number_is_not_a_basis_id(self):
        answer = GOOD.replace(guards.DISCLAIMER, "청구번호: CLM-20261105-001\n\n" + guards.DISCLAIMER)
        self.assertEqual(evaluate.must_zero_violations(resp(answer)), [])


class RunItemTest(unittest.TestCase):
    def test_continues_once_when_debate_followup(self):
        calls = []

        def run_query(q, customer_id=None, thread_id=None):
            calls.append((q, thread_id))
            return resp("되묻기", route="followup", thread="t-9") if len(calls) == 1 else resp(GOOD, thread="t-9")

        item = evaluate.run_item(DEBATE_Q, run_query, judge_fn=lambda q, f, c=None: JudgeResult(score=5))
        self.assertTrue(item.continued)
        self.assertEqual(calls[1], (evaluate.CONTINUE_TEXT, "t-9"))
        self.assertTrue(item.passed)

    def test_low_judge_score_fails_debate(self):
        item = evaluate.run_item(DEBATE_Q, lambda q, **k: resp(GOOD), judge_fn=lambda q, f, c=None: JudgeResult(score=2))
        self.assertFalse(item.passed)

    def test_judge_error_fails_debate(self):
        def boom(q, f, c=None):
            raise RuntimeError("judge down")
        item = evaluate.run_item(DEBATE_Q, lambda q, **k: resp(GOOD), judge_fn=boom)
        self.assertEqual(item.judge_error, "RuntimeError")
        self.assertFalse(item.passed)

    def test_kept_claim_must_be_confirmed(self):
        claims = [{"id": "A3-1-1", "verdict": "기각", "kept": True}]
        item = evaluate.run_item(DEBATE_Q, lambda q, **k: resp(GOOD, claims=claims), judge_fn=lambda q, f, c=None: JudgeResult(score=5))
        self.assertFalse(item.checks["A3 주장 필터(kept=확인만)"])

    def test_non_debate_has_no_judge(self):
        q = Query(id="R", type="refuse", question="영화 추천", customer_id="C001", expected_route="refused",
                  must_include=["범위가 아닙니다"])
        item = evaluate.run_item(q, lambda q, **k: resp(guards.REFUSAL_MESSAGES["off_topic"], route="refused", contexts=()),
                                 judge_fn=lambda q, f, c=None: self.fail("심사하면 안 됨"))
        self.assertTrue(item.passed)
        self.assertIsNone(item.judge)


class VerdictAndReportTest(unittest.TestCase):
    def make_items(self, fail_ids=()):
        items = []
        for q in evaluate.load_queries():
            ok = q.id not in fail_ids
            items.append(evaluate.ItemResult(query=q, responses=[resp("x", route=q.expected_route)],
                                             checks={"c": ok}, violations=[],
                                             judge=JudgeResult(score=5) if q.type == "debate" else None))
        return items

    def test_all_pass(self):
        v = evaluate.verdict(self.make_items())
        self.assertTrue(v["passed"])
        self.assertEqual(v["total"], 10)

    def test_two_debate_failures_fail_type_condition(self):
        v = evaluate.verdict(self.make_items(fail_ids=("D1", "D2")))
        self.assertEqual(v["total"], 8)  # 전체는 8/10이지만
        self.assertFalse(v["conditions"]["사고 평가(토론) 3/4 이상"])  # 토론 유형 기준에서 불합격
        self.assertFalse(v["passed"])

    def test_direct_must_be_all(self):
        self.assertFalse(evaluate.verdict(self.make_items(fail_ids=("X1",)))["passed"])

    def test_report_written(self):
        items = self.make_items()
        path = Path(tempfile.mkdtemp()) / "report.md"
        evaluate.write_report(items, evaluate.verdict(items), path=path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("**최종: 합격**", text)
        self.assertIn("| D1 | 사고 평가(토론) | PASS |", text)


if __name__ == "__main__":
    unittest.main()
