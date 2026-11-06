"""A2 근거 검색 노드 테스트 (LLM은 가짜 응답, 검색은 BM25만)."""
import unittest
from unittest.mock import patch

from src import agent, debate, guards, llm, retriever
from src.schemas import A1Opinion, A2Evidence, CoverageCitation, Intake, QueryResponse
from src.tests import offline_search
from src.tests.fakes import default_value

FULL = Intake(intent="accident", accident_type="후미추돌", occurred_at="어제", location="강남역",
              injury="minor", damage_level="moderate")


class FakeA2:
    def __init__(self, evidence: A2Evidence, intake: Intake = FULL):
        self.evidence = evidence
        self.intake = intake
        self.a2_prompt = ""

    def structured(self, schema, messages):
        if schema is Intake:
            return llm.LLMResult(value=self.intake, model_id="fake")
        if schema is A2Evidence:
            self.a2_prompt = str(messages[-1].content)
            return llm.LLMResult(value=self.evidence, model_id="fake")
        return llm.LLMResult(value=default_value(schema), model_id="fake")

    def text(self, messages):
        return llm.LLMResult(value="안내 문장입니다.", model_id="fake")


def run(fake: FakeA2, question="어제 강남역에서 뒤차가 받았고 목이 아파요", search=None):
    agent.reset_memory()
    with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text), \
            patch.object(agent, "retrieval_search", search or offline_search):
        return agent.run_query(question)


def cite(**by_cov) -> A2Evidence:
    return A2Evidence(coverage_citations=[CoverageCitation(coverage=k, clause_ids=v) for k, v in by_cov.items()])


class A2EvidenceTest(unittest.TestCase):
    def test_contexts_only_contain_retrieved_ids(self):
        fake = FakeA2(cite(대물=["ART-04", "ART-05"], 자차=["ART-06"], 대인=["ART-02", "ART-03"]))
        resp = run(fake)
        QueryResponse.model_validate(resp)
        # A2가 받은 프롬프트에 있던(=검색된) 조항만 contexts에 들어간다
        retrieved = {line.split("]")[0][1:] for line in fake.a2_prompt.splitlines() if line.startswith("[")}
        clause_ids = [c["id"] for c in resp["contexts"] if c["source"] == "약관"]
        self.assertTrue(clause_ids)
        self.assertTrue(set(clause_ids) <= retrieved)
        self.assertIn("약관 근거(가상 약관): 대물 ART-04, ART-05", resp["answer"])

    def test_unretrieved_id_is_dropped(self):
        fake = FakeA2(cite(대물=["ART-04", "ART-99"], 자차=["SP-77"]))
        resp = run(fake)
        self.assertEqual(resp["trace"]["dropped_ids"], ["ART-99", "SP-77"])
        ids = [c["id"] for c in resp["contexts"]]
        self.assertNotIn("ART-99", ids)
        self.assertNotIn("ART-99", resp["answer"])
        self.assertIn(f"자차 {guards.NOT_CONFIRMED}", resp["answer"])

    def test_no_results_is_not_confirmed(self):
        resp = run(FakeA2(cite(대물=["ART-04"])), search=lambda q: [])
        self.assertIn(f"대물 {guards.NOT_CONFIRMED} / 자차 {guards.NOT_CONFIRMED} / 대인 {guards.NOT_CONFIRMED}", resp["answer"])
        self.assertEqual([c["id"] for c in resp["contexts"]], ["FT-01"])
        self.assertIsNone(resp["trace"]["error"])

    def test_retrieval_failure(self):
        def boom(q):
            raise retriever.RetrievalFailed()
        resp = run(FakeA2(cite(대물=["ART-04"])), search=boom)
        self.assertEqual(resp["trace"]["error"], "retrieval_failed")
        self.assertIn(f"대물 {guards.NOT_CONFIRMED}", resp["answer"])
        self.assertTrue(resp["answer"].endswith(guards.DISCLAIMER))  # 답변은 계속 나온다
        QueryResponse.model_validate(resp)

    def test_unconfirmed_topic_from_question(self):
        ev = cite(대물=["ART-04"])
        ev.unconfirmed_topics = ["무보험차 상해"]
        resp = run(FakeA2(ev), question="어제 강남역에서 뒤차가 받았는데 무보험차 상해도 되나요")
        self.assertIn(f"약관에서 {guards.NOT_CONFIRMED}: 무보험차 상해", resp["answer"])

    def test_general_info_is_labeled_and_article_numbers_removed(self):
        ev = cite(대물=["ART-04"])
        ev.general_info = ["후미추돌은 일반적으로 뒤차의 안전거리 미확보로 본다.", "도로교통법 제19조에 따라 안전거리를 지켜야 한다.",
                           "일반적으로 뒤차에 100% 과실이 인정된다."]
        resp = run(FakeA2(ev))
        self.assertIn("[일반 정보] 후미추돌은 일반적으로", resp["answer"])
        self.assertNotIn("제19조", resp["answer"])
        self.assertNotIn("뒤차에 100% 과실", resp["answer"])  # 숫자로 단정하는 일반 정보는 버린다
        self.assertTrue(resp["answer"].endswith(guards.DISCLAIMER))

    def test_validate_evidence_unit(self):
        ev, dropped = debate.validate_evidence(cite(대물=["ART-04", "X-1", "ART-04"]), {"ART-04"})
        self.assertEqual(ev.coverage_citations[0].clause_ids, ["ART-04"])
        self.assertEqual(dropped, ["X-1"])

    def test_a2_schema_failure_is_soft(self):
        def structured(schema, messages):
            if schema is A2Evidence:
                raise llm.LLMUnavailable([("m1", "ValidationError"), ("m2", "ValidationError")])
            return FakeA2(cite()).structured(schema, messages)
        agent.reset_memory()
        with patch.object(agent, "call_structured", structured), patch.object(agent, "call_text", FakeA2(cite()).text):
            resp = agent.run_query("어제 강남역에서 뒤차가 받았고 목이 아파요")
        self.assertEqual(resp["trace"]["error"], "agent_failed:A2")
        self.assertIn(f"대물 {guards.NOT_CONFIRMED}", resp["answer"])


if __name__ == "__main__":
    unittest.main()
