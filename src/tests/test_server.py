"""POST /query 서버 테스트 (LLM은 가짜 응답으로 교체)."""
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from src import agent, guards, llm, server
from src.schemas import Intake, QueryResponse
from src.tests.test_agent_minimal import REAR_END, FakeLLM

client = TestClient(server.app)


def number_block(answer: str) -> str:
    """답변에서 코드가 만든 [추정 결과] 블록만 꺼낸다. 요청마다 달라지는 청구번호가 든 [필요 서류] 앞까지다."""
    start = answer.index("[추정 결과]")
    ends = [i for i in (answer.find("[필요 서류]"), answer.find("[일반 정보]"), answer.find(guards.DISCLAIMER)) if i > start]
    return answer[start:min(ends)].strip()


class ServerTest(unittest.TestCase):
    def post(self, json=None, content=None):
        fake = FakeLLM(REAR_END)
        with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
            return client.post("/query", json=json, content=content)

    def test_ok(self):
        r = self.post({"question": "신호 대기 중에 뒤에서 차가 들이받았어요"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        QueryResponse.model_validate(body)
        self.assertEqual(set(body), {"answer", "contexts", "trace"})
        self.assertTrue(body["trace"]["thread_id"])
        self.assertTrue(body["answer"].endswith(guards.DISCLAIMER))

    def test_thread_id_is_echoed(self):
        r = self.post({"question": "뒤에서 받혔어요", "thread_id": "t-fixed", "customer_id": "C001"})
        self.assertEqual(r.json()["trace"]["thread_id"], "t-fixed")

    def test_missing_question(self):
        r = self.post({"customer_id": "C001"})
        self.assertEqual(r.status_code, 400)
        body = r.json()
        self.assertEqual(body["trace"]["error"], "bad_request")
        self.assertEqual(body["answer"], guards.BAD_REQUEST_MESSAGE)
        self.assertEqual(set(body), {"answer", "contexts", "trace"})

    def test_empty_question_and_bad_json(self):
        self.assertEqual(self.post({"question": ""}).status_code, 400)
        self.assertEqual(self.post(content=b"not json").status_code, 400)

    def test_unknown_customer(self):
        r = self.post({"question": "뒤에서 받혔어요", "customer_id": "C999"})
        self.assertEqual(r.json()["trace"]["error"], "bad_request")

    def test_llm_unavailable_is_503(self):
        def boom(*a, **k):
            raise llm.LLMUnavailable([("m", "Err")])
        with patch.object(agent, "call_structured", boom):
            r = client.post("/query", json={"question": "뒤에서 받혔어요"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["trace"]["error"], "llm_unavailable")

    def test_api_and_direct_number_block_match(self):
        q = "신호 대기 중에 뒤에서 차가 들이받았어요"
        api = self.post({"question": q}).json()
        fake = FakeLLM(REAR_END)
        with patch.object(agent, "call_structured", fake.structured), patch.object(agent, "call_text", fake.text):
            direct = agent.run_query(q)
        self.assertEqual(number_block(api["answer"]), number_block(direct["answer"]))
        self.assertEqual(api["contexts"], direct["contexts"])

    def test_health(self):
        self.assertEqual(client.get("/health").json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
