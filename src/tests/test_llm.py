"""src/llm.py 대체 모델 로직 테스트 (네트워크 없이 가짜 모델 사용)."""
import unittest
from unittest.mock import patch

from src import llm


class FakeChat:
    def __init__(self, model_id, fail):
        self.model_id = model_id
        self.fail = fail

    def invoke(self, messages):
        if self.fail:
            raise RuntimeError("model error")
        return type("Msg", (), {"content": [{"text": f"ok from {self.model_id}"}]})()


def fake_factory(failing):
    return lambda model_id, temperature=0: FakeChat(model_id, model_id in failing)


class FakeStructuredChat:
    """with_structured_output만 흉내 낸다. answers[model_id]가 None이면 도구를 안 부른 경우다."""

    def __init__(self, model_id, answers, seen):
        self.model_id, self.answers, self.seen = model_id, answers, seen

    def with_structured_output(self, schema):
        chat = self

        class Runner:
            def invoke(self, messages):
                chat.seen[chat.model_id] = messages
                return chat.answers[chat.model_id]
        return Runner()


class StructuredTest(unittest.TestCase):
    def run_structured(self, answers):
        seen = {}
        with patch.object(llm, "make_chat", lambda m, temperature=0: FakeStructuredChat(m, answers, seen)):
            res = llm.structured(dict, "질문", models=list(answers))
        return res, seen

    def test_missing_tool_call_falls_back(self):
        res, _ = self.run_structured({"openai.gpt-oss-120b-1:0": None, "claude": {"ok": 1}})
        self.assertEqual((res.model_id, res.value), ("claude", {"ok": 1}))

    def test_missing_everywhere_is_schema_error(self):
        with self.assertRaises(llm.LLMUnavailable) as ctx:
            self.run_structured({"openai.gpt-oss-120b-1:0": None})
        self.assertEqual(ctx.exception.attempts, [("openai.gpt-oss-120b-1:0", "OutputParserException")])

    def test_tool_hint_only_for_gpt_oss(self):
        _, seen = self.run_structured({"openai.gpt-oss-120b-1:0": None, "claude": {"ok": 1}})
        self.assertIn("dict 도구를 한 번 호출", seen["openai.gpt-oss-120b-1:0"][-1].content)
        self.assertEqual(len(seen["claude"]), 1)  # Claude에는 지시를 붙이지 않는다


class FallbackTest(unittest.TestCase):
    def test_primary_success(self):
        with patch.object(llm, "make_chat", fake_factory(set())):
            res = llm.chat("hi", models=["A", "B"])
        self.assertEqual(res.model_id, "A")
        self.assertEqual(res.value, "ok from A")

    def test_falls_back_when_primary_fails(self):
        with patch.object(llm, "make_chat", fake_factory({"A"})):
            res = llm.chat("hi", models=["A", "B"])
        self.assertEqual(res.model_id, "B")

    def test_all_fail_raises_unavailable(self):
        with patch.object(llm, "make_chat", fake_factory({"A", "B"})):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat("hi", models=["A", "B"])
        self.assertEqual(ctx.exception.code, "llm_unavailable")
        self.assertEqual([m for m, _ in ctx.exception.attempts], ["A", "B"])

    def test_default_models_order(self):
        self.assertEqual(llm.DEFAULT_MODELS, (llm.PRIMARY_MODEL, *llm.FALLBACK_MODELS))
        self.assertEqual(llm.PRIMARY_MODEL, "openai.gpt-oss-120b-1:0")
        self.assertEqual(llm.FALLBACK_MODEL, llm.FALLBACK_MODELS[0])

    def test_get_text_handles_str_and_blocks(self):
        self.assertEqual(llm.get_text(type("M", (), {"content": "plain"})()), "plain")
        self.assertEqual(llm.get_text(type("M", (), {"content": [{"text": "a"}, {"text": "b"}]})()), "ab")


if __name__ == "__main__":
    unittest.main()
