"""LLM 설정과 대체 모델 호출.

모델 목록은 이 파일 한 곳에서 관리한다. 기본 모델 호출이 실패하면(권한·한도·모델 오류)
다음 모델로 넘어가고, 모두 실패하면 LLMUnavailable 예외를 낸다.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

# 자격 증명은 레포 루트(mini-pjt의 상위 폴더)의 .env에서 읽는다.
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# 기본은 us-east-1의 gpt-oss-120b다. Claude global 프로필은 호출 한도(ThrottlingException)에
# 자주 걸려 대체 모델로 둔다. 순서대로 시도한다.
PRIMARY_MODEL = "openai.gpt-oss-120b-1:0"
FALLBACK_MODELS: tuple[str, ...] = (
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-6",
)
FALLBACK_MODEL = FALLBACK_MODELS[0]
DEFAULT_MODELS: tuple[str, ...] = (PRIMARY_MODEL, *FALLBACK_MODELS)
# 역할별 모델: 고객 안내문(요청당 1번)은 지시를 더 잘 지키는 Claude를 먼저 쓴다.
# gpt-oss는 안내문에 숫자를 쓰거나 담보 방향을 틀리는 일이 있었다. Claude가 한도에 걸리면 gpt-oss로 넘어간다.
ANSWER_MODELS: tuple[str, ...] = (*FALLBACK_MODELS, PRIMARY_MODEL)

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """모든 모델 호출이 실패했을 때 낸다. trace.error 값은 code를 쓴다."""

    code = "llm_unavailable"

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts  # [(모델 ID, 오류 종류)]
        summary = ", ".join(f"{m}: {e}" for m, e in attempts)
        super().__init__(f"모든 모델 호출 실패 ({summary})")


@dataclass
class LLMResult:
    value: Any  # 호출 결과 (메시지 또는 구조화 출력 객체)
    model_id: str  # 실제로 응답한 모델 ID


def make_chat(model_id: str, temperature: float = 0) -> ChatBedrockConverse:
    return ChatBedrockConverse(model=model_id, region_name=REGION, temperature=temperature)


def needs_tool_hint(model_id: str) -> bool:
    """Bedrock에서 gpt-oss는 tool_choice='auto'만 지원해 도구(구조화 출력)를 건너뛰고 문장으로 답하곤 한다."""
    return "gpt-oss" in model_id


def structured_invoke(chat: ChatBedrockConverse, schema: type, messages: Any) -> Any:
    """구조화 출력 호출. 결과가 없으면(도구를 안 부름) 형식 오류로 보고 다음 모델·재시도로 넘긴다."""
    msgs = [HumanMessage(content=messages)] if isinstance(messages, str) else list(messages)
    if needs_tool_hint(chat.model_id):
        msgs.append(SystemMessage(content=f"답은 반드시 {schema.__name__} 도구를 한 번 호출해서 낸다. "
                                          "일반 문장으로 답하지 않는다."))
    result = chat.with_structured_output(schema).invoke(msgs)
    if result is None:
        raise OutputParserException(f"{schema.__name__} 구조화 출력이 없습니다")
    return result


def structured(schema: type, messages: Any, models: Sequence[str] | None = None, temperature: float = 0) -> LLMResult:
    """구조화 출력 호출 + 대체 모델. value에는 schema 객체가 들어간다."""
    return invoke_with_fallback(lambda c: structured_invoke(c, schema, messages), models, temperature)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모은다."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def invoke_with_fallback(
    run: Callable[[ChatBedrockConverse], Any],
    models: Sequence[str] | None = None,
    temperature: float = 0,
) -> LLMResult:
    """models 순서대로 run(chat)을 시도하고, 처음 성공한 결과와 모델 ID를 돌려준다.

    run은 채팅 모델을 받아 호출 결과를 돌려주는 함수다.
    예) lambda chat: chat.invoke(messages)
        lambda chat: chat.with_structured_output(Schema).invoke(messages)
    """
    attempts: list[tuple[str, str]] = []
    for model_id in models or DEFAULT_MODELS:
        try:
            return LLMResult(value=run(make_chat(model_id, temperature)), model_id=model_id)
        except Exception as exc:  # 권한·한도·모델 오류 모두 다음 모델로 넘어간다
            # 로그에는 모델 ID와 오류 종류만 남긴다(입력 원문은 남기지 않음).
            logger.warning("LLM 호출 실패: model=%s error=%s", model_id, type(exc).__name__)
            attempts.append((model_id, type(exc).__name__))
    raise LLMUnavailable(attempts)


def chat(messages: Any, models: Sequence[str] | None = None, temperature: float = 0) -> LLMResult:
    """일반 텍스트 호출. value에는 응답 텍스트가 들어간다."""
    result = invoke_with_fallback(lambda c: c.invoke(messages), models, temperature)
    return LLMResult(value=get_text(result.value), model_id=result.model_id)


if __name__ == "__main__":
    # 확인 스크립트: python -m src.llm [기본 모델 ID] [대체 모델 ID]
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    models = sys.argv[1:] or list(DEFAULT_MODELS)
    try:
        res = chat("한 문장으로 '연결 확인 완료'라고만 답해 주세요.", models=models)
        print(f"응답: {res.value.strip()}")
        print(f"사용 모델: {res.model_id}")
    except LLMUnavailable as exc:
        print(f"{exc.code}: {exc}")
        sys.exit(1)
