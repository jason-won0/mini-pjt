"""POST /query API (starlette). run_query()를 감싸는 얇은 인터페이스다.

실행: uvicorn src.server:app --port 8000   (mini-pjt/에서)
상태 코드: 정상 200, bad_request 400, llm_unavailable 503. 본문은 항상 answer·contexts·trace 세 키다.
"""
from __future__ import annotations

import logging

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src import agent
from src.schemas import QueryRequest

logger = logging.getLogger(__name__)

STATUS_BY_ERROR = {"bad_request": 400, "llm_unavailable": 503}


def _json(body: dict) -> JSONResponse:
    status = STATUS_BY_ERROR.get(body["trace"].get("error"), 200)
    return JSONResponse(body, status_code=status)


async def query(request: Request) -> JSONResponse:
    payload = None
    try:
        payload = await request.json()
        req = QueryRequest.model_validate(payload)
    except (ValueError, ValidationError):
        # JSON이 아니거나 question이 없거나 비어 있음. 요청 본문은 로그에 남기지 않는다.
        thread_id = payload.get("thread_id") if isinstance(payload, dict) else None
        return _json(agent.error_response("bad_request", thread_id))

    body = await run_in_threadpool(agent.run_query, req.question, req.customer_id, req.thread_id)
    return _json(body)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[Route("/query", query, methods=["POST"]), Route("/health", health, methods=["GET"])])
