"""테스트 공용 설정.

1. 에이전트의 약관 검색을 네트워크 없는 BM25 검색으로 바꾼다. 그래서 전체 테스트는 Bedrock 없이 돈다.
   실제 검색(임베딩)을 확인하는 테스트는 retriever.search를 직접 부르고 RUN_LIVE=1일 때만 돈다.
2. 청구 저장소를 임시 파일(샘플 청구 복사본)로 바꾸고, A4의 MCP 호출을 같은 프로세스 호출로 바꾼다.
   사고 평가 테스트마다 MCP 서버를 띄우면 느리고 실제 data/claims.json에 청구가 쌓이기 때문이다.
   실제 MCP 경로(stdio)는 test_mcp_server.py와 test_a4.py의 MCP 테스트가 확인한다.
"""
import atexit
import json
import shutil
import tempfile
from pathlib import Path

from src import agent, mcp_server, retriever, tools


def offline_search(query: str):
    return retriever.search(query, vector_fn=lambda q: [])


def local_call_mcp(calls):
    """agent.call_mcp와 같은 규칙으로, mcp_server의 도구 함수를 같은 프로세스에서 부른다."""
    results = []
    for name, args in calls:
        args = args(results) if callable(args) else args
        tool = getattr(mcp_server, name)
        fn = getattr(tool, "fn", tool)  # fastmcp 도구 객체의 원래 함수
        results.append(json.loads(fn(**args)))
    return results


_TMP = Path(tempfile.mkdtemp(prefix="claims-test-"))
REAL_CLAIMS = tools.CLAIMS_PATH
TEST_CLAIMS = _TMP / "claims.json"
shutil.copy(REAL_CLAIMS, TEST_CLAIMS)
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

REAL_CALL_MCP = agent.call_mcp  # 실제 MCP 경로를 확인하는 테스트가 쓴다

tools.CLAIMS_PATH = TEST_CLAIMS
agent.retrieval_search = offline_search
agent.call_mcp = local_call_mcp
