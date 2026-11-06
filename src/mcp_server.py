"""청구·서류 MCP 서버 (fastmcp, stdio). 교육 코드 day7_practice/mcp_server.py 구조를 따른다.

노출하는 도구 3개 (실제 처리는 모두 src/tools.py 함수가 한다):
  - save_claim          청구 저장과 청구번호 발급 (같은 대화면 기존 번호)
  - get_claim_status    본인 청구의 진행 상태 조회
  - document_checklist  사고 유형·부상 여부별 필요 서류

실행: python src/mcp_server.py  (단독으로 쓰지 않고, A4의 MCP 클라이언트가 자식 프로세스로 띄운다)
환경 변수 MINI_PJT_CLAIMS_PATH가 있으면 그 파일을 청구 저장소로 쓴다 (테스트용).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")  # 서버가 뜰 때마다 찍히는 INFO 로그를 끈다
from fastmcp import FastMCP  # noqa: E402

from src import guards, tools  # noqa: E402
from src.schemas import CoverageAmount, PercentRange  # noqa: E402

if os.getenv("MINI_PJT_CLAIMS_PATH"):
    tools.CLAIMS_PATH = Path(os.environ["MINI_PJT_CLAIMS_PATH"])

mcp = FastMCP("car-claim-desk")


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
def save_claim(customer_id: str, thread_id: str, accident_summary: str, missing_fields: list[str] = [],
               customer_fault: Optional[dict] = None, coverages: list[dict] = [], documents: list[str] = []) -> str:
    """사고 평가가 끝난 청구를 저장하고 청구번호를 발급한다. 같은 고객·같은 대화(thread_id)면 기존 번호를 돌려준다.

    Args:
        customer_id: 가상 고객 ID (예: C001)
        thread_id: 대화 ID
        accident_summary: 사고 요약 (마스킹된 문장)
        missing_fields: 비어 있는 필수 항목 이름
        customer_fault: 고객 과실 범위 {"min": int, "max": int}
        coverages: 담보별 금액 [{"coverage": "대물", "covered": true, "amount": {"min": int, "max": int} | null}]
        documents: 필요 서류 이름
    """
    try:
        record, created = tools.save_claim(
            customer_id, thread_id, accident_summary, missing_fields=missing_fields,
            customer_fault=PercentRange(**customer_fault) if customer_fault else None,
            coverages=[CoverageAmount.model_validate(c) for c in coverages], documents=documents,
        )
    except tools.StoreError:
        return _json({"error": tools.StoreError.code})
    return _json({"created": created, "claim": record.model_dump(mode="json")})


@mcp.tool()
def get_claim_status(claim_no: str, customer_id: str) -> str:
    """청구번호로 본인 청구의 진행 상태를 조회한다. 다른 고객의 번호나 없는 번호는 찾을 수 없다고 답한다.

    Args:
        claim_no: 청구번호 (CLM-YYYYMMDD-NNN)
        customer_id: 지금 대화 중인 가상 고객 ID
    """
    try:
        record = tools.get_claim(claim_no, customer_id)
    except tools.StoreError:
        return _json({"error": tools.StoreError.code})
    if record is None:
        return _json({"found": False, "message": guards.CLAIM_NOT_FOUND})
    return _json({"found": True, "claim": record.model_dump(mode="json")})


@mcp.tool()
def document_checklist(fault_table_id: Optional[str] = None, injury: Optional[str] = None) -> str:
    """사고 유형(기준표 ID)과 부상 여부로 필요 서류를 돌려준다.

    Args:
        fault_table_id: 과실비율 기준표 ID (예: FT-01). 모르면 생략
        injury: 부상 정도 none / minor / severe. 모르면 생략
    """
    required, conditional = tools.required_documents(fault_table_id, injury)
    return _json({"required": required, "conditional": conditional})


if __name__ == "__main__":
    mcp.run(show_banner=False)  # stdio 모드로 실행 (도구를 부를 때마다 찍히는 시작 배너는 끈다)
