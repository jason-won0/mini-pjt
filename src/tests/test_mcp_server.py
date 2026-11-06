"""MCP 서버 통합 테스트: MCP 클라이언트(langchain-mcp-adapters)로 stdio 서버를 띄워 세 도구를 호출한다.

청구 저장소는 MINI_PJT_CLAIMS_PATH로 임시 파일을 쓴다 (실제 data/claims.json은 건드리지 않는다).
네트워크는 쓰지 않는다 (MCP 서버는 같은 PC의 자식 프로세스).
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER = str(Path(__file__).resolve().parents[1] / "mcp_server.py")
REAL_STORE = Path(__file__).resolve().parents[2] / "data" / "claims.json"


def text_of(result) -> str:
    """도구 결과에서 텍스트를 꺼낸다 (어댑터 버전에 따라 문자열 또는 콘텐츠 블록 목록)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in result)
    return str(result)


class MCPServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.store = cls.dir / "claims.json"
        shutil.copy(REAL_STORE, cls.store)  # 샘플 청구 4건으로 시작
        cls.real_before = REAL_STORE.read_bytes()
        client = MultiServerMCPClient({"claims": {
            "command": sys.executable, "args": [SERVER], "transport": "stdio",
            "env": {**os.environ, "MINI_PJT_CLAIMS_PATH": str(cls.store)},
        }})
        cls.tools = {t.name: t for t in asyncio.run(client.get_tools())}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def call(self, name, **args):
        return json.loads(text_of(asyncio.run(self.tools[name].ainvoke(args))))

    def test_tools_are_exposed(self):
        self.assertEqual(set(self.tools), {"save_claim", "get_claim_status", "document_checklist"})

    def test_save_then_lookup(self):
        out = self.call("save_claim", customer_id="C001", thread_id="t-mcp-1", accident_summary="뒤차 추돌",
                        customer_fault={"min": 0, "max": 10},
                        coverages=[{"coverage": "대물", "covered": True, "amount": {"min": 1350000, "max": 5000000}}],
                        documents=["사고 경위서"])
        self.assertTrue(out["created"])
        claim_no = out["claim"]["claim_no"]
        self.assertRegex(claim_no, r"^CLM-\d{8}-\d{3}$")
        self.assertEqual(out["claim"]["status"], "접수")
        again = self.call("save_claim", customer_id="C001", thread_id="t-mcp-1", accident_summary="뒤차 추돌")
        self.assertFalse(again["created"])
        self.assertEqual(again["claim"]["claim_no"], claim_no)
        found = self.call("get_claim_status", claim_no=claim_no, customer_id="C001")
        self.assertTrue(found["found"])
        self.assertEqual(found["claim"]["status"], "접수")

    def test_lookup_sample_and_other_customer(self):
        found = self.call("get_claim_status", claim_no="CLM-20261101-002", customer_id="C001")
        self.assertEqual(found["claim"]["status"], "심사중")
        other = self.call("get_claim_status", claim_no="CLM-20261101-002", customer_id="C003")
        self.assertEqual(other, {"found": False, "message": "해당 청구를 찾을 수 없습니다."})

    def test_document_checklist(self):
        out = self.call("document_checklist", fault_table_id="FT-01", injury="minor")
        self.assertIn("진단서", out["required"])
        unknown = self.call("document_checklist", fault_table_id="FT-01")
        self.assertIn("진단서", unknown["conditional"])

    def test_real_store_untouched(self):
        self.assertEqual(REAL_STORE.read_bytes(), self.real_before)


if __name__ == "__main__":
    unittest.main()
