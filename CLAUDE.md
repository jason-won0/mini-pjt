# 프로젝트 규칙

자동차 사고 보험금 청구 도우미 데모 (삼성SDS AX 교육 미니 프로젝트). 이 문서는 **어떻게 만드는가**를 정한다.
무엇을 왜 만드는지는 [SERVICE.md](SERVICE.md)가 기준이고, 구현 순서는 [TASKS.md](TASKS.md)를 따른다. 요구사항이 바뀌면 SERVICE.md부터 수정한다. 사용자와는 한국어로 소통한다.

## 기술 스택

- Python, LangChain, LangGraph(`StateGraph`로 직접 구성), Streamlit
- **LLM:** AWS Bedrock `ChatBedrockConverse`
  - 기본 `openai.gpt-oss-120b-1:0`(us-east-1). 실패하면(권한·한도·모델 오류) `global.anthropic.claude-sonnet-4-5-20250929-v1:0` → `global.anthropic.claude-sonnet-4-6` 순서로 넘어가 중단 없이 이어 간다. Claude `global.` 프로필이 호출 한도에 자주 걸려 2026-11-05 사용자 요청으로 기본 모델을 바꿨다(`us.` 프로필은 한도, 3.5 Sonnet v2는 사용 불가).
  - 모델 목록은 `src/llm.py` 한 곳에서 관리한다. 필요하면 역할별로 모델을 바꿔 쓸 수 있게 한다. 고객 안내문(요청당 1번)은 `ANSWER_MODELS`(Claude 4.5 → 4.6 → gpt-oss)를 쓴다. gpt-oss가 안내문에 숫자를 쓰고 담보 방향을 틀린 일이 있어서다. 안내문에 숫자·근거 ID가 든 문장은 코드가 지운다.
- **임베딩:** Bedrock `amazon.titan-embed-text-v2:0`
- **RAG:** Chroma + rank-bm25 + kiwipiepy (한국어 형태소)
- **MCP:** 서버는 fastmcp, 클라이언트는 langchain-mcp-adapters
- **HTTP API:** starlette + uvicorn (fastmcp 설치 시 함께 깔려 있음)
- **패키지:** 상위 폴더의 `../requirements.txt`를 따른다. 그 목록에 없는 starlette·uvicorn(fastmcp와 함께 설치됨)과 Streamlit 1.66.1(2026-11-05 사용자 확인 후 `../.venv`에 설치)은 여기 적힌 것만 허용한다.
- **자격 증명:** `../.env`를 `python-dotenv`로 읽는다.

### 그래프 구성

- `create_supervisor`(교육 코드)는 질문을 에이전트 하나에 넘기는 방식이라 쓰지 않는다. 노드는 `StateGraph`로 직접 연결한다.
- **최종 구조:** 마스킹 → 거절 판정 → Supervisor(분류·접수) → 토론 서브그래프 → 종합 → A4(MCP) → Supervisor(답변 작성) → 면책 문구 부착
- **토론 라운드:** 1라운드는 A1(초기 판단) → A2 검색(`A2Evidence`) → A3(반론) → A2 검증(`A2Review`)이다. 2라운드부터는 A3 → A2 검증만 돌고 A1·A2 검색 결과를 재사용한다(과실 조정은 토론 뒤 종합에서 하므로 토론 중에는 두 결과가 바뀌지 않는다). 라운드 끝마다 코드가 종료를 판정한다. A2는 에이전트 하나지만 노드는 둘이다.
- **라운드 사이의 상태:** 주장과 판정은 주장 ID별로 누적한다. A2 검증은 이번 라운드의 새 주장에만 돌리고, 최종 인정은 종합 노드에서만 확정한다.
- **단계별 구현:** TASKS.md 묶음 1은 마스킹 → Supervisor → A1 → 답변만 있는 최소 그래프로 시작한다. 묶음이 진행될 때마다 노드를 끼워 넣어 최종 구조로 키운다. 노드를 추가할 때 앞 묶음의 테스트가 계속 통과해야 한다.
- **책임 구분:** 종합 노드는 **코드**로 `Synthesis`를 만든다(확인된 주장만 남김, 과실조정은 재계산, 면책·감액은 `review_notes`). Supervisor는 고객용 문장만 LLM으로 쓰고, 숫자와 근거 ID는 코드가 만든 블록을 그대로 넣는다. 면책 문구는 마지막에 코드가 붙인다.
- 멀티턴 대화는 checkpointer로 `thread_id` 단위로 이어 간다.

### MCP 연결

- A4는 langchain-mcp-adapters로 `src/mcp_server.py`를 stdio로 띄워 도구를 호출한다(`final_scenario.py` 방식). `tools.py`의 청구 함수를 직접 import하지 않는다.
- 단위 테스트는 `tools.py` 함수를 직접 테스트하고, MCP 경로는 통합 테스트로 확인한다.

### RAG 설정

- 약관은 조항 단위로 쪼개고, 문서 메타데이터에 조항 ID와 관련 담보(`대물`/`자차`/`대인`/`공통`)를 넣는다.
- 검색 순서: ① Chroma 상위 8개, 거리를 유사도(0~1)로 바꿔 `MIN_SIMILARITY` 미만 제거 ② BM25(kiwipiepy) 상위 8개, 0점 제거 ③ RRF(k=60)로 합쳐 상위 4개(RRF 점수 하한 없음) ④ 결과가 없으면 "근거 없음"
- `MIN_SIMILARITY`는 0.2다(T3-3에서 관련 질문 최저 0.247, 무관한 질문 최고 0.163을 보고 정함). `src/retriever.py`의 상수로 둔다.
- 토크나이저는 교육 코드의 품사 필터에서 의존명사·대명사를 빼고, 흔한 어간("하", "되", "있", "받", "알리" 등)을 불용어로 둔다. 무관한 질문이 이런 말 하나로 약관에 걸리지 않게 하려는 것이다.
- 고객이 쓰는 말("자차", "대물", "내 차 수리비")이 BM25에 걸리도록 대표 담보 조항 본문에 통칭 설명을 붙인다.
- `contexts`에는 실제로 검색된 조항, 사용한 기준표 항목, 답변에 쓴 고객 가입 특약 조항만 넣는다. 답변에 나오는 조항·기준표 ID는 모두 `contexts`에 있어야 한다. LLM이 인용한 ID 중 검색 결과에 없는 것은 코드가 지우고 `trace.dropped_ids`에 남긴다.

### 에이전트 입출력 스키마

금액은 원 단위 정수, 비율은 0~100 정수(%)다. 범위는 항상 `{"min": .., "max": ..}`이고 `min <= max`다. 스키마는 pydantic으로 정의하고 `src/schemas.py`에 둔다.

| 스키마 | 만드는 곳 | 필드 |
|---|---|---|
| `Intake` | Supervisor (LLM) | `intent`(`accident`/`claim_status`/`documents`/`refuse`), `deferred_intents`(list), `accident_type`, `customer_role`(`가해`/`피해`/`쌍방`/`null`, 기준표 항목의 방향과 반대면 코드가 그 유형을 쓰지 않음), `occurred_at`, `location`, `injury`(`none`/`minor`/`severe`/`null`), `other_vehicle`, `damage_level`(`minor`/`moderate`/`severe`/`null`), `claim_no` |
| `Estimate` | A1 계산 (코드) | `fault_table_id`, `customer_fault`(범위), `coverages`: list[`CoverageEstimate`] |
| `CoverageEstimate` | 코드 | `coverage`(`대물`/`자차`/`대인`), `covered`(bool), `amount`(범위 또는 null), `condition`(str 또는 null), `basis_ids`(list) |
| `A1Opinion` | A1 (LLM) | `explanation`(str). 숫자는 `Estimate`를 그대로 인용하고 새로 만들지 않는다 |
| `A2Evidence` | A2 검색 (LLM) | `coverage_citations`: list[{`coverage`, `clause_ids`}], `general_info`: list[str], `customer_info_needed`: list[str] |
| `A2Review` | A2 검증 (LLM) + 코드 검사 | `verdicts`: list[`Verdict`] |
| `Verdict` | A2 검증 | `claim_id`, `verdict`(`확인`/`기각`/`근거없음`), `evidence_ids`(list), `reason` |
| `A3Opinion` | A3 (LLM) | `claims`: list[`Claim`], `no_new_issues`(bool), `customer_info_needed`: list[str] |
| `Claim` | A3 (LLM) | `id`, `kind`(`과실조정`/`면책`/`감액`/`기타`), `coverage`(`대물`/`자차`/`대인`/`공통`), `text`, `candidate_basis_ids`(list) |
| `FaultModifier` | 데이터 | `id`, `description`, `bound`(`min`/`max`, 지금 계산에는 쓰지 않음), `adjustment`(int, ±%p). 확인되면 +는 상한, −는 하한을 넓힌다 |
| `Synthesis` | 종합 (코드) | `estimate`(조정 후 `Estimate`), `kept_claims`, `dropped_claims`, `review_notes`: list[{`claim_id`, `text`, `evidence_ids`}], `customer_info_needed`: list[str], `unresolved_issues`: list[str], `context_ids`(list) |

- **A3 주장 ID와 중복 제거:** 새 주장 ID는 코드가 `A3-{라운드}-{순번}`으로 붙인다. LLM이 ID를 재사용할 거라고 가정하지 않는다. `(kind, coverage, 정렬된 candidate_basis_ids)`가 같은 주장은 코드가 기존 주장으로 합치고 기존 ID를 유지한다. 근거 후보 ID가 없는 주장은 받지 않는다. 답변에 나가는 주장 문장에서 `contexts`에 없는 ID는 지우고(`dropped_ids`), 숫자가 들었거나 한국어가 아닌 문장은 뺀다.
- **검증 단계 (`A2Review` 뒤 코드):**
  1. 근거 ID가 실제 데이터에 있는지 확인한다. 없으면 `근거없음`이다.
  2. 근거 종류가 주장 종류와 맞는지 확인한다. 과실조정은 `FaultModifier`, 면책·감액은 약관 조항이어야 한다. 아니면 `근거없음`이다.
  3. 약관 조항의 담보가 주장의 `coverage`와 같거나 `공통`인지 확인한다. 아니면 `근거없음`이다.
  4. 위를 통과한 경우에만 A2의 `확인`/`기각` 판정을 그대로 쓴다.
- **계산·과실 조정:** SERVICE.md 4장 "보험금 추정"의 규칙을 `tools.py`에 그대로 구현한다. 과실 조정 뒤에는 담보별 보험금을 다시 계산한다.

## 폴더 구조 (제출 규약. 바꾸지 않는다)

```
src/agent.py       메인 에이전트 그래프
src/tools.py       도메인 도구
src/retriever.py   RAG 파이프라인
data/              사용한 문서와 데이터
evaluation/        test_queries.csv 와 평가 리포트
```

위 다섯 경로는 바꾸지 않는다. 보조 파일은 `src/` 아래에만 추가한다.

| 파일 | 내용 |
|---|---|
| `src/agent.py` | Supervisor·A4 노드, 그래프 조립, 진입 함수 `run_query(question, customer_id, thread_id, on_event=None)`. API와 Streamlit 모두 이 함수 하나만 부른다. `on_event`(선택)는 노드 완료·토론 발언을 알려 화면이 토론을 실시간으로 보여주는 데 쓴다 |
| `src/debate.py` | 토론 서브그래프(A1·A2·A3, 라운드 제한)와 종합 노드 |
| `src/schemas.py` | 위 pydantic 스키마와 응답 스키마 |
| `src/tools.py` | 과실 조회, 담보별 계산(`calc_property`, `calc_own_damage`, `calc_bodily`), 보장 판정, 청구 저장소·청구번호·조회 권한, 서류 체크리스트 (순수 파이썬) |
| `src/retriever.py` | 약관 인덱싱과 하이브리드 검색 |
| `src/llm.py` | 모델 설정과 대체 모델 호출 |
| `src/guards.py` | 개인정보 마스킹, 거절 판정, 고정 문구 상수 |
| `src/mcp_server.py` | `tools.py`의 청구·서류 함수를 fastmcp 도구로 노출 |
| `src/server.py` | `POST /query` (starlette) |
| `src/app.py` | Streamlit 화면 |
| `src/evaluate.py` | 평가 실행기. `evaluation/test_queries.csv`를 읽어 `evaluation/report.md`를 쓴다 |
| `src/tests/` | `unittest` 단위·통합 테스트 |
| `data/` | 가상 약관, 기준표, 손해액·대인 기준, 고객, 서류 기준, 청구 저장소(`claims.json`), Chroma 인덱스 |

## 주고받는 형식 (제출 규약)

### 요청: `POST /query` (JSON)

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question` | str | 필수 | 고객 입력 |
| `customer_id` | str | 선택 | 없으면 기본 고객(`data/`의 첫 번째 고객)을 쓴다 |
| `thread_id` | str | 선택 | 없으면 새 대화로 보고 새 ID를 만들어 `trace.thread_id`로 돌려준다. 클라이언트는 이어서 보낼 요청에 그 값을 그대로 넣는다 |

### 응답

상태 코드는 정상 `200`, `bad_request` `400`, `llm_unavailable` `503`이다. 본문은 어느 경우든 아래 세 키 형식이다.

```json
{
  "answer": "… ※ 본 안내는 유사 사례 기준의 참고용 추정이며, …",
  "contexts": [{"id": "ART-12", "source": "약관", "text": "…"}],
  "trace": {"thread_id": "t-123", "route": "debate", "rounds": 2, "...": "아래 표"}
}
```

- 응답은 이 세 키(`answer`, `contexts`, `trace`)만 최상위에 둔다.
- `answer`: 고객에게 보여줄 최종 답변(str). 면책 문구는 코드로 붙인 상태다.
- `contexts`: list[{`id`: str, `source`: `약관`/`기준표`, `text`: str}]. 근거가 없으면 빈 목록이다.

| `trace` 필드 | 타입 | 설명 |
|---|---|---|
| `thread_id` | str | 대화 ID |
| `intent` | str 또는 null | `accident` / `claim_status` / `documents` / `refuse`. 분류 전 오류면 null |
| `route` | str 또는 null | `debate` / `direct` / `followup` / `refused`. 분류 전 오류면 null |
| `rounds` | int | 토론 라운드 수. 토론이 없으면 0 |
| `debate_log` | list[{round: int, agent: str, content: str}] | 라운드별 발언. `agent`는 `A1` / `A2-evidence` / `A3` / `A2-review`. Streamlit 화면의 토론 과정에 쓴다 |
| `claims` | list[{id, kind, coverage, text, verdict, evidence_ids, kept: bool, round: int}] | A3 주장과 판정 |
| `review_notes` | list[{claim_id, text, evidence_ids}] | 답변에 "보험사가 검토할 수 있는 사항"으로 표시한 면책·감액 주장 |
| `unresolved_issues` | list[str] | 해결되지 않은 쟁점 |
| `followup_used` | bool | 이 사고에서 되묻기를 썼는지 |
| `followup_reason` | str 또는 null | `intake` / `debate` |
| `missing_fields` | list[str] | 비어 있는 필수 항목 |
| `deferred_intents` | list[str] | 복합 요청에서 처리하지 않은 요청 |
| `dropped_ids` | list[str] | LLM이 인용했지만 검색 결과에 없어 지운 ID |
| `claim_no` | str 또는 null | 발급·조회한 청구번호 |
| `models_used` | list[str] | 실제로 호출한 모델 ID |
| `error` | str 또는 null | 오류 종류 (아래 예외 처리 참고) |

### 평가셋: `evaluation/test_queries.csv`

- 컬럼: `id`, `type`(`debate`/`direct`/`refuse`/`followup`), `question`, `customer_id`, `expected_route`, `expected_context_ids`, `must_include`, `must_not_include`
- 목록 값은 `;`로 구분하고, 없으면 빈칸으로 둔다.

## 코드 규칙

- **계산과 규칙 판정은 파이썬 코드로 한다.** 과실·담보별 보험금 계산, 보장 판정, 필수 항목 누락 판정, 라운드 제한, A3 주장 필터와 과실 조정, 근거 ID 검증, 청구번호 발급과 중복 방지, 조회 권한, 개인정보 마스킹, 면책 문구 부착이 여기에 해당한다. LLM 판정은 필요할 때 코드 판정 위에 덧붙인다.
- LLM 출력에서 값을 뽑을 때는 구조화 출력(`src/schemas.py`)을 쓴다. 자유 텍스트를 파싱하지 않는다.
- 면책 문구, 거절 문구, 조회 실패 문구, 오류 문구는 `src/guards.py`의 상수로 두고 LLM이 만들지 않는다.
- 데이터 파일은 `data/`에만 둔다. 모든 약관 조항, 기준표 항목, 수정 요소에는 고유 ID를 붙인다.
- **청구 저장소:** 쓰기는 파일 잠금 안에서 한다. 청구번호 발급과 저장도 같은 잠금 안에서 한다. 임시 파일에 쓴 뒤 교체해서 파일 손상을 막는다. 같은 `thread_id`로 이미 청구가 있으면 새로 만들지 않고 기존 번호를 돌려준다.
- 라우팅은 코드 규칙(청구번호 패턴, 서류 키워드)을 먼저 적용하고, 걸리지 않으면 `Intake.intent`(LLM)를 쓴다. 우선순위는 `accident` > `claim_status` > `documents`다.
- 테스트는 표준 라이브러리 `unittest`로 쓰고 `src/tests/`에 둔다. 묶음이 끝날 때마다 전체 테스트를 다시 돌린다.
- 레퍼런스는 먼저 `../day7_practice/` 교육 코드를 참조하고, 없으면 GitHub의 괜찮은 레퍼런스 코드를 참고한다.
  - `final_scenario.py`: RAG·임베딩·MCP 클라이언트 / `plan_and_execute.py`, `long_term_memory.py`: `StateGraph`, checkpointer / `mcp_server.py`: MCP 서버 구조(청구 저장은 새로 작성)
  - `guards_input.py`, `guards_refusal.py`: 가드레일, 거절 문구 / `run_eval.py`, `llm_judge.py`: 평가와 LLM 심사 / `make_data.py`: 가상 JSON 데이터 생성

### 예외 처리

어떤 오류에서도 응답은 세 키 형식을 지키고, 오류 메시지에는 마스킹된 내용만 쓴다.

| 상황 | 처리 | `trace.error` |
|---|---|---|
| LLM 호출 실패 | 대체 모델로 다시 호출한다. 모두 실패하면 고정 오류 문구로 답한다 | `llm_unavailable` |
| 구조화 출력 검증 실패 | 1번 다시 요청한다. 또 실패하면 아래 "에이전트 실패" 규칙으로 진행한다 | `schema_invalid:<에이전트>` |
| 에이전트 실패 | Supervisor 분류 실패: 고정 오류 문구로 답한다. A1 설명 실패: 코드 계산값만 보여준다. A2 검색 실패: 보장 안내를 "확인되지 않음"으로 한다. A3 실패: 주장 없이 토론을 끝낸다. A2 검증 실패: 그 라운드의 새 주장을 모두 `근거없음`으로 본다 | `agent_failed:<에이전트>` |
| RAG 검색 실패 | 근거 없음으로 처리하고 보장 안내는 "확인되지 않음"으로 한다 | `retrieval_failed` |
| MCP 연결·도구 실패, 청구 저장 실패 | 청구번호 없이 안내하고 "잠시 후 다시 시도"를 덧붙인다 | `mcp_failed` / `store_failed` |
| 잘못된 요청(`question` 없음) | 고정 오류 문구로 답한다 | `bad_request` |

### 로그와 호출 횟수

- 로그는 표준 `logging`으로 콘솔에만 남긴다. 로그에는 마스킹 뒤의 텍스트만 쓰고 원문은 남기지 않는다.
- 요청 한 번의 LLM 호출은 재시도를 빼고 최대 10번이다(분류 1 + 1라운드 4 + 2·3라운드 각 2 + 답변 1). 사고 평가 한 번에 약 35~60초가 걸린다. 너무 느리면 SERVICE.md의 축소 ③(토론 1라운드, 최대 6번)을 적용한다.

## 실행 방법

- `mini-pjt/`에서 실행한다. 테스트 `python -m unittest discover src/tests` / API `uvicorn src.server:app --port 8000` / 화면 `streamlit run src/app.py` / 평가 `python -m src.evaluate`
- Bedrock을 부르는 실제 호출 테스트는 `RUN_LIVE=1`일 때만 돈다(평소 전체 테스트는 네트워크 없이 돈다).
- MCP 서버는 A4의 MCP 클라이언트가 stdio로 띄우므로 따로 실행하지 않는다.
- 가상환경은 상위 폴더의 `../.venv`를 쓴다.

## 하지 말 것

- 실제 보험사의 약관 원문, 실제 개인정보, 실제 회사명·브랜드를 쓰지 않는다.
- 과실비율·보험금을 단정적인 숫자로 말하지 않는다. LLM이 숫자를 새로 쓰게 하지 않는다.
- 주장 ID 부여와 중복 판단을 LLM에 맡기지 않는다.
- 마스킹 전 원문을 LLM, 로그, 저장소, 오류 메시지에 넘기지 않는다.
- 로그인, 실제 지급·심사, 청구 수정, 파일 업로드, 사진 분석, 배포는 만들지 않는다.
- 제출 규약의 폴더 구조와 주고받는 형식을 바꾸지 않는다.
- 사용자 허락 없이 새 프레임워크나 라이브러리를 들이지 않는다.
- API 키와 자격 증명 값을 코드나 문서에 적지 않는다. `.env`에서만 읽는다.
- 애매한 요구사항을 추측해서 구현하지 않는다. 사용자에게 묻는다.
- 한 번에 TASKS.md 작업 여러 개를 하지 않는다. 하나를 끝내고 "확인" 방법으로 검증해 `[x]`로 체크한 뒤 멈춘다.
- 사용자가 요청하지 않으면 커밋하지 않는다.
