"""입력 가드레일과 고정 문구 상수.

- 개인정보 마스킹: 주민등록번호, 전화번호, 계좌번호, 차량번호를 정규식으로 가린다.
  run_query()의 가장 앞에서 적용하고, 이후에는 마스킹된 텍스트만 쓴다.
- 고정 문구: 면책·오류·조회 실패 문구는 LLM이 만들지 않고 여기 상수를 쓴다.
- 거절 판정 (T7-1): 보험사기 유도·무관한 질문을 코드 규칙으로 먼저 거르고, 걸리지 않으면 LLM 분류가 보완한다.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- 고정 문구
DISCLAIMER = "※ 본 안내는 유사 사례 기준의 참고용 추정이며, 실제 과실비율과 보험금은 보험사 심사 후 확정됩니다."
ERROR_MESSAGE = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
BAD_REQUEST_MESSAGE = "질문 내용이 비어 있습니다. 사고 내용이나 궁금한 점을 입력해 주세요."
CLAIM_NOT_FOUND = "해당 청구를 찾을 수 없습니다."
RETRY_LATER = "청구 처리 시스템에 일시적인 문제가 있어 청구번호를 발급하지 못했습니다. 잠시 후 다시 시도해 주세요."
NOT_CONFIRMED = "확인되지 않음"
# 사고 유형이 기준표에 없을 때: 처음 한 번은 상황을 묻고(되묻기 1번에 포함), 그래도 없으면 안내로 마무리한다
UNMATCHED_ASK = (
    "말씀하신 사고 상황에 맞는 기준을 가상 기준표에서 찾지 못해 과실과 보험금 범위는 " + NOT_CONFIRMED + "입니다. "
    "사고 상황을 조금 더 자세히 알려 주세요. "
    "(예: 두 차가 모두 움직이고 있었는지, 상대 차가 주차·정차해 있었는지, 어느 쪽이 어디를 부딪혔는지)"
)
UNMATCHED_FINAL = (
    "말씀하신 사고 상황은 가상 기준표에 맞는 유형이 없어 과실과 보험금 범위를 안내해 드리기 어렵습니다. "
    "실제 과실과 보상은 보험사 상담과 심사로 정해집니다."
)
UNMATCHED_TYPES_HEAD = "가상 기준표에서 안내할 수 있는 비슷한 사고 유형입니다. 이 중 맞는 상황이 있으면 그 상황으로 다시 말씀해 주세요."
# 안내문의 모든 문장이 숫자·근거 ID 때문에 지워졌을 때 대신 쓰는 문장
ANSWER_FALLBACK = "말씀하신 사고를 기준으로 예상되는 보장과 과실·보험금 범위를 아래에 정리했습니다."
REVIEW_NOTE_FALLBACK = "아래 근거 조항에 따라 보험사가 면책·감액을 검토할 수 있습니다."
DIRECT_NOT_READY = "청구 조회와 서류 안내 기능은 준비 중입니다. 사고 내용을 알려 주시면 보장과 과실·보험금 범위를 먼저 안내해 드리겠습니다."
# 거절 고정 문구 (교육 코드 day7_practice/guards_refusal.py처럼 종류별로 둔다. LLM이 만들지 않는다)
REFUSAL_MESSAGES = {
    "fraud": ("죄송합니다. 사실과 다르게 사고나 손해를 꾸미거나 부풀리는 일은 도와드릴 수 없습니다. "
              "실제 사고 내용을 알려 주시면 보장과 과실·보험금 범위를 안내해 드리겠습니다."),
    "off_topic": ("죄송합니다. 그 주제는 제가 도와드릴 수 있는 범위가 아닙니다. "
                  "자동차 사고 접수, 보장·과실·보험금 안내, 필요 서류 안내, 청구 조회를 도와드릴 수 있습니다."),
}
REFUSAL_TEMP = REFUSAL_MESSAGES["off_topic"]  # 예전 이름 (T2-3 테스트 호환)
ASK_CLAIM_NO = "조회하실 청구번호(CLM-YYYYMMDD-NNN 형식)를 알려 주세요."
CLAIM_STATUS_TEXT = {
    "접수": "청구가 접수되어 검토를 기다리고 있습니다.",
    "서류보완": "추가 서류가 필요합니다. 안내받은 서류를 보내 주시면 심사가 이어집니다.",
    "심사중": "보험사가 손해와 과실을 심사하고 있습니다.",
    "지급완료": "보험금 지급이 끝났습니다.",
    "부지급": "심사 결과 보험금이 지급되지 않는 것으로 결정되었습니다.",
}
INTENT_LABELS = {"accident": "사고 평가", "claim_status": "청구 조회", "documents": "서류 안내", "refuse": "기타 요청"}


def deferred_note(intents: list[str]) -> str:
    """복합 요청에서 이번에 처리하지 않은 요청에 대한 안내."""
    if not intents:
        return ""
    names = ", ".join(f"'{INTENT_LABELS[i]}'" for i in intents)
    return f"함께 말씀하신 {names} 요청은 한 번에 하나씩 도와드리고 있어 이번에는 안내하지 못했어요. 따로 물어봐 주세요."


def with_disclaimer(answer: str) -> str:
    """답변 끝에 면책 문구를 붙인다. 이미 붙어 있으면 다시 붙이지 않는다."""
    if answer.rstrip().endswith(DISCLAIMER):
        return answer
    return f"{answer.rstrip()}\n\n{DISCLAIMER}"


# ---------------------------------------------------------------- 거절 판정 (T7-1, 코드 규칙)
# 보험사기 유도: 고객 자신이 사실을 꾸미거나 부풀리려는 표현. 고객이 피해를 알리는 말("상대가 일부러 받은 것 같아요",
# "보험사기 당한 것 같아요")은 걸리지 않게 '꾸미는 행위' 쪽 표현만 잡는다.
FRAUD_PATTERNS = [
    re.compile(r"(보험금|수리비|치료비|견적|진단|피해|손해|금액).{0,12}(부풀|뻥튀기|과장|더 ?받|늘려|올려)"),
    re.compile(r"(부풀|뻥튀기|과장)(려|해|하|서).{0,12}(청구|받|신고|접수)"),
    re.compile(r"(허위|거짓|가짜|조작|위조).{0,10}(청구|진단|서류|사고|견적|영수증|신고|진술)"),
    re.compile(r"(진단서|영수증|견적서|서류).{0,8}(위조|조작|꾸며|만들어)"),
    re.compile(r"(블랙박스|영상|기록).{0,10}(지우|지워|삭제|없애)"),
    re.compile(r"(일부러|고의로|짜고)\s*(사고를?\s*(내|낼|냈|낸)|부딪(히|혀|쳐|치))"),
    re.compile(r"(안 ?아픈데|멀쩡한데|다치지도 않았는데).{0,15}(입원|진단|치료|병원)"),
    re.compile(r"나이롱\s*환자"),
]
OFF_TOPIC_WORDS = ["주식", "코인", "비트코인", "로또", "날씨", "맛집", "메뉴", "레시피", "영화", "드라마", "노래",
                   "게임", "연애", "숙제", "코딩", "프로그래밍", "여행 추천", "운세"]
SERVICE_WORDS = ["사고", "차", "보험", "과실", "청구", "서류", "추돌", "부딪", "받혔", "수리", "다쳤", "보상", "CLM-"]


def refusal_kind(text: str) -> str | None:
    """코드 규칙으로 거절 대상을 찾는다. 보험사기 유도면 'fraud', 무관한 질문이면 'off_topic', 아니면 None.

    무관한 질문은 서비스와 관련된 말(사고·차·보험 등)이 하나도 없을 때만 거절한다. 그 밖은 LLM 분류가 보완한다.
    """
    if any(p.search(text) for p in FRAUD_PATTERNS):
        return "fraud"
    if any(w in text for w in OFF_TOPIC_WORDS) and not any(w in text for w in SERVICE_WORDS):
        return "off_topic"
    return None


# ---------------------------------------------------------------- 개인정보 마스킹
# 숫자 앞뒤에 다른 숫자가 붙어 있으면 맞추지 않는다(더 긴 숫자의 일부를 잘못 가리지 않도록).
_NB = r"(?<!\d)"
_NA = r"(?!\d)"
_SEP = r"[-.\s]?"

# 주민등록번호·외국인등록번호: 생년월일 6자리(실제 월·일) + 성별 1자리(1~8) + 6자리
_RRN = re.compile(_NB + r"\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\s?-?\s?[1-8]\d{6}" + _NA)

# 전화번호: 휴대폰(010 등), 지역번호(02, 031 등), 대표번호(1588 등), 국제 표기(+82)
_PHONE = re.compile(
    "|".join(
        [
            _NB + r"01[016789]" + _SEP + r"\d{3,4}" + _SEP + r"\d{4}" + _NA,
            _NB + r"0(?:2|[3-6][1-5]|70)" + _SEP + r"\d{3,4}" + _SEP + r"\d{4}" + _NA,
            r"\+82" + _SEP + r"(?:0?1[016789]|0?2|0?[3-6][1-5])" + _SEP + r"\d{3,4}" + _SEP + r"\d{4}" + _NA,
            _NB + r"1[5-9]\d{2}-\d{4}" + _NA,
        ]
    )
)

# 계좌번호: 하이픈으로 나뉜 숫자 묶음 3~4개, 숫자 합계 10~14자리일 때만(날짜 2026-11-04 등 제외)
_ACCOUNT_GROUPED = re.compile(_NB + r"\d{2,6}(?:-\d{2,7}){2,3}" + _NA)
# 하이픈 없는 계좌번호: '계좌'·'통장' 뒤에 오는 10~14자리 숫자
_ACCOUNT_PLAIN = re.compile(r"((?:계좌|통장)[^\d\n]{0,12})(\d{10,14})" + _NA)

# 차량번호: (지역명) 2~3자리 숫자 + 번호판 한글 1자 + 4자리 숫자. 예) 12가3456, 123가 4567, 서울12가3456
_PLATE_HANGUL = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호"
_PLATE_REGION = "서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주"
_PLATE = re.compile(
    r"(?:(?:" + _PLATE_REGION + r")\s?)?" + _NB + r"\d{2,3}\s?[" + _PLATE_HANGUL + r"]\s?\d{4}" + _NA
)

MASKS = {
    "rrn": "[주민번호]",
    "phone": "[전화번호]",
    "account": "[계좌번호]",
    "plate": "[차량번호]",
}


def _account_grouped(match: re.Match) -> str:
    digits = sum(c.isdigit() for c in match.group(0))
    return MASKS["account"] if 10 <= digits <= 14 else match.group(0)


def mask_pii(text: str) -> str:
    """개인정보를 마스킹한 텍스트를 돌려준다. 긴 형식부터 순서대로 적용한다."""
    if not text:
        return text
    # '계좌'·'통장' 뒤 숫자는 주민번호 형식과 겹쳐도 계좌번호로 본다.
    out = _ACCOUNT_PLAIN.sub(lambda m: m.group(1) + MASKS["account"], text)
    out = _RRN.sub(MASKS["rrn"], out)
    out = _PHONE.sub(MASKS["phone"], out)
    out = _ACCOUNT_GROUPED.sub(_account_grouped, out)
    out = _PLATE.sub(MASKS["plate"], out)
    return out


def find_pii(text: str) -> list[str]:
    """마스킹되지 않은 개인정보 종류를 찾아 돌려준다. 평가에서 '개인정보 노출 0건' 검사에 쓴다.

    mask_pii와 같은 순서로 찾고, 앞에서 찾은 부분은 지운 뒤 다음 종류를 찾는다.
    (예: 010-1234-5678을 전화번호와 계좌번호로 두 번 세지 않는다)
    """
    if not text:
        return []
    found = []
    rest = text
    if _ACCOUNT_PLAIN.search(rest):
        found.append("account")
        rest = _ACCOUNT_PLAIN.sub(lambda m: m.group(1), rest)
    for kind, pattern in (("rrn", _RRN), ("phone", _PHONE)):
        if pattern.search(rest):
            found.append(kind)
            rest = pattern.sub(" ", rest)
    if "account" not in found and any(
        10 <= sum(c.isdigit() for c in m.group(0)) <= 14 for m in _ACCOUNT_GROUPED.finditer(rest)
    ):
        found.append("account")
    if _PLATE.search(rest):
        found.append("plate")
    return found
