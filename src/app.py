"""Streamlit 고객용 데모 화면. run_query()를 직접 호출한다.

- T5-1: 가상 고객 선택, 채팅 입력·출력, 처리 중 표시, thread_id로 대화 이어 가기
- T5-2: "새 대화" 버튼, 고객을 바꾸면 대화 초기화, 라운드별 토론 과정 접기 영역 (미반영 주장은 취소선)
- 처리 중에는 토론 발언을 한 글자씩(타자 효과), 고객 안내문은 LLM 토큰을 받는 대로(실제 스트리밍) 보여주고, 끝나면 답변과 접기 영역으로 바꾼다
- 첫 화면 사고 접수 양식(사고 일시·장소·부상 여부·사고 내용). 대화가 시작되면 사라지고 자유 입력창은 계속 쓴다

실행: mini-pjt/에서  ..\\.venv\\Scripts\\streamlit run src\\app.py
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

# streamlit run src/app.py 로 실행하면 src/ 폴더만 경로에 잡히므로, mini-pjt/를 경로에 넣는다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import agent, tools  # noqa: E402

st.set_page_config(page_title="자동차 사고 보험금 청구 도우미 (데모)", page_icon="🚗", layout="centered")

WAITING = "요청을 확인하는 중입니다…"
# 타자 효과에서 한 번 그릴 때마다 쉬는 시간(초). 0이면 효과 없이 바로 그린다 (화면 테스트에서 0으로 둔다)
TYPE_DELAY = float(os.getenv("MINI_PJT_TYPE_DELAY", "0.012"))
# 노드가 끝났을 때 진행 표시에 띄울 '다음 단계' 문구 (on_event의 node 이벤트)
NEXT_STEP = {
    "supervisor_intake": "접수 정보를 정리했습니다. 처리 방법을 정하는 중…",
    "a1_estimate": "A2 약관·법률 자문가가 약관 근거를 찾는 중…",
    "a2_evidence": "A3 기업 CEO가 반론을 준비하는 중…",
    "a3_opinion": "A2 약관·법률 자문가가 반론을 검증하는 중…",
    "a2_review": "라운드를 정리하는 중…",
    "synthesis": "확인된 내용만 모아 종합 의견을 만드는 중…",
    "a4_after_debate": "청구를 저장했습니다. 고객 안내문을 쓰는 중…",
}
AGENT_LABELS = {
    "A1": "A1 보장·과실·보험금 추정",
    "A2-evidence": "A2 약관·법률 자문가 (근거 검색)",
    "A3": "A3 기업 CEO (비용 절감 반론)",
    "A2-review": "A2 약관·법률 자문가 (반론 검증)",
}
VERDICT_ICONS = {"확인": "✅", "기각": "❌", "근거없음": "❔"}
# 첫 화면 사고 접수 양식: (키, 항목 이름, 입력 안내)
TEMPLATE_FIELDS = [
    ("occurred_at", "사고 일시", "언제 사고가 났는지 (예: 어제 오후 3시쯤)"),
    ("location", "사고 장소", "어디서 사고가 났는지 (예: 강남역 사거리)"),
    ("injury", "부상 여부", "다치신 곳이 있는지 (예: 목이 조금 아파요 / 다친 곳 없어요)"),
    ("description", "사고 내용", "예: 신호대기 중에 뒤에서 차가 들이받았어요 / 차선 변경하다 접촉 사고가 났습니다"),
]


def md(text: str) -> str:
    """마크다운에서 '~'로 감싼 부분이 취소선이 되지 않도록 이스케이프한다 (예: 0~10%, 45만 원 ~ 150만 원)."""
    return text.replace("~", r"\~")


# ---------------------------------------------------------------- 상태
def init_state() -> None:
    # messages: [{"role": "user"/"assistant", "content": str, "trace": dict | None}]
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", None)  # 첫 응답의 trace.thread_id를 받아 이어서 쓴다


def reset_conversation() -> None:
    """대화 내용과 thread_id만 비운다. 고른 고객은 그대로 둔다. 청구 저장소는 건드리지 않는다."""
    st.session_state.messages = []
    st.session_state.thread_id = None
    for key, _, _ in TEMPLATE_FIELDS:  # 접수 양식도 빈 칸으로 되돌린다
        st.session_state.pop(f"tpl_{key}", None)


# ---------------------------------------------------------------- 사이드바
def customer_label(customer_id: str) -> str:
    c = tools.get_customer(customer_id)
    return f"{c.name} ({c.id}) · {c.plan}"


def sidebar():
    with st.sidebar:
        st.header("가상 고객 선택")
        ids = [c.id for c in tools.load_customers()]
        # 고객을 바꾸면 대화를 초기화한다 (SERVICE.md 4장 "고객 변경")
        customer_id = st.selectbox("고객", ids, format_func=customer_label, key="customer_id",
                                   on_change=reset_conversation)
        st.caption("로그인 대신 가상 고객을 고릅니다. 모든 데이터는 가상입니다.")
        st.button("새 대화", on_click=reset_conversation, width="stretch", key="new_chat",
                  help="대화 내용을 비우고 새 사고로 시작합니다. 고른 고객은 유지됩니다.")
    return tools.get_customer(customer_id)


# ---------------------------------------------------------------- 토론 과정 (접지 않고 답변 위에 그대로 보여준다)
DEBATE_TITLE = "**🗣️ 에이전트 토론**"
CLAIMS_TITLE = "**A3 주장별 판정** (반영되지 않은 주장은 취소선)"


def debate_line(e: dict) -> str:
    return f"- **{AGENT_LABELS.get(e['agent'], e['agent'])}**: " + md(e["content"])


def claim_line(c: dict) -> str:
    """주장 한 줄. 최종 답변에 반영되지 않은 주장은 취소선으로 긋는다.
    본문의 '~'는 먼저 이스케이프하고(md), 취소선 기호 '~~'는 그 바깥에 붙인다."""
    basis = ", ".join(c["evidence_ids"]) or "없음"
    icon = VERDICT_ICONS.get(c["verdict"], "·")
    if c["kept"]:
        return f"- {icon} " + md(f"`{c['id']}` {c['kind']}/{c['coverage']} — **{c['verdict']}** "
                                 f"(근거 {basis}, 반영): {c['text']}")
    body = md(f"`{c['id']}` {c['kind']}/{c['coverage']} — {c['verdict'] or '판정 없음'} (근거 {basis}): {c['text']}")
    return f"- {icon} ~~{body}~~ _(미반영)_"


def render_debate(trace: dict | None) -> None:
    """지난 답변을 다시 그릴 때: 라운드별 발언과 주장별 판정을 답변 위에 펼친 채로 보여준다."""
    if not trace or not trace.get("debate_log"):
        return
    st.markdown(DEBATE_TITLE)
    last = None
    for e in trace["debate_log"]:
        if e["round"] != last:
            st.markdown(f"**{e['round']}라운드**")
            last = e["round"]
        st.markdown(debate_line(e))
    if trace.get("claims"):
        st.markdown(CLAIMS_TITLE)
        for c in trace["claims"]:
            st.markdown(claim_line(c))
    st.divider()


# ---------------------------------------------------------------- 실시간 출력 (타자 효과)
def typewrite(placeholder, text: str, prefix: str = "", start: int = 0, hurry: bool = False) -> None:
    """text를 글자 단위로 늘려 가며 그린다. 긴 글은 몇 글자씩 늘려 한 덩어리를 약 2초 안에 끝낸다.
    hurry면(밀린 이벤트가 많으면) 더 빨리 쓴다. TYPE_DELAY가 0이면 효과 없이 바로 그린다(테스트용)."""
    if TYPE_DELAY > 0:
        step = max(1, (len(text) - start) // (40 if hurry else 150))
        for i in range(start + step, len(text), step):
            placeholder.markdown(prefix + md(text[:i]) + " ▌")
            time.sleep(TYPE_DELAY)
    placeholder.markdown(prefix + md(text))


class LiveAnswer:
    """처리 중 화면. run_query는 작업 스레드에서 돌리고, 메인 스레드가 이벤트 큐를 받아 나오는 순서대로 이어 쓴다.
    - 토론 발언(구조화 출력이라 완성된 뒤 옴): 오는 즉시 타자 효과로 쓴다
    - A3 주장별 판정(종합 노드 뒤): 한 줄씩 쓴다. 미반영은 취소선
    - 고객 안내문(일반 텍스트 생성): LLM 토큰을 받는 대로 쓴다 (실제 스트리밍)
    접는 영역 없이 쓴 내용이 그대로 남는다. 작업 스레드에서는 Streamlit을 부르지 않는다."""

    def __init__(self):
        self.body = st.container()  # 쓴 내용이 차례로 쌓이는 곳
        self.progress = st.empty()  # 맨 아래 한 줄: 지금 무엇을 하는 중인지
        self.progress.caption(f"⏳ {WAITING}")
        self.last_round = None
        self.debate_started = False
        self.answer_ph = None
        self.streamed = ""  # 토큰으로 받은 안내문

    def run(self, fn) -> dict:
        """fn(on_event)를 작업 스레드에서 돌리며, 이벤트가 오는 대로 화면에 쓴다. fn의 결과를 돌려준다."""
        events: queue.Queue = queue.Queue()
        box: dict = {}

        def work():
            try:
                box["resp"] = fn(events.put)
            except BaseException as exc:  # noqa: BLE001 - 메인 스레드에서 다시 낸다
                box["error"] = exc
            finally:
                events.put(None)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while (event := events.get()) is not None:
            self.handle(event, hurry=events.qsize() > 3)
        worker.join()
        if "error" in box:
            raise box["error"]
        return box["resp"]

    def handle(self, event: dict, hurry: bool = False) -> None:
        kind = event["type"]
        if kind == "node":
            if event["node"] == "start_round" and event.get("round"):
                self.progress.caption(f"⏳ {event['round']}라운드 토론 중…")
            elif event["node"] in NEXT_STEP:
                self.progress.caption(f"⏳ {NEXT_STEP[event['node']]}")
        elif kind == "debate":
            if not self.debate_started:
                self.body.markdown(DEBATE_TITLE)
                self.debate_started = True
            if event["round"] != self.last_round:
                self.body.markdown(f"**{event['round']}라운드**")
                self.last_round = event["round"]
            who = AGENT_LABELS.get(event["agent"], event["agent"])
            typewrite(self.body.empty(), event["content"], prefix=f"- **{who}**: ", hurry=hurry)
        elif kind == "claims":
            if event["claims"]:
                self.body.markdown(CLAIMS_TITLE)
                for c in event["claims"]:
                    self.body.markdown(claim_line(c))
                    if TYPE_DELAY > 0:
                        time.sleep(TYPE_DELAY * 15)
            self.body.divider()
        elif kind == "token":
            if self.answer_ph is None:
                self.progress.caption("⏳ 고객 안내문을 쓰는 중…")
                self.answer_ph = self.body.empty()
            before, self.streamed = len(self.streamed), self.streamed + event["text"]
            typewrite(self.answer_ph, self.streamed, start=before, hurry=hurry)

    def finish(self, answer: str) -> None:
        """최종 답변으로 마무리한다. 이미 흘려보낸 안내문과 같은 앞부분은 두고 나머지(숫자 블록 등)를 이어 쓴다.
        코드가 안내문에서 숫자·근거 ID 문장을 지웠으면 달라진 곳부터 다시 쓴다."""
        self.progress.empty()
        if self.answer_ph is None:
            self.answer_ph = self.body.empty()
        start = len(os.path.commonprefix([self.streamed, answer]))
        typewrite(self.answer_ph, answer, start=start)


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_debate(m.get("trace"))
            st.markdown(md(m["content"]))
        else:
            # 양식으로 보낸 여러 줄 입력이 한 줄로 붙지 않게 한다 (마크다운 줄바꿈)
            st.markdown(md(m["content"]).replace("\n", "  \n"))


# ---------------------------------------------------------------- 첫 화면 접수 양식
def compose_question(values: dict) -> str:
    """양식 입력을 '항목: 내용' 줄로 합친다. 빈 칸은 뺀다 (빠진 필수 항목은 Supervisor가 되묻는다)."""
    return "\n".join(f"{label}: {values[key].strip()}" for key, label, _ in TEMPLATE_FIELDS
                     if values.get(key, "").strip())


def template_form() -> str | None:
    """대화가 비어 있을 때만 보이는 사고 접수 양식. 제출하면 합친 질문을 돌려준다."""
    area = st.empty()  # 제출하면 이 영역을 비워, 같은 화면에서 바로 대화로 넘어간다
    with area.container():
        st.markdown("사고가 나셨다면 아래 양식에 적어 주세요. 사고 일시·장소·부상 여부를 함께 주시면 바로 평가해 드립니다. "
                    "청구 진행 상태나 필요 서류는 아래 입력창에 바로 물어보셔도 됩니다.")
        with st.form("accident_template", border=True):
            values = {}
            for key, label, hint in TEMPLATE_FIELDS:
                widget = st.text_area if key == "description" else st.text_input
                values[key] = widget(label, placeholder=hint, key=f"tpl_{key}")
            submitted = st.form_submit_button("사고 접수하기", type="primary", width="stretch")
    if not submitted:
        return None
    if not values["description"].strip():
        st.warning("사고 내용을 적어 주세요.")
        return None
    area.empty()
    return compose_question(values)


# ---------------------------------------------------------------- 화면
def main() -> None:
    init_state()
    customer = sidebar()

    st.title("🚗 자동차 사고 보험금 청구 도우미")
    st.caption("교육용 데모입니다. 실제 보험사와 연동하지 않으며, 안내는 참고용 추정입니다.")

    form_question = template_form() if not st.session_state.messages else None
    for m in st.session_state.messages:
        render_message(m)

    chat_question = st.chat_input("사고 내용이나 궁금한 점을 입력하세요 (예: 신호 대기 중에 뒤에서 차가 들이받았어요)")
    question = form_question or chat_question
    if not question:
        return

    user_msg = {"role": "user", "content": question, "trace": None}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        live = LiveAnswer()  # 토론 과정과 안내문을 나오는 대로 이어 쓴다
        customer_id, thread_id = customer.id, st.session_state.thread_id  # 작업 스레드에는 값만 넘긴다
        resp = live.run(lambda on_event: agent.run_query(question, customer_id=customer_id, thread_id=thread_id,
                                                         on_event=on_event))
        live.finish(resp["answer"])
        st.session_state.thread_id = resp["trace"]["thread_id"]
    st.session_state.messages.append({"role": "assistant", "content": resp["answer"], "trace": resp["trace"]})


main()
