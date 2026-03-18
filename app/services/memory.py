"""
세션 기반 대화 히스토리 관리 모듈

- 세션별로 대화 내역을 메모리에 저장
- 대화가 MAX_TURNS를 초과하면 LLM으로 요약하여 컨텍스트 압축
- LangChain ChatMessageHistory 활용
- 세션 만료(TTL) 및 최대 세션 수 제한
"""
import asyncio
import time
from collections import OrderedDict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import MAX_TURNS_BEFORE_SUMMARY

# 세션 관리 상수
MAX_SESSIONS = 200          # 최대 동시 세션 수
SESSION_TTL = 3600          # 세션 만료 시간 (초, 1시간)

# 세션별 대화 히스토리 저장소: {session_id: (InMemoryChatMessageHistory, last_access_time)}
_session_store: OrderedDict[str, tuple[InMemoryChatMessageHistory, float]] = OrderedDict()

# 세션별 요약 저장소: {session_id: (summary_text, last_access_time)}
_summary_store: OrderedDict[str, tuple[str, float]] = OrderedDict()

# 요약용 LLM (가벼운 모델 사용)
_summary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

_summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 대화 내용을 간결하게 요약하는 어시스턴트입니다. "
     "아래 대화 내역을 핵심 정보 위주로 한국어로 요약하세요. "
     "사용자가 어떤 주제에 대해 물었고, 어떤 답변을 받았는지 핵심만 남기세요."),
    ("human", "다음 대화를 요약해주세요:\n\n{conversation}")
])

_summary_chain = _summary_prompt | _summary_llm


def _cleanup_expired():
    """만료된 세션을 제거하고, 최대 세션 수를 초과하면 가장 오래된 세션을 제거"""
    now = time.time()
    # 만료된 세션 제거
    expired = [sid for sid, (_, ts) in _session_store.items() if now - ts > SESSION_TTL]
    for sid in expired:
        _session_store.pop(sid, None)
        _summary_store.pop(sid, None)
    # 최대 세션 수 초과 시 가장 오래된 세션 제거
    while len(_session_store) > MAX_SESSIONS:
        oldest_sid, _ = _session_store.popitem(last=False)
        _summary_store.pop(oldest_sid, None)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """세션 ID에 해당하는 대화 히스토리를 반환합니다. 없으면 새로 생성."""
    _cleanup_expired()
    now = time.time()
    if session_id in _session_store:
        history, _ = _session_store[session_id]
        _session_store[session_id] = (history, now)
        _session_store.move_to_end(session_id)
        return history
    history = InMemoryChatMessageHistory()
    _session_store[session_id] = (history, now)
    return history


def get_session_summary(session_id: str) -> str:
    """세션의 요약본을 반환합니다."""
    entry = _summary_store.get(session_id)
    if entry:
        return entry[0]
    return ""


def add_user_message(session_id: str, message: str):
    """사용자 메시지를 히스토리에 추가합니다."""
    history = get_session_history(session_id)
    history.add_user_message(message)


def add_ai_message(session_id: str, message: str):
    """AI 응답을 히스토리에 추가합니다."""
    history = get_session_history(session_id)
    history.add_ai_message(message)


def _get_turn_count(session_id: str) -> int:
    """현재 세션의 대화 턴 수를 반환합니다. (user+ai 한 쌍 = 1턴)"""
    history = get_session_history(session_id)
    return sum(1 for m in history.messages if isinstance(m, HumanMessage))


def _format_messages_for_summary(messages) -> str:
    """메시지 리스트를 요약용 텍스트로 변환합니다."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"AI: {msg.content}")
    return "\n".join(lines)


def should_summarize(session_id: str) -> bool:
    """요약이 필요한지 판단합니다."""
    return _get_turn_count(session_id) >= MAX_TURNS_BEFORE_SUMMARY


async def summarize_and_compress(session_id: str):
    """
    현재 대화 히스토리를 요약하고, 히스토리를 압축합니다.
    - 기존 요약 + 현재 대화를 합쳐서 새 요약 생성
    - 히스토리는 최근 2턴만 남기고 나머지는 요약으로 대체
    - LLM 호출은 별도 스레드에서 실행 (이벤트 루프 블로킹 방지)
    """
    history = get_session_history(session_id)
    messages = history.messages

    if len(messages) < 2:
        return

    existing_summary = get_session_summary(session_id)
    conversation_text = _format_messages_for_summary(messages)

    if existing_summary:
        conversation_text = f"[이전 대화 요약]\n{existing_summary}\n\n[최근 대화]\n{conversation_text}"

    print(f"[MEMORY] 세션 {session_id}: 대화 요약 중... (턴 수: {_get_turn_count(session_id)})")
    result = await asyncio.to_thread(_summary_chain.invoke, {"conversation": conversation_text})
    _summary_store[session_id] = (result.content, time.time())
    print(f"[MEMORY] 요약 완료: {result.content[:80]}...")

    # 최근 2턴(4개 메시지)만 남기고 히스토리 압축
    keep_count = 4
    recent_messages = messages[-keep_count:] if len(messages) > keep_count else messages
    history.clear()
    for msg in recent_messages:
        history.add_message(msg)


def build_chat_context(session_id: str) -> str:
    """
    LLM에 전달할 대화 컨텍스트를 구성합니다.
    - 요약이 있으면 요약 + 최근 대화
    - 없으면 전체 대화 히스토리
    """
    summary = get_session_summary(session_id)
    history = get_session_history(session_id)

    parts = []
    if summary:
        parts.append(f"[이전 대화 요약]\n{summary}")

    recent = _format_messages_for_summary(history.messages)
    if recent:
        parts.append(f"[최근 대화]\n{recent}")

    return "\n\n".join(parts)
