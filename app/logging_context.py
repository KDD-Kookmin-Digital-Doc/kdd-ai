"""세션 단위 로그 트레이싱 컨텍스트 (이슈 #50, PR-50).

`contextvars.ContextVar` 로 현재 요청의 ``session_id`` 를 보관하고,
``logging.Filter`` 가 모든 ``LogRecord`` 에 ``session_id`` 속성을 자동 첨부한다.
``logging.basicConfig`` 의 ``format`` 문자열에 ``%(session_id)s`` 를 넣으면
모든 로그 라인에 자동으로 session 토큰이 박힌다.

Python ``asyncio`` 는 ``create_task`` 시 현재 ``Context`` 를 복사하므로,
``_save_answer_cache`` 같은 background task 에서도 set 한 ``session_id`` 가
자연스럽게 살아남는다.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token

SESSION_ID_VAR: ContextVar[str] = ContextVar("session_id", default="-")


class SessionIdFilter(logging.Filter):
    """모든 ``LogRecord`` 에 현재 컨텍스트의 ``session_id`` 를 첨부한다."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = SESSION_ID_VAR.get()
        return True


def set_session_id(session_id: str) -> Token[str]:
    """현재 컨텍스트에 ``session_id`` 를 설정한다.

    반환된 ``Token`` 을 ``SESSION_ID_VAR.reset(token)`` 으로 넘기면 이전 값으로
    복원된다 (FastAPI dependency ``yield`` 패턴에서 사용).
    """
    return SESSION_ID_VAR.set(session_id)


def reset_session_id(token: Token[str]) -> None:
    """``set_session_id`` 가 반환한 ``Token`` 으로 이전 값으로 복원한다."""
    SESSION_ID_VAR.reset(token)
