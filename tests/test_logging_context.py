"""세션 단위 로그 트레이싱 테스트 (이슈 #50).

`SESSION_ID_VAR` + `SessionIdFilter` 가 의도대로 작동하는지 회귀 잠금.
가장 중요한 케이스는 `asyncio.create_task` 안에서도 값이 살아남는 것
(`_save_answer_cache` 가 background task 라 이게 깨지면 캐시 저장 시점 로그가
session 추적 불가).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.logging_context import (
    SESSION_ID_VAR,
    SessionIdFilter,
    reset_session_id,
    set_session_id,
)


class TestSessionIdContextVar:
    def test_default_is_dash(self):
        """미설정 상태에서는 default `-` 가 반환된다 (startup/lifespan 안전)."""
        # 이전 테스트가 set 했을 수 있으니 신선한 토큰 + reset
        token = set_session_id("-")
        try:
            assert SESSION_ID_VAR.get() == "-"
        finally:
            reset_session_id(token)

    def test_set_and_get(self):
        token = set_session_id("sess-X")
        try:
            assert SESSION_ID_VAR.get() == "sess-X"
        finally:
            reset_session_id(token)

    def test_reset_restores_previous(self):
        outer = set_session_id("outer")
        try:
            inner = set_session_id("inner")
            assert SESSION_ID_VAR.get() == "inner"
            reset_session_id(inner)
            assert SESSION_ID_VAR.get() == "outer"
        finally:
            reset_session_id(outer)


class TestSessionIdFilter:
    def test_filter_attaches_session_id(self):
        token = set_session_id("sess-Y")
        try:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="hello", args=(), exc_info=None,
            )
            assert SessionIdFilter().filter(record) is True
            assert record.session_id == "sess-Y"
        finally:
            reset_session_id(token)

    def test_filter_uses_default_when_unset(self):
        # 명시적으로 default 로 set
        token = set_session_id("-")
        try:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="msg", args=(), exc_info=None,
            )
            SessionIdFilter().filter(record)
            assert record.session_id == "-"
        finally:
            reset_session_id(token)

    def test_logger_emits_with_session_id(self, caplog):
        """logger.info() 호출 시 LogRecord 에 session_id 가 첨부된다."""
        logger = logging.getLogger("app.test_session_filter")
        # 핸들러에 SessionIdFilter 직접 부착 — caplog 의 핸들러는 root 일 수도
        logger.addFilter(SessionIdFilter())
        token = set_session_id("sess-Z")
        try:
            with caplog.at_level(logging.INFO, logger=logger.name):
                logger.info("test message")
            # caplog records 의 마지막 항목 확인
            assert any(
                getattr(r, "session_id", None) == "sess-Z" and r.message == "test message"
                for r in caplog.records
            )
        finally:
            reset_session_id(token)


class TestAsyncioTaskPropagation:
    """가장 중요한 회귀 케이스 — `_save_answer_cache` 가 asyncio.create_task 로
    분기될 때 session_id 가 살아남아야 한다.
    """

    async def test_create_task_inherits_session_id(self):
        token = set_session_id("sess-bg")
        try:
            captured: list[str] = []

            async def background():
                captured.append(SESSION_ID_VAR.get())

            task = asyncio.create_task(background())
            await task

            assert captured == ["sess-bg"]
        finally:
            reset_session_id(token)

    async def test_set_after_create_task_does_not_affect(self):
        """task 생성 후 부모가 session_id 를 바꿔도 task 는 생성 시점 값 유지."""
        outer = set_session_id("sess-A")
        try:
            captured: list[str] = []

            async def background():
                # 의도적 await 으로 부모가 set 하고 반환할 시간 확보
                await asyncio.sleep(0)
                captured.append(SESSION_ID_VAR.get())

            task = asyncio.create_task(background())
            # 부모가 다른 session 으로 set 해도 task 는 sess-A 로 시작했음
            new_token = set_session_id("sess-B")
            try:
                await task
            finally:
                reset_session_id(new_token)

            assert captured == ["sess-A"]
        finally:
            reset_session_id(outer)
