"""질문 재작성 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import PipelineContext, TokenUsage
from app.pipeline.query_rewriter import _build_rewrite_messages, rewrite_query


# ── 헬퍼 ──


def _create_settings() -> Settings:
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


def _create_bedrock(
    rewritten: str = "재작성된 질문",
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.invoke_llm.return_value = (
        rewritten,
        TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return bedrock


def _make_context(
    question: str = "테스트 질문",
    history: list[dict] | None = None,
) -> PipelineContext:
    return PipelineContext(
        original_question=question,
        history=history or [],
    )


def _make_history(n: int = 2) -> list[dict]:
    """n턴 대화 히스토리 생성 (user/assistant 교대)."""
    history = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"메시지 {i + 1}"})
    return history


# ── Property 2: 빈 히스토리 시 질문 원본 보존 ──
# Feature: rag-chatbot-ai-server, Property 2: 빈 히스토리 시 질문 원본 보존
# Validates: Requirements 2.2


class TestEmptyHistoryPreservesQuestion:
    """Property 2: 빈 history일 때 원본 질문이 변경 없이 반환되는지 검증."""

    @hyp_settings(max_examples=100)
    @given(question=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    async def test_empty_history_returns_original(self, question):
        """히스토리가 비어있으면 원본 질문이 그대로 rewritten_question에 설정된다."""
        bedrock = _create_bedrock()

        ctx = _make_context(question=question, history=[])
        result = await rewrite_query(ctx, bedrock)

        assert result.rewritten_question == question
        bedrock.invoke_llm.assert_not_called()

    @hyp_settings(max_examples=100)
    @given(question=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    async def test_empty_history_no_token_usage(self, question):
        """히스토리가 비어있으면 토큰 사용량이 0이다."""
        bedrock = _create_bedrock()

        ctx = _make_context(question=question, history=[])
        result = await rewrite_query(ctx, bedrock)

        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0


# ── 히스토리 존재 시 LLM 호출 검증 ──


class TestHistoryTriggersRewrite:
    """히스토리가 있을 때 LLM을 호출하여 질문을 재작성하는지 검증."""

    @hyp_settings(max_examples=50)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        n_turns=st.integers(min_value=1, max_value=10),
    )
    async def test_non_empty_history_calls_llm(self, question, n_turns):
        """히스토리가 1건 이상이면 LLM 호출이 수행된다."""
        bedrock = _create_bedrock()
        history = _make_history(n_turns)

        ctx = _make_context(question=question, history=history)
        await rewrite_query(ctx, bedrock)

        bedrock.invoke_llm.assert_called_once()

    @hyp_settings(max_examples=50)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        is_empty=st.booleans(),
    )
    async def test_llm_called_iff_history_present(self, question, is_empty):
        """히스토리 유무에 따라 LLM 호출 여부가 결정된다."""
        bedrock = _create_bedrock()
        history = [] if is_empty else _make_history(2)

        ctx = _make_context(question=question, history=history)
        await rewrite_query(ctx, bedrock)

        if is_empty:
            bedrock.invoke_llm.assert_not_called()
        else:
            bedrock.invoke_llm.assert_called_once()


# ── 단위 테스트 ──


class TestRewriteQueryUnit:
    async def test_rewrite_sets_rewritten_question(self):
        """LLM 응답이 rewritten_question에 설정된다."""
        bedrock = _create_bedrock(rewritten="휴학 기간은 최대 몇 년인가요?")
        ctx = _make_context(
            question="그건 최대 몇 년이야?",
            history=_make_history(2),
        )

        result = await rewrite_query(ctx, bedrock)

        assert result.rewritten_question == "휴학 기간은 최대 몇 년인가요?"

    async def test_rewrite_strips_whitespace(self):
        """LLM 응답의 앞뒤 공백이 제거된다."""
        bedrock = _create_bedrock(rewritten="  재작성된 질문  \n")
        ctx = _make_context(question="질문", history=_make_history(2))

        result = await rewrite_query(ctx, bedrock)

        assert result.rewritten_question == "재작성된 질문"

    async def test_whitespace_only_rewrite_falls_back_to_original(self):
        """LLM이 공백만 반환하면 원본 질문으로 폴백한다."""
        bedrock = _create_bedrock(rewritten=" \n\t ")
        ctx = _make_context(question="원본 질문", history=_make_history(2))

        result = await rewrite_query(ctx, bedrock)

        assert result.rewritten_question == "원본 질문"

    async def test_token_usage_accumulated(self):
        """토큰 사용량이 기존 값에 합산된다."""
        bedrock = _create_bedrock(prompt_tokens=100, completion_tokens=30)
        ctx = _make_context(question="질문", history=_make_history(2))
        ctx.token_usage.prompt_tokens = 10
        ctx.token_usage.completion_tokens = 5
        ctx.token_usage.total_tokens = 15

        result = await rewrite_query(ctx, bedrock)

        assert result.token_usage.prompt_tokens == 110
        assert result.token_usage.completion_tokens == 35
        assert result.token_usage.total_tokens == 145

    async def test_original_question_unchanged(self):
        """재작성 후에도 original_question은 변경되지 않는다."""
        bedrock = _create_bedrock(rewritten="재작성된 질문")
        original = "원본 질문"
        ctx = _make_context(question=original, history=_make_history(2))

        result = await rewrite_query(ctx, bedrock)

        assert result.original_question == original
        assert result.rewritten_question == "재작성된 질문"

    async def test_system_prompt_passed_to_llm(self):
        """LLM 호출 시 시스템 프롬프트가 전달된다."""
        bedrock = _create_bedrock()
        ctx = _make_context(question="질문", history=_make_history(2))

        await rewrite_query(ctx, bedrock)

        call_kwargs = bedrock.invoke_llm.call_args
        assert "system_prompt" in call_kwargs.kwargs
        assert len(call_kwargs.kwargs["system_prompt"]) > 0

    async def test_history_included_in_messages(self):
        """히스토리 내용이 LLM 메시지에 포함된다."""
        bedrock = _create_bedrock()
        history = [
            {"role": "user", "content": "휴학 신청 방법 알려줘"},
            {"role": "assistant", "content": "학교 홈페이지에서 신청하세요."},
        ]
        ctx = _make_context(question="기간은?", history=history)

        await rewrite_query(ctx, bedrock)

        call_kwargs = bedrock.invoke_llm.call_args
        messages = call_kwargs.kwargs["messages"]
        user_text = messages[0]["content"][0]["text"]
        assert "휴학 신청 방법 알려줘" in user_text
        assert "학교 홈페이지에서 신청하세요." in user_text
        assert "기간은?" in user_text


class TestBuildRewriteMessages:
    def test_message_structure(self):
        """메시지 배열이 Bedrock Converse 형식에 맞는지 확인."""
        history = [{"role": "user", "content": "이전 질문"}]
        messages = _build_rewrite_messages(history, "현재 질문")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "content" in messages[0]
        assert messages[0]["content"][0]["text"]

    def test_role_labels(self):
        """히스토리 역할이 한국어 레이블로 변환된다."""
        history = [
            {"role": "user", "content": "사용자 메시지"},
            {"role": "assistant", "content": "어시스턴트 메시지"},
        ]
        messages = _build_rewrite_messages(history, "질문")
        text = messages[0]["content"][0]["text"]

        assert "사용자: 사용자 메시지" in text
        assert "어시스턴트: 어시스턴트 메시지" in text
