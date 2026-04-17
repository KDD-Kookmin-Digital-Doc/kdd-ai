"""LLM 생성 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, PropertyMock

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import PipelineContext, SearchResult, TokenUsage
from app.pipeline.llm_generator import (
    _CHITCHAT_SYSTEM_PROMPT,
    _estimate_tokens,
    build_academic_messages,
    build_chitchat_messages,
    generate_response,
    truncate_history,
)


# ── 헬퍼 ──


def _create_settings() -> Settings:
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


def _create_bedrock(tokens: list[str] | None = None) -> AsyncMock:
    bedrock = AsyncMock()

    async def _mock_stream(*args, **kwargs):
        for t in (tokens or ["안녕", "하세요"]):
            yield t

    bedrock.invoke_llm_stream = _mock_stream
    bedrock.last_stream_usage = TokenUsage(
        prompt_tokens=100, completion_tokens=20, total_tokens=120
    )
    return bedrock


def _make_context(
    question: str = "테스트 질문",
    rewritten: str | None = None,
    intent: str = "academic",
    user_context: str = "소프트웨어학부 3학년",
    history: list[dict] | None = None,
    search_results: list[SearchResult] | None = None,
) -> PipelineContext:
    return PipelineContext(
        original_question=question,
        rewritten_question=rewritten,
        intent=intent,
        user_context=user_context,
        history=history or [],
        search_results=search_results,
    )


def _make_search_result(
    content: str = "제1조 내용",
    doc_name: str = "학사요람.pdf",
    page: int = 10,
    enforcement_date: str = "2026-03-01",
) -> SearchResult:
    return SearchResult(
        chunk_id=1,
        doc_id=1,
        content=content,
        metadata={
            "doc_name": doc_name,
            "page": page,
            "enforcement_date": enforcement_date,
        },
        similarity_score=0.85,
    )


def _make_history(n: int, content_length: int = 10) -> list[dict]:
    """n턴 대화 히스토리 생성."""
    history = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": "가" * content_length})
    return history


# ── Property 19: user_context 프롬프트 포함 ──
# Feature: rag-chatbot-ai-server, Property 19: user_context 프롬프트 포함
# Validates: Requirements 6.2


class TestUserContextInPrompt:
    """Property 19: LLM 프롬프트에 user_context 정보가 포함되는지 검증."""

    @hyp_settings(max_examples=50)
    @given(
        user_context=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    async def test_user_context_included_in_academic_prompt(self, user_context):
        """학사규정 경로에서 user_context가 시스템 프롬프트에 포함된다."""
        settings = _create_settings()
        ctx = _make_context(
            user_context=user_context,
            search_results=[_make_search_result()],
        )

        system_prompt, _ = build_academic_messages(ctx, settings)

        assert user_context in system_prompt

    @hyp_settings(max_examples=50)
    @given(
        user_context=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    async def test_user_context_not_in_chitchat_prompt(self, user_context):
        """잡담 경로에서는 user_context가 프롬프트에 포함되지 않는다."""
        settings = _create_settings()
        ctx = _make_context(intent="chitchat", user_context=user_context)

        system_prompt, _ = build_chitchat_messages(ctx, settings)

        assert system_prompt == _CHITCHAT_SYSTEM_PROMPT


# ── history truncation 테스트 ──
# Validates: Requirements 6.1 (컨텍스트 윈도우 관리)


class TestHistoryTruncation:
    def test_within_budget_keeps_all(self):
        """예산 내이면 전체 history를 유지한다."""
        history = _make_history(4, content_length=10)
        # 10자 / 1.5 / 0.8 ≈ 9토큰 per msg, 4개 = 36토큰
        result = truncate_history(history, budget_tokens=100)

        assert len(result) == 4

    def test_over_budget_removes_oldest(self):
        """예산 초과 시 가장 오래된 메시지부터 제거한다."""
        history = _make_history(4, content_length=15)
        # 15자 / 1.5 / 0.8 = 13토큰 per msg, 4개 = 52토큰
        # budget=25 → 1개만 수용 가능
        result = truncate_history(history, budget_tokens=25)

        assert len(result) < 4
        # 남은 메시지는 history 끝부분 (최신)
        assert result[-1] == history[-1]

    def test_single_message_over_budget_returns_empty(self):
        """1개만 남았는데도 예산 초과면 빈 리스트를 반환한다."""
        history = [{"role": "user", "content": "가" * 300}]
        # 300자 / 1.5 / 0.8 = 250토큰, budget=10
        result = truncate_history(history, budget_tokens=10)

        assert result == []

    def test_exact_budget_keeps_all(self):
        """예산과 정확히 일치하면 전체를 유지한다."""
        history = [{"role": "user", "content": "가" * 15}]
        # 15자 / 1.5 / 0.8 = 12.5 → ceil = 13토큰
        result = truncate_history(history, budget_tokens=13)

        assert len(result) == 1

    def test_zero_budget_returns_empty(self):
        """예산이 0이면 빈 리스트를 반환한다."""
        history = _make_history(3)
        result = truncate_history(history, budget_tokens=0)

        assert result == []

    def test_empty_history_returns_empty(self):
        """빈 history는 그대로 빈 리스트를 반환한다."""
        result = truncate_history([], budget_tokens=1000)

        assert result == []

    def test_preserves_newest_messages(self):
        """truncation 후 남은 메시지는 최신 메시지이다."""
        history = [
            {"role": "user", "content": "첫번째"},
            {"role": "assistant", "content": "두번째"},
            {"role": "user", "content": "세번째"},
            {"role": "assistant", "content": "네번째"},
        ]
        # 각 ~2~3토큰, budget을 아주 작게 설정
        result = truncate_history(history, budget_tokens=5)

        if result:
            assert result[-1] == history[-1]

    def test_char_approximation_accuracy(self):
        """한국어 토큰 근사치 검증 (안전 마진 0.8 적용)."""
        assert _estimate_tokens("가나다") == 3  # 3자 / 1.5 / 0.8 = 2.5 → ceil = 3
        assert _estimate_tokens("가") == 1  # 1자 / 1.5 / 0.8 = 0.83 → ceil = 1
        assert _estimate_tokens("가나다라마바") == 5  # 6자 / 1.5 / 0.8 = 5.0 → ceil = 5
        assert _estimate_tokens("") == 0


# ── 단위 테스트 ──


class TestBuildAcademicMessages:
    def test_doc_context_format(self):
        """문서 컨텍스트가 올바른 포맷으로 구성된다."""
        settings = _create_settings()
        ctx = _make_context(
            search_results=[
                _make_search_result(
                    content="제2조 휴학",
                    doc_name="2026_학사요람.pdf",
                    page=45,
                    enforcement_date="2026-03-01",
                ),
            ],
        )

        system_prompt, _ = build_academic_messages(ctx, settings)

        assert "2026_학사요람.pdf" in system_prompt
        assert "시행일: 2026-03-01" in system_prompt
        assert "페이지: 45" in system_prompt
        assert "제2조 휴학" in system_prompt

    def test_multi_year_rule_in_prompt(self):
        """다중 연도 판단 규칙이 시스템 프롬프트에 포함된다."""
        settings = _create_settings()
        ctx = _make_context(search_results=[_make_search_result()])

        system_prompt, _ = build_academic_messages(ctx, settings)

        assert "입학년도" in system_prompt
        assert "enforcement_date" in system_prompt
        assert "최근" in system_prompt

    def test_question_in_messages(self):
        """재작성된 질문이 마지막 메시지로 포함된다."""
        settings = _create_settings()
        ctx = _make_context(
            question="원본",
            rewritten="재작성된 질문",
            search_results=[_make_search_result()],
        )

        _, messages = build_academic_messages(ctx, settings)

        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"][0]["text"] == "재작성된 질문"

    def test_history_included_in_messages(self):
        """history가 메시지 배열에 포함된다."""
        settings = _create_settings()
        ctx = _make_context(
            history=[
                {"role": "user", "content": "이전 질문"},
                {"role": "assistant", "content": "이전 답변"},
            ],
            search_results=[_make_search_result()],
        )

        _, messages = build_academic_messages(ctx, settings)

        # history 2개 + 현재 질문 1개 = 3개
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "이전 질문"
        assert messages[1]["content"][0]["text"] == "이전 답변"

    def test_no_user_context_shows_placeholder(self):
        """user_context가 빈 문자열이면 플레이스홀더가 포함된다."""
        settings = _create_settings()
        ctx = _make_context(
            user_context="",
            search_results=[_make_search_result()],
        )

        system_prompt, _ = build_academic_messages(ctx, settings)

        assert "(정보 없음)" in system_prompt


class TestBuildChitchatMessages:
    def test_uses_fixed_system_prompt(self):
        """잡담 경로는 고정된 시스템 프롬프트를 사용한다."""
        settings = _create_settings()
        ctx = _make_context(intent="chitchat")

        system_prompt, _ = build_chitchat_messages(ctx, settings)

        assert "2문장 이내" in system_prompt

    def test_question_in_messages_no_history(self):
        """history가 없으면 질문 1개만 메시지에 포함된다."""
        settings = _create_settings()
        ctx = _make_context(
            question="안녕하세요",
            rewritten="안녕하세요",
            intent="chitchat",
        )

        _, messages = build_chitchat_messages(ctx, settings)

        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "안녕하세요"

    def test_history_included_in_messages(self):
        """history가 있으면 메시지 배열에 포함된다."""
        settings = _create_settings()
        ctx = _make_context(
            question="네 반가워요",
            rewritten="네 반가워요",
            intent="chitchat",
            history=[
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "안녕하세요! 학사규정 궁금한 점 있으면 질문해주세요."},
            ],
        )

        _, messages = build_chitchat_messages(ctx, settings)

        # history 2개 + 현재 질문 1개 = 3개
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "안녕"
        assert messages[1]["content"][0]["text"] == "안녕하세요! 학사규정 궁금한 점 있으면 질문해주세요."
        assert messages[2]["content"][0]["text"] == "네 반가워요"


class TestGenerateResponse:
    async def test_academic_uses_answer_model(self):
        """학사규정 경로는 answer 모델을 사용한다."""
        settings = _create_settings()
        bedrock = AsyncMock()
        call_log: list[dict] = []

        async def _mock_stream(*args, **kwargs):
            call_log.append(kwargs)
            if kwargs.get("usage_out") is not None:
                kwargs["usage_out"].total_tokens = 0
            yield "답변"

        bedrock.invoke_llm_stream = _mock_stream

        ctx = _make_context(
            intent="academic",
            search_results=[_make_search_result()],
        )

        async for _ in generate_response(ctx, bedrock, settings):
            pass

        assert call_log[0]["model"] == "answer"

    async def test_chitchat_uses_light_model(self):
        """잡담 경로는 light 모델을 사용한다."""
        settings = _create_settings()
        bedrock = AsyncMock()
        call_log: list[dict] = []

        async def _mock_stream(*args, **kwargs):
            call_log.append(kwargs)
            if kwargs.get("usage_out") is not None:
                kwargs["usage_out"].total_tokens = 0
            yield "인사"

        bedrock.invoke_llm_stream = _mock_stream

        ctx = _make_context(intent="chitchat")

        async for _ in generate_response(ctx, bedrock, settings):
            pass

        assert call_log[0]["model"] == "light"

    async def test_token_usage_accumulated(self):
        """토큰 사용량이 기존 값에 합산된다."""
        settings = _create_settings()
        bedrock = AsyncMock()

        async def _mock_stream(*args, **kwargs):
            usage_out = kwargs.get("usage_out")
            if usage_out is not None:
                usage_out.prompt_tokens = 100
                usage_out.completion_tokens = 20
                usage_out.total_tokens = 120
            yield "토큰"

        bedrock.invoke_llm_stream = _mock_stream

        ctx = _make_context(
            intent="academic",
            search_results=[_make_search_result()],
        )
        ctx.token_usage.prompt_tokens = 50
        ctx.token_usage.completion_tokens = 3
        ctx.token_usage.total_tokens = 53

        async for _ in generate_response(ctx, bedrock, settings):
            pass

        assert ctx.token_usage.prompt_tokens == 150
        assert ctx.token_usage.completion_tokens == 23
        assert ctx.token_usage.total_tokens == 173

    async def test_yields_tokens(self):
        """스트리밍 토큰이 yield된다."""
        settings = _create_settings()
        bedrock = AsyncMock()

        async def _mock_stream(*args, **kwargs):
            if kwargs.get("usage_out") is not None:
                kwargs["usage_out"].total_tokens = 0
            yield "안녕"
            yield "하세요"

        bedrock.invoke_llm_stream = _mock_stream

        ctx = _make_context(
            intent="academic",
            search_results=[_make_search_result()],
        )

        tokens = []
        async for t in generate_response(ctx, bedrock, settings):
            tokens.append(t)

        assert tokens == ["안녕", "하세요"]
