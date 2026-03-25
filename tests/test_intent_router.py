"""의도 분류 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.models.pipeline import PipelineContext, TokenUsage
from app.pipeline.intent_router import classify_intent


# ── 헬퍼 ──


def _create_bedrock(
    response: str = "academic",
    prompt_tokens: int = 30,
    completion_tokens: int = 3,
) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.invoke_llm.return_value = (
        response,
        TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return bedrock


def _make_context(
    question: str = "테스트 질문",
    rewritten: str | None = None,
) -> PipelineContext:
    return PipelineContext(
        original_question=question,
        rewritten_question=rewritten,
    )


# ── Property 1: 의도 분류 기반 파이프라인 라우팅 ──
# Feature: rag-chatbot-ai-server, Property 1: 의도 분류 기반 파이프라인 라우팅
# Validates: Requirements 1.2, 1.3


class TestIntentRoutingProperty:
    """Property 1: 의도 분류 결과에 따른 파이프라인 라우팅 검증.

    academic → 벡터 검색 실행 경로, chitchat → 벡터 검색 우회 경로.
    이 테스트에서는 의도 분류 결과가 올바르게 PipelineContext.intent에
    설정되는지를 검증한다 (실제 라우팅은 오케스트레이터 책임).
    """

    @hyp_settings(max_examples=100)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        intent=st.sampled_from(["academic", "chitchat"]),
    )
    async def test_valid_intent_always_set(self, question, intent):
        """LLM이 유효한 의도를 반환하면 그대로 설정된다."""
        bedrock = _create_bedrock(response=intent)
        ctx = _make_context(question=question, rewritten=question)

        result = await classify_intent(ctx, bedrock)

        assert result.intent == intent

    @hyp_settings(max_examples=50)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    async def test_intent_is_always_academic_or_chitchat(self, question):
        """분류 결과는 항상 academic 또는 chitchat 중 하나이다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(question=question)

        result = await classify_intent(ctx, bedrock)

        assert result.intent in {"academic", "chitchat"}

    @hyp_settings(max_examples=50)
    @given(
        invalid_response=st.text(min_size=1, max_size=100).filter(
            lambda x: x.strip().lower() not in {"academic", "chitchat"}
        ),
    )
    async def test_invalid_intent_falls_back_to_academic(self, invalid_response):
        """LLM이 유효하지 않은 응답을 반환하면 academic으로 폴백한다."""
        bedrock = _create_bedrock(response=invalid_response)
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock)

        assert result.intent == "academic"


# ── 단위 테스트 ──


class TestClassifyIntentUnit:
    async def test_academic_classification(self):
        """학사규정 질문은 academic으로 분류된다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(question="휴학 기간은 얼마인가요?")

        result = await classify_intent(ctx, bedrock)

        assert result.intent == "academic"

    async def test_chitchat_classification(self):
        """잡담 질문은 chitchat으로 분류된다."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="안녕하세요")

        result = await classify_intent(ctx, bedrock)

        assert result.intent == "chitchat"

    async def test_uses_rewritten_question_when_available(self):
        """rewritten_question이 있으면 그것을 사용한다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(
            question="그건 몇 년이야?",
            rewritten="휴학 기간은 최대 몇 년인가요?",
        )

        await classify_intent(ctx, bedrock)

        call_kwargs = bedrock.invoke_llm.call_args
        messages = call_kwargs.kwargs["messages"]
        user_text = messages[0]["content"][0]["text"]
        assert user_text == "휴학 기간은 최대 몇 년인가요?"

    async def test_falls_back_to_original_when_no_rewritten(self):
        """rewritten_question이 None이면 original_question을 사용한다."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="안녕하세요", rewritten=None)

        await classify_intent(ctx, bedrock)

        call_kwargs = bedrock.invoke_llm.call_args
        messages = call_kwargs.kwargs["messages"]
        user_text = messages[0]["content"][0]["text"]
        assert user_text == "안녕하세요"

    async def test_case_insensitive(self):
        """LLM 응답의 대소문자를 무시한다."""
        bedrock = _create_bedrock(response="  Academic  ")
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock)

        assert result.intent == "academic"

    async def test_token_usage_accumulated(self):
        """토큰 사용량이 기존 값에 합산된다."""
        bedrock = _create_bedrock(prompt_tokens=50, completion_tokens=3)
        ctx = _make_context(question="질문")
        ctx.token_usage.prompt_tokens = 100
        ctx.token_usage.completion_tokens = 20
        ctx.token_usage.total_tokens = 120

        result = await classify_intent(ctx, bedrock)

        assert result.token_usage.prompt_tokens == 150
        assert result.token_usage.completion_tokens == 23
        assert result.token_usage.total_tokens == 173

    async def test_invalid_response_with_extra_text(self):
        """LLM이 부연 설명을 포함하면 academic으로 폴백한다."""
        bedrock = _create_bedrock(response="이 질문은 academic입니다.")
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock)

        assert result.intent == "academic"

    async def test_empty_response_falls_back(self):
        """LLM이 빈 응답을 반환하면 academic으로 폴백한다."""
        bedrock = _create_bedrock(response="   ")
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock)

        assert result.intent == "academic"

    async def test_max_tokens_is_small(self):
        """의도 분류는 짧은 응답만 필요하므로 max_tokens가 작게 설정된다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(question="질문")

        await classify_intent(ctx, bedrock)

        call_kwargs = bedrock.invoke_llm.call_args
        assert call_kwargs.kwargs["max_tokens"] <= 32
