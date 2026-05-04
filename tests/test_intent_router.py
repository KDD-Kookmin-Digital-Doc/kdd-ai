"""의도 분류 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import PipelineContext, TokenUsage
from app.pipeline.intent_router import classify_intent


# ── 헬퍼 ──


def _create_settings(**overrides: str) -> Settings:
    env = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
        **overrides,
    }
    with patch.dict(os.environ, env):
        return Settings(_env_file=None)


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
    history: list[dict] | None = None,
    user_context: str | None = None,
) -> PipelineContext:
    return PipelineContext(
        original_question=question,
        rewritten_question=rewritten,
        history=history or [],
        user_context=user_context,
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

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == intent

    @hyp_settings(max_examples=50)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    async def test_intent_is_always_academic_or_chitchat(self, question):
        """분류 결과는 항상 academic 또는 chitchat 중 하나이다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(question=question)

        result = await classify_intent(ctx, bedrock, _create_settings())

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

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "academic"


# ── 단위 테스트 ──


class TestClassifyIntentUnit:
    async def test_academic_classification(self):
        """학사규정 질문은 academic으로 분류된다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(question="휴학 기간은 얼마인가요?")

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "academic"

    async def test_chitchat_classification(self):
        """잡담 질문은 chitchat으로 분류된다."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="안녕하세요")

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "chitchat"

    async def test_classify_input_uses_original_not_rewritten(self):
        """rewritten_question이 set돼 있어도 분류 입력은 original_question을 사용한다.

        L안 핵심: history 톤이 잡담을 학사로 비트는 회귀를 차단하기 위해
        rewrite 결과가 아닌 사용자 원문을 분류 대상으로 삼는다.
        """
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(
            question="오늘 날씨 좋네!",
            rewritten="오늘 학사 일정 중 날씨 관련 공지가 있나요?",
        )

        await classify_intent(ctx, bedrock, _create_settings())

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        assert "오늘 날씨 좋네!" in user_text
        assert "오늘 학사 일정 중 날씨 관련 공지가 있나요?" not in user_text

    async def test_classify_input_uses_original_when_no_rewritten(self):
        """rewritten_question이 None이어도 original_question이 입력에 포함된다."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="안녕하세요", rewritten=None)

        await classify_intent(ctx, bedrock, _create_settings())

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        assert "안녕하세요" in user_text

    async def test_case_insensitive(self):
        """LLM 응답의 대소문자를 무시한다."""
        bedrock = _create_bedrock(response="  Academic  ")
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "academic"

    async def test_token_usage_accumulated(self):
        """토큰 사용량이 기존 값에 합산된다."""
        bedrock = _create_bedrock(prompt_tokens=50, completion_tokens=3)
        ctx = _make_context(question="질문")
        ctx.token_usage.prompt_tokens = 100
        ctx.token_usage.completion_tokens = 20
        ctx.token_usage.total_tokens = 120

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.token_usage.prompt_tokens == 150
        assert result.token_usage.completion_tokens == 23
        assert result.token_usage.total_tokens == 173

    async def test_invalid_response_with_extra_text(self):
        """LLM이 부연 설명을 포함하면 academic으로 폴백한다."""
        bedrock = _create_bedrock(response="이 질문은 academic입니다.")
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "academic"

    async def test_empty_response_falls_back(self):
        """LLM이 빈 응답을 반환하면 academic으로 폴백한다."""
        bedrock = _create_bedrock(response="   ")
        ctx = _make_context(question="질문")

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "academic"

    async def test_max_tokens_is_small(self):
        """의도 분류는 짧은 응답만 필요하므로 max_tokens가 작게 설정된다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(question="질문")

        await classify_intent(ctx, bedrock, _create_settings())

        call_kwargs = bedrock.invoke_llm.call_args
        assert call_kwargs.kwargs["max_tokens"] <= 32


# ── L안 회귀: history-aware 분류 (Task 21) ──


class TestHistoryAwareClassification:
    """학사 history 누적 후 잡담 메시지가 chitchat으로 분류되는지 회귀 검증.

    실제 LLM 거동은 통합/수동 테스트로 확인하고, 단위 테스트에서는
    (a) 입력 구성이 정확하고 (b) 응답이 결과에 반영되는지 검증한다.
    """

    _ACADEMIC_HISTORY = [
        {"role": "user", "content": "휴학은 최대 몇 년이야?"},
        {"role": "assistant", "content": "일반휴학은 최대 4년(8학기)입니다."},
    ]

    async def test_history_attached_to_classify_input(self):
        """history가 있으면 분류 LLM 입력에 직전 대화가 포함된다."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(
            question="오늘 날씨 좋네!",
            history=self._ACADEMIC_HISTORY,
        )

        await classify_intent(ctx, bedrock, _create_settings())

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        assert "직전 대화" in user_text
        assert "휴학은 최대 몇 년이야?" in user_text
        assert "일반휴학은 최대 4년" in user_text
        assert "오늘 날씨 좋네!" in user_text

    async def test_history_truncated_by_settings_turn_count(self):
        """history가 settings.INTENT_HISTORY_TURNS 보다 길면 마지막 N개만 첨부된다."""
        long_history = [
            {"role": "user", "content": "수강신청 언제야?"},
            {"role": "assistant", "content": "다음주 월요일입니다."},
            {"role": "user", "content": "장학금 신청 기간은?"},
            {"role": "assistant", "content": "이번달 말까지입니다."},
            {"role": "user", "content": "휴학은 최대 몇 년?"},
            {"role": "assistant", "content": "최대 4년입니다."},
        ]
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="고마워!", history=long_history)

        # settings로 직전 2턴만 첨부하도록 제한 (튜닝 가능 검증)
        await classify_intent(
            ctx, bedrock, _create_settings(INTENT_HISTORY_TURNS="2")
        )

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        assert "휴학은 최대 몇 년?" in user_text
        assert "최대 4년입니다" in user_text
        assert "수강신청" not in user_text
        assert "장학금" not in user_text

    async def test_default_history_turns_keeps_recent_six(self):
        """기본값 INTENT_HISTORY_TURNS=6에서 8턴 history는 마지막 6개만 첨부된다."""
        long_history = [
            {"role": "user", "content": "Q1-수강신청"},
            {"role": "assistant", "content": "A1-수강안내"},
            {"role": "user", "content": "Q2-장학금"},
            {"role": "assistant", "content": "A2-장학안내"},
            {"role": "user", "content": "Q3-휴학"},
            {"role": "assistant", "content": "A3-4년"},
            {"role": "user", "content": "Q4-복학"},
            {"role": "assistant", "content": "A4-복학절차"},
        ]
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="고마워!", history=long_history)

        await classify_intent(ctx, bedrock, _create_settings())  # 기본 6

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        # 마지막 6개(Q2~A4)는 포함, 첫 2개(Q1/A1)는 제외
        assert "Q1-수강신청" not in user_text
        assert "A1-수강안내" not in user_text
        assert "Q2-장학금" in user_text
        assert "A4-복학절차" in user_text

    async def test_long_history_message_truncated(self):
        """history 한 메시지가 200자를 넘으면 절단된다."""
        long_content = "휴학 " * 200  # 600자
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(
            question="고마워",
            history=[
                {"role": "user", "content": "긴질문"},
                {"role": "assistant", "content": long_content},
            ],
        )

        await classify_intent(ctx, bedrock, _create_settings())

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        # 600자 통째로 들어가지 않았는지 — 200자 절단
        assert long_content not in user_text

    async def test_user_context_included_in_classify_input(self):
        """user_context가 분류 LLM 입력에 포함된다."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(
            question="복학 어떻게 해?",
            user_context="소프트웨어학부 3학년 재학생",
        )

        await classify_intent(ctx, bedrock, _create_settings())

        user_text = bedrock.invoke_llm.call_args.kwargs["messages"][0]["content"][0]["text"]
        assert "소프트웨어학부 3학년 재학생" in user_text

    async def test_chitchat_with_academic_history_weather(self):
        """학사 history + '날씨' 잡담은 chitchat (mock 응답 검증)."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="오늘 날씨 좋네!", history=self._ACADEMIC_HISTORY)

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "chitchat"

    async def test_chitchat_with_academic_history_thanks(self):
        """학사 history + '고마워' 잡담은 chitchat."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="고마워!", history=self._ACADEMIC_HISTORY)

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "chitchat"

    async def test_chitchat_with_academic_history_laugh(self):
        """학사 history + 'ㅋㅋㅋ' 잡담은 chitchat."""
        bedrock = _create_bedrock(response="chitchat")
        ctx = _make_context(question="ㅋㅋㅋ", history=self._ACADEMIC_HISTORY)

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "chitchat"

    async def test_academic_followup_with_history(self):
        """학사 history + 후속 학사 질문은 academic."""
        bedrock = _create_bedrock(response="academic")
        ctx = _make_context(
            question="그럼 복학은 어떻게 해?", history=self._ACADEMIC_HISTORY
        )

        result = await classify_intent(ctx, bedrock, _create_settings())

        assert result.intent == "academic"
