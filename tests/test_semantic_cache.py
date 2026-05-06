"""시맨틱 캐시 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import logging
import os
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import CacheMatch, PipelineContext, SourceDoc, TokenUsage
from app.pipeline.semantic_cache import check_cache


# ── 헬퍼: 속성 테스트와 fixture 모두에서 사용 ──


def _create_settings() -> Settings:
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


def _create_bedrock() -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.embed_texts.return_value = [[0.1] * 1024]
    return bedrock


def _create_supabase() -> AsyncMock:
    return AsyncMock()


# ── Fixtures (단위 테스트용) ──


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


@pytest.fixture
def mock_bedrock():
    return _create_bedrock()


@pytest.fixture
def mock_supabase():
    return _create_supabase()


def _make_context(question: str = "테스트 질문") -> PipelineContext:
    return PipelineContext(original_question=question)


def _make_cache_match(
    question: str = "캐시된 질문",
    answer: str = "캐시된 답변",
    similarity: float = 0.97,
    sources: list[dict] | None = None,
) -> CacheMatch:
    return CacheMatch(
        question=question,
        answer=answer,
        similarity_score=similarity,
        sources=sources or [{"doc_id": 1, "chunk_id": 1, "doc_name": "학사요람.pdf", "page": 45}],
    )


# ── Property 3: 시맨틱 캐시 조건부 탐색 ──
# Feature: rag-chatbot-ai-server, Property 3: 시맨틱 캐시 조건부 탐색
# Validates: Requirements 3.2, 3.4


class TestSemanticCacheConditionalSearch:
    """Property 3: is_first_message 값에 따른 캐시 탐색 수행 여부 검증."""

    @hyp_settings(max_examples=100)
    @given(question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    async def test_first_message_true_triggers_cache_search(self, question):
        """is_first_message=True이면 캐시 탐색이 수행된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.search_answer_cache.return_value = None

        ctx = _make_context(question)
        await check_cache(ctx, True, bedrock, supabase, settings)

        bedrock.embed_texts.assert_called_once()
        supabase.search_answer_cache.assert_called_once()

    @hyp_settings(max_examples=100)
    @given(question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    async def test_first_message_false_skips_cache_search(self, question):
        """is_first_message=False이면 캐시 탐색이 건너뛰어진다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context(question)
        await check_cache(ctx, False, bedrock, supabase, settings)

        bedrock.embed_texts.assert_not_called()
        supabase.search_answer_cache.assert_not_called()

    @hyp_settings(max_examples=100)
    @given(is_first=st.booleans())
    async def test_cache_search_only_when_first_message(self, is_first):
        """is_first_message 값에 따라 캐시 탐색 여부가 결정된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        await check_cache(ctx, is_first, bedrock, supabase, settings)

        if is_first:
            supabase.search_answer_cache.assert_called_once()
        else:
            supabase.search_answer_cache.assert_not_called()


# ── Property 4: 캐시 히트 시 토큰 사용량 제로 ──
# Feature: rag-chatbot-ai-server, Property 4: 캐시 히트 시 토큰 사용량 제로
# Validates: Requirements 3.3


class TestCacheHitTokenZero:
    """Property 4: 캐시 히트 응답의 모든 토큰 값이 0인지 검증."""

    @hyp_settings(max_examples=100)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        similarity=st.floats(min_value=0.95, max_value=1.0),
    )
    async def test_cache_hit_keeps_token_usage_zero(self, question, similarity):
        """캐시 히트 시 토큰 사용량이 모두 0이어야 한다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.search_answer_cache.return_value = _make_cache_match(
            similarity=similarity,
        )

        ctx = _make_context(question)
        result = await check_cache(ctx, True, bedrock, supabase, settings)

        assert result.cache_hit is True
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0

    async def test_cache_miss_keeps_token_usage_zero(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 미스 시에도 시맨틱 캐시 단계에서 토큰이 소비되지 않는다."""
        mock_supabase.search_answer_cache.return_value = None

        ctx = _make_context()
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.cache_hit is False
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0


# ── 단위 테스트: 캐시 히트/미스/에러 시나리오 ──


class TestCheckCacheUnit:
    async def test_cache_hit_sets_context(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 히트 시 PipelineContext가 올바르게 설정된다."""
        mock_supabase.search_answer_cache.return_value = _make_cache_match(
            answer="최대 4년입니다.",
            sources=[{"doc_id": 1, "chunk_id": 42, "doc_name": "학사요람.pdf", "page": 45}],
        )

        ctx = _make_context("휴학 기간")
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.cache_hit is True
        assert result.cached_answer == "최대 4년입니다."
        assert len(result.cached_sources) == 1
        assert result.cached_sources[0].doc_id == 1
        assert result.cached_sources[0].chunk_id == 42
        assert result.cached_sources[0].doc_name == "학사요람.pdf"
        assert result.cached_sources[0].page == 45

    async def test_cache_miss_leaves_context_unchanged(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 미스 시 PipelineContext의 캐시 필드가 변경되지 않는다."""
        mock_supabase.search_answer_cache.return_value = None

        ctx = _make_context("새로운 질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.cache_hit is False
        assert result.cached_answer is None
        assert result.cached_sources == []

    async def test_embed_texts_called_with_search_query(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """임베딩 호출 시 input_type이 search_query인지 확인."""
        mock_supabase.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        mock_bedrock.embed_texts.assert_called_once_with(
            ["질문"], input_type="search_query"
        )

    async def test_threshold_from_settings(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 탐색 시 settings의 CACHE_SIMILARITY_THRESHOLD를 사용한다."""
        mock_supabase.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        call_kwargs = mock_supabase.search_answer_cache.call_args
        assert call_kwargs.kwargs["threshold"] == mock_settings.CACHE_SIMILARITY_THRESHOLD

    async def test_graceful_degradation_on_bedrock_error(
        self, mock_settings, mock_bedrock, mock_supabase, caplog
    ):
        """Bedrock 임베딩 실패 시 graceful degradation."""
        mock_bedrock.embed_texts.side_effect = RuntimeError("Bedrock 연결 실패")

        ctx = _make_context("질문")
        with caplog.at_level(logging.WARNING):
            result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.cache_hit is False
        assert "시맨틱 캐시 탐색 실패" in caplog.text

    async def test_graceful_degradation_on_supabase_error(
        self, mock_settings, mock_bedrock, mock_supabase, caplog
    ):
        """Supabase 캐시 탐색 실패 시 graceful degradation."""
        mock_supabase.search_answer_cache.side_effect = RuntimeError("DB 연결 실패")

        ctx = _make_context("질문")
        with caplog.at_level(logging.WARNING):
            result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.cache_hit is False
        assert "시맨틱 캐시 탐색 실패" in caplog.text

    async def test_multiple_sources_in_cache_hit(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 히트 시 여러 출처가 올바르게 매핑된다."""
        sources = [
            {"doc_id": 1, "chunk_id": 1, "doc_name": "학사요람.pdf", "page": 45},
            {"doc_id": 1, "chunk_id": 2, "doc_name": "학사요람.pdf", "page": 46},
            {"doc_id": 2, "chunk_id": 3, "doc_name": "학칙.pdf", "page": 10},
        ]
        mock_supabase.search_answer_cache.return_value = _make_cache_match(
            sources=sources,
        )

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert len(result.cached_sources) == 3
        assert result.cached_sources[2].doc_id == 2
        assert result.cached_sources[2].doc_name == "학칙.pdf"
        assert result.cached_sources[2].page == 10


# ── Task 16: 임베딩 컨텍스트 캐싱 ──


class TestEmbeddingCaching:
    """check_cache가 임베딩 직후 PipelineContext에 결과를 보관해 하위 단계가 재사용하도록."""

    async def test_caches_embedding_on_miss(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 미스 시 question_embedding과 embedded_question_text가 컨텍스트에 저장된다."""
        cached_embedding = [0.42] * 1024
        mock_bedrock.embed_texts.return_value = [cached_embedding]
        mock_supabase.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.question_embedding == cached_embedding
        assert result.embedded_question_text == "질문"

    async def test_caches_embedding_on_hit(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """캐시 히트 시에도 동일하게 보관 (일관성)."""
        cached_embedding = [0.42] * 1024
        mock_bedrock.embed_texts.return_value = [cached_embedding]
        mock_supabase.search_answer_cache.return_value = _make_cache_match()

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.cache_hit is True
        assert result.question_embedding == cached_embedding
        assert result.embedded_question_text == "질문"

    async def test_does_not_cache_when_skipped(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """is_first_message=False (멀티턴) 일 때는 임베딩 자체를 안 하므로 None 유지."""
        ctx = _make_context("멀티턴 질문")
        result = await check_cache(ctx, False, mock_bedrock, mock_supabase, mock_settings)

        mock_bedrock.embed_texts.assert_not_called()
        assert result.question_embedding is None
        assert result.embedded_question_text is None

    async def test_does_not_cache_on_embed_error(
        self, mock_settings, mock_bedrock, mock_supabase
    ):
        """embed_texts 실패 시 컨텍스트가 None 유지 (graceful degradation 후 새로 시도 가능)."""
        mock_bedrock.embed_texts.side_effect = RuntimeError("Bedrock 실패")

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_supabase, mock_settings)

        assert result.question_embedding is None
        assert result.embedded_question_text is None
