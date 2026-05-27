"""시맨틱 캐시 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import logging
import os
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import CacheMatch, PipelineContext, SourceDoc, TokenUsage
from app.pipeline.semantic_cache import (
    CACHE_EMBED_INPUT_TYPE,
    _build_cache_key,
    _extract_question_from_cache_key,
    _strip_freeform_from_user_context,
    check_cache,
    fetch_similar_questions_for_user,
)


# ── 헬퍼: 속성 테스트와 fixture 모두에서 사용 ──


def _create_settings(**overrides: str) -> Settings:
    env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        **overrides,
    }
    # clear=True: CI/로컬 셸에 남아있는 환경변수가 테스트로 새는 것을 차단
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)


def _create_bedrock() -> AsyncMock:
    bedrock = AsyncMock()

    # 입력 길이에 맞춰 임베딩 반환 — off-by-one 잡힘 (단건 호출에 2개 받는 buggy
    # mock 회피). PR #62/#64 의 격리 패턴과 일관. input_type 검증은 별도
    # 전용 테스트가 책임 (mock 안에 박으면 먼 곳에서 혼란스러운 실패 유발).
    def _embed(texts, **kwargs):
        return [[0.1 + 0.01 * i] * 1024 for i in range(len(texts))]

    bedrock.embed_texts.side_effect = _embed
    return bedrock


def _create_postgres() -> AsyncMock:
    return AsyncMock()


# ── Fixtures (단위 테스트용) ──


@pytest.fixture
def mock_settings():
    return _create_settings()


@pytest.fixture
def mock_bedrock():
    return _create_bedrock()


@pytest.fixture
def mock_postgres():
    return _create_postgres()


def _make_context(question: str = "테스트 질문") -> PipelineContext:
    return PipelineContext(original_question=question)


def _make_cache_match(
    question: str = "캐시된 질문",
    answer: str = "캐시된 답변",
    similarity: float = 0.97,
    sources: list[dict] | None = None,
    confidence: str = "high",
) -> CacheMatch:
    return CacheMatch(
        question=question,
        answer=answer,
        similarity_score=similarity,
        sources=sources or [{"doc_id": 1, "chunk_id": 1, "doc_name": "학사요람.pdf", "page": 45}],
        confidence=confidence,
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
        postgres = _create_postgres()
        postgres.search_answer_cache.return_value = None

        ctx = _make_context(question)
        await check_cache(ctx, True, bedrock, postgres, settings)

        bedrock.embed_texts.assert_called_once()
        postgres.search_answer_cache.assert_called_once()

    @hyp_settings(max_examples=100)
    @given(question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    async def test_first_message_false_skips_cache_search(self, question):
        """is_first_message=False이면 캐시 탐색이 건너뛰어진다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        ctx = _make_context(question)
        await check_cache(ctx, False, bedrock, postgres, settings)

        bedrock.embed_texts.assert_not_called()
        postgres.search_answer_cache.assert_not_called()

    @hyp_settings(max_examples=100)
    @given(is_first=st.booleans())
    async def test_cache_search_only_when_first_message(self, is_first):
        """is_first_message 값에 따라 캐시 탐색 여부가 결정된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        await check_cache(ctx, is_first, bedrock, postgres, settings)

        if is_first:
            postgres.search_answer_cache.assert_called_once()
        else:
            postgres.search_answer_cache.assert_not_called()


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
        postgres = _create_postgres()
        postgres.search_answer_cache.return_value = _make_cache_match(
            similarity=similarity,
        )

        ctx = _make_context(question)
        result = await check_cache(ctx, True, bedrock, postgres, settings)

        assert result.cache_hit is True
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0

    async def test_cache_miss_keeps_token_usage_zero(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 미스 시에도 시맨틱 캐시 단계에서 토큰이 소비되지 않는다."""
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context()
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is False
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0


# ── 단위 테스트: 캐시 히트/미스/에러 시나리오 ──


class TestCheckCacheUnit:
    async def test_cache_hit_sets_context(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 히트 시 PipelineContext가 올바르게 설정된다."""
        mock_postgres.search_answer_cache.return_value = _make_cache_match(
            answer="최대 4년입니다.",
            sources=[{"doc_id": 1, "chunk_id": 42, "doc_name": "학사요람.pdf", "page": 45}],
        )

        ctx = _make_context("휴학 기간")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is True
        assert result.cached_answer == "최대 4년입니다."
        assert len(result.cached_sources) == 1
        assert result.cached_sources[0].doc_id == 1
        assert result.cached_sources[0].chunk_id == 42
        assert result.cached_sources[0].doc_name == "학사요람.pdf"
        assert result.cached_sources[0].page == 45

    async def test_cache_hit_propagates_confidence(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 히트 시 CacheMatch.confidence 가 PipelineContext.cached_confidence 로 전파된다.

        SSE 시나리오 C(cache) meta 가 시나리오 A(document) 와 동일하게 confidence
        를 노출하도록 박제값을 끝까지 흘려보내는 회귀 잠금.
        """
        mock_postgres.search_answer_cache.return_value = _make_cache_match(
            confidence="medium",
        )

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is True
        assert result.cached_confidence == "medium"

    async def test_cache_miss_leaves_context_unchanged(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 미스 시 PipelineContext의 캐시 필드가 변경되지 않는다."""
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("새로운 질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is False
        assert result.cached_answer is None
        assert result.cached_sources == []

    async def test_embed_texts_called_with_search_query(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """임베딩 호출 시 input_type이 search_query인지 확인.

        Batch 입력은 [cache_key, original_question]. user_context 가 비어있는
        ``_make_context`` 기본 케이스에서는 _build_cache_key 가 question 만
        반환하므로 두 입력이 같다.
        """
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        mock_bedrock.embed_texts.assert_called_once_with(
            ["질문", "질문"], input_type="search_query"
        )

    async def test_threshold_from_settings(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 탐색 시 settings의 CACHE_SIMILARITY_THRESHOLD를 사용한다."""
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        call_kwargs = mock_postgres.search_answer_cache.call_args
        assert call_kwargs.kwargs["threshold"] == mock_settings.CACHE_SIMILARITY_THRESHOLD

    async def test_graceful_degradation_on_bedrock_error(
        self, mock_settings, mock_bedrock, mock_postgres, caplog
    ):
        """Bedrock 임베딩 실패 시 graceful degradation."""
        mock_bedrock.embed_texts.side_effect = RuntimeError("Bedrock 연결 실패")

        ctx = _make_context("질문")
        with caplog.at_level(logging.WARNING):
            result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is False
        assert "시맨틱 캐시 탐색 실패" in caplog.text

    async def test_graceful_degradation_on_postgres_error(
        self, mock_settings, mock_bedrock, mock_postgres, caplog
    ):
        """Postgres 캐시 탐색 실패 시 graceful degradation."""
        mock_postgres.search_answer_cache.side_effect = RuntimeError("DB 연결 실패")

        ctx = _make_context("질문")
        with caplog.at_level(logging.WARNING):
            result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is False
        assert "시맨틱 캐시 탐색 실패" in caplog.text

    async def test_multiple_sources_in_cache_hit(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 히트 시 여러 출처가 올바르게 매핑된다."""
        sources = [
            {"doc_id": 1, "chunk_id": 1, "doc_name": "학사요람.pdf", "page": 45},
            {"doc_id": 1, "chunk_id": 2, "doc_name": "학사요람.pdf", "page": 46},
            {"doc_id": 2, "chunk_id": 3, "doc_name": "학칙.pdf", "page": 10},
        ]
        mock_postgres.search_answer_cache.return_value = _make_cache_match(
            sources=sources,
        )

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert len(result.cached_sources) == 3
        assert result.cached_sources[2].doc_id == 2
        assert result.cached_sources[2].doc_name == "학칙.pdf"
        assert result.cached_sources[2].page == 10


# ── Task 16: 임베딩 컨텍스트 캐싱 ──


class TestEmbeddingCaching:
    """check_cache가 임베딩 직후 PipelineContext에 결과를 보관해 하위 단계가 재사용하도록."""

    async def test_caches_embedding_on_miss(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 미스 시 question_embedding과 embedded_question_text가 컨텍스트에 저장된다.

        Batch 임베딩 [cache_key, question] 의 [1] 이 question_embedding 으로 매핑.
        """
        cache_key_emb = [0.11] * 1024
        question_emb = [0.42] * 1024
        mock_bedrock.embed_texts.side_effect = lambda texts, **k: [cache_key_emb, question_emb]
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.question_embedding == question_emb
        assert result.embedded_question_text == "질문"

    async def test_caches_embedding_on_hit(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """캐시 히트 시에도 동일하게 보관 (일관성)."""
        cache_key_emb = [0.11] * 1024
        question_emb = [0.42] * 1024
        mock_bedrock.embed_texts.side_effect = lambda texts, **k: [cache_key_emb, question_emb]
        mock_postgres.search_answer_cache.return_value = _make_cache_match()

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is True
        assert result.question_embedding == question_emb
        assert result.embedded_question_text == "질문"

    async def test_does_not_cache_when_skipped(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """is_first_message=False (멀티턴) 일 때는 임베딩 자체를 안 하므로 None 유지."""
        ctx = _make_context("멀티턴 질문")
        result = await check_cache(ctx, False, mock_bedrock, mock_postgres, mock_settings)

        mock_bedrock.embed_texts.assert_not_called()
        assert result.question_embedding is None
        assert result.embedded_question_text is None

    async def test_does_not_cache_on_embed_error(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """embed_texts 실패 시 컨텍스트가 None 유지 (graceful degradation 후 새로 시도 가능)."""
        mock_bedrock.embed_texts.side_effect = RuntimeError("Bedrock 실패")

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.question_embedding is None
        assert result.embedded_question_text is None
        assert result.embedded_question_input_type is None
        assert result.cache_key_embedding is None

    async def test_caches_input_type_with_embedding(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """input_type 도 함께 보관 (silent quality degradation 가드)."""
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.embedded_question_input_type == "search_query"

    async def test_preserves_embedding_on_postgres_error(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """Postgres 실패 시에도 question_embedding 은 보존 (embed 후에 실패했으므로).

        하위 단계(vector_search) 가 cache 가 못 한 일을 마저 할 수 있도록 컨텍스트는
        살려둔다. 누군가 except 블록에서 reset 해버리면 silent regression 이라 lock down.
        """
        cache_key_emb = [0.11] * 1024
        question_emb = [0.42] * 1024
        mock_bedrock.embed_texts.side_effect = lambda texts, **k: [cache_key_emb, question_emb]
        mock_postgres.search_answer_cache.side_effect = RuntimeError("DB 실패")

        ctx = _make_context("질문")
        result = await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        assert result.cache_hit is False
        # 임베딩은 이미 만들어졌으니 보존되어야 함
        assert result.question_embedding == question_emb
        assert result.embedded_question_text == "질문"
        assert result.embedded_question_input_type == "search_query"
        # cache_key 임베딩도 보존 — _save_answer_cache 가 재사용
        assert result.cache_key_embedding == cache_key_emb


# ── 캐시 키 user_context 분리 ──
# Cross-user_context cache 누설 차단 (id=10 사고 후속).


class TestCacheKeyIsolation:
    """user_context prefix 를 포함한 캐시 키 임베딩으로 cross-user_context 누설을 차단."""

    def test_cache_key_includes_user_context_prefix(self):
        """_build_cache_key 결과에 [사용자]/[질문] 마커가 모두 포함된다."""
        result = _build_cache_key("소프트웨어학부 3학년 재학", "휴학 신청 방법")

        assert "[사용자]" in result
        assert "[질문]" in result
        assert "소프트웨어학부 3학년 재학" in result
        assert "휴학 신청 방법" in result

    def test_cache_key_omits_prefix_when_user_context_empty(self):
        """user_context 가 비어있으면 question 만 반환 (BE 폴백 경로 호환)."""
        assert _build_cache_key("", "휴학 신청 방법") == "휴학 신청 방법"

    async def test_check_cache_embeds_with_user_context_prefix(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """user_context 가 있으면 embed_texts 의 [0] 입력에 prefix 포함 텍스트."""
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("휴학 신청 방법")
        ctx.user_context = "소프트웨어학부 3학년 재학"
        await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        call = mock_bedrock.embed_texts.call_args
        texts = call.args[0]
        assert len(texts) == 2
        # [0] = 캐시 키 (prefix 포함), [1] = 검색 키 (원본)
        assert "[사용자]" in texts[0]
        assert "소프트웨어학부 3학년 재학" in texts[0]
        assert texts[1] == "휴학 신청 방법"
        assert call.kwargs["input_type"] == "search_query"

    async def test_check_cache_stores_cache_key_and_question_embeddings_separately(
        self, mock_settings, mock_postgres
    ):
        """cache_key_embedding 과 question_embedding 이 별도 키 공간이라 다른 벡터로 보관된다."""
        cache_key_emb = [0.1] * 1024
        question_emb = [0.9] * 1024
        bedrock = AsyncMock()
        bedrock.embed_texts.return_value = [cache_key_emb, question_emb]
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        ctx.user_context = "테스트 컨텍스트"
        result = await check_cache(ctx, True, bedrock, mock_postgres, mock_settings)

        assert result.cache_key_embedding == cache_key_emb
        assert result.question_embedding == question_emb
        assert result.cache_key_embedding != result.question_embedding


# ── _extract_question_from_cache_key (역연산) ──


class TestCacheKeyExtraction:
    """answer_cache.question 컬럼 prefix 텍스트 → 사용자 노출용 원문 추출."""

    def test_extract_with_prefix(self):
        """prefix 포함 텍스트에서 원문만 추출."""
        result = _extract_question_from_cache_key(
            "[사용자] 소프트웨어학부 3학년 재학\n[질문] 휴학 신청 방법"
        )
        assert result == "휴학 신청 방법"

    def test_extract_without_prefix(self):
        """prefix 없는 텍스트는 그대로 반환 (BE 폴백 / 레거시 row 호환)."""
        assert _extract_question_from_cache_key("휴학 신청 방법") == "휴학 신청 방법"

    def test_extract_roundtrip_with_build(self):
        """_build_cache_key 의 역연산 — 정확한 round-trip."""
        original = "휴학 기간이 궁금합니다"
        uc = "전자공학과 4학년 휴학"
        key = _build_cache_key(uc, original)
        assert _extract_question_from_cache_key(key) == original


# ── fetch_similar_questions_for_user (A2 dedup + over-fetch) ──


class TestFetchSimilarQuestionsForUser:
    """fallback 추천 — extract + dedup + over-fetch 캡슐화."""

    async def test_extracts_prefix_from_cache_keys(self):
        """search_similar_questions 결과의 prefix 가 제거된다."""
        postgres = AsyncMock()
        postgres.search_similar_questions.return_value = [
            "[사용자] 소프트웨어학부 3학년 재학\n[질문] 휴학 신청 방법",
            "[사용자] 영문학부 2학년 재학\n[질문] 졸업 요건",
            "복학 절차",  # 레거시 (prefix 없음) 호환
        ]
        result = await fetch_similar_questions_for_user(
            postgres=postgres,
            embedding=[0.1] * 1024,
            top_k=3,
            threshold=0.5,
        )
        assert result == ["휴학 신청 방법", "졸업 요건", "복학 절차"]

    async def test_dedupes_same_question_different_cohorts(self):
        """같은 원문이 cohort 별 별도 row 로 등록된 케이스 → 사용자에겐 1번만."""
        postgres = AsyncMock()
        # cohort prefix 만 다르고 원문 동일한 3개 + 다른 질문 1개
        postgres.search_similar_questions.return_value = [
            "[사용자] 컴공 3학년 재학\n[질문] 휴학 신청 방법",
            "[사용자] 영문 2학년 재학\n[질문] 휴학 신청 방법",
            "[사용자] 전자 4학년 재학\n[질문] 휴학 신청 방법",
            "[사용자] 컴공 3학년 재학\n[질문] 졸업 요건",
        ]
        result = await fetch_similar_questions_for_user(
            postgres=postgres,
            embedding=[0.1] * 1024,
            top_k=3,
            threshold=0.5,
        )
        # "휴학 신청 방법" 은 첫 등장 1번만, 그 다음 "졸업 요건"
        assert result == ["휴학 신청 방법", "졸업 요건"]
        assert result.count("휴학 신청 방법") == 1

    async def test_over_fetches_to_satisfy_top_k_after_dedup(self):
        """top_k * 2 over-fetch → dedup 후 top_k 채우기."""
        postgres = AsyncMock()
        await fetch_similar_questions_for_user(
            postgres=postgres,
            embedding=[0.1] * 1024,
            top_k=3,
            threshold=0.5,
        )
        # over-fetch: top_k=3 → 6 요청
        call = postgres.search_similar_questions.call_args
        assert call.kwargs["top_k"] == 6
        assert call.kwargs["threshold"] == 0.5


# ── CACHE_EMBED_INPUT_TYPE 상수 단일화 검증 ──


class TestCacheEmbedInputType:
    """check_cache / _save_answer_cache 양쪽이 동일한 input_type 사용 보장.

    Task 16 의 (text, input_type) 가드 필드를 두는 대신 모듈 상수 단일화로
    invariant 가 구조적으로 성립함을 잠근다 (외부 리뷰 7.4).
    """

    def test_cache_embed_input_type_is_search_query(self):
        """상수 값 회귀 잠금 — 변경 시 caller 모두 동기 인지 강제."""
        assert CACHE_EMBED_INPUT_TYPE == "search_query"

    async def test_check_cache_uses_constant_for_input_type(
        self, mock_settings, mock_bedrock, mock_postgres
    ):
        """check_cache 가 CACHE_EMBED_INPUT_TYPE 상수를 input_type 으로 전달."""
        mock_postgres.search_answer_cache.return_value = None

        ctx = _make_context("질문")
        ctx.user_context = "테스트 컨텍스트"
        await check_cache(ctx, True, mock_bedrock, mock_postgres, mock_settings)

        call = mock_bedrock.embed_texts.call_args
        assert call.kwargs["input_type"] == CACHE_EMBED_INPUT_TYPE


# ── BE PR #96 후속 — user_context 자유 입력 (additionalInfo/jobDescription) sanitize ──


class TestUserContextFreeformStripping:
    """BE 가 user_context 에 부착하는 250자 자유 입력 (additionalInfo/jobDescription)
    부분이 cache 키에 들어가지 않도록 sanitize. 같은 cohort 사용자가 자유 입력
    차이만으로 cache miss 받지 않도록.
    """

    def test_strip_removes_additional_info(self):
        """학생 추가 정보 구분자 이후 부분 제거."""
        uc = "소프트웨어학부 2024학번 3학년 재학. 추가 정보: 부전공으로 통계학 신청 예정"
        assert _strip_freeform_from_user_context(uc) == "소프트웨어학부 2024학번 3학년 재학"

    def test_strip_removes_job_description(self):
        """직원 담당 업무 구분자 이후 부분 제거."""
        uc = "학사지원과 직원. 담당 업무: 졸업 사정 + 학적 변동 처리"
        assert _strip_freeform_from_user_context(uc) == "학사지원과 직원"

    def test_strip_passes_through_when_no_marker(self):
        """구분자 없는 user_context 는 그대로 (cohort 만 박힌 케이스 / BE 폴백 호환)."""
        uc = "소프트웨어학부 2024학번 3학년 재학"
        assert _strip_freeform_from_user_context(uc) == uc

    def test_strip_handles_empty(self):
        """빈 user_context 는 빈 문자열 그대로."""
        assert _strip_freeform_from_user_context("") == ""

    def test_build_cache_key_excludes_additional_info(self):
        """_build_cache_key 결과에 자유 입력 부분이 포함되지 않는다."""
        uc = "소프트웨어학부 2024학번 3학년 재학. 추가 정보: 부전공 통계학"
        key = _build_cache_key(uc, "휴학 신청 방법")

        assert "추가 정보" not in key
        assert "부전공 통계학" not in key
        assert "소프트웨어학부 2024학번 3학년 재학" in key
        assert "휴학 신청 방법" in key

    def test_build_cache_key_same_cohort_different_freeform_yields_same_key(self):
        """같은 cohort 사용자가 다른 자유 입력 가질 때 cache 키 동일 — cache hit 정합.

        이 PR (BE #96 후속) 의 핵심 의도 잠금.
        """
        uc_a = "소프트웨어학부 2024학번 3학년 재학. 추가 정보: 부전공 통계학"
        uc_b = "소프트웨어학부 2024학번 3학년 재학. 추가 정보: 복수전공 경영학"

        key_a = _build_cache_key(uc_a, "휴학 신청 방법")
        key_b = _build_cache_key(uc_b, "휴학 신청 방법")

        assert key_a == key_b

    def test_build_cache_key_different_cohort_yields_different_key(self):
        """다른 cohort 는 자유 입력 유무와 무관하게 cache 키 분리 — 격리 유지."""
        uc_a = "소프트웨어학부 2024학번 3학년 재학. 추가 정보: 부전공 통계학"
        uc_b = "영문학부 2024학번 3학년 재학. 추가 정보: 부전공 통계학"

        key_a = _build_cache_key(uc_a, "휴학 신청 방법")
        key_b = _build_cache_key(uc_b, "휴학 신청 방법")

        assert key_a != key_b
