"""서버 시작 검증 테스트. 임베딩 차원 정합성 + 외부 의존성 연결 검증."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from app.config import Settings
from app.startup import validate_startup


# ── 헬퍼 ──


def _create_settings(embedding_dimension: int = 1024) -> Settings:
    with patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
        "EMBEDDING_DIMENSION": str(embedding_dimension),
    }):
        return Settings(_env_file=None)


def _create_bedrock(
    embed_dimension: int = 1024,
    llm_ok: bool = True,
    embedding_ok: bool = True,
) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.health_check_llm.return_value = llm_ok
    bedrock.health_check_embedding.return_value = embedding_ok
    bedrock.embed_texts.return_value = [[0.1] * embed_dimension]
    return bedrock


def _create_supabase(db_ok: bool = True) -> AsyncMock:
    supabase = AsyncMock()
    supabase.health_check.return_value = db_ok
    return supabase


# ── Property 14: 서버 시작 차원 검증 ──
# Validates: Requirements 11.2, 11.3


class TestEmbeddingDimensionValidation:
    """Property 14: DB 차원과 임베딩 차원 일치 시 정상 시작, 불일치 시 시작 중단 검증."""

    @pytest.mark.asyncio
    async def test_matching_dimension_passes(self):
        """임베딩 차원 일치 → 정상 시작 (예외 없음)."""
        settings = _create_settings(embedding_dimension=1024)
        bedrock = _create_bedrock(embed_dimension=1024)
        supabase = _create_supabase()

        await validate_startup(settings, bedrock, supabase)

        bedrock.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_mismatching_dimension_raises(self):
        """임베딩 차원 불일치 → RuntimeError."""
        settings = _create_settings(embedding_dimension=1024)
        bedrock = _create_bedrock(embed_dimension=512)
        supabase = _create_supabase()

        with pytest.raises(RuntimeError, match="임베딩 차원 불일치"):
            await validate_startup(settings, bedrock, supabase)

    @pytest.mark.asyncio
    async def test_dimension_check_uses_settings_value(self):
        """환경변수 EMBEDDING_DIMENSION 값을 기준으로 검증한다."""
        settings = _create_settings(embedding_dimension=768)
        bedrock = _create_bedrock(embed_dimension=768)
        supabase = _create_supabase()

        await validate_startup(settings, bedrock, supabase)

    @pytest.mark.asyncio
    async def test_embed_failure_during_dimension_check_raises(self):
        """차원 검증 중 임베딩 호출 실패 → RuntimeError."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        bedrock.embed_texts.side_effect = Exception("Bedrock timeout")
        supabase = _create_supabase()

        with pytest.raises(RuntimeError, match="Bedrock 호출 실패"):
            await validate_startup(settings, bedrock, supabase)


# ── 외부 의존성 연결 검증 ──


class TestDependencyValidation:
    """외부 서비스 연결 실패 시 서버 시작 중단 검증."""

    @pytest.mark.asyncio
    async def test_supabase_failure_raises(self):
        """Supabase 연결 실패 → RuntimeError."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(db_ok=False)

        with pytest.raises(RuntimeError, match="Supabase"):
            await validate_startup(settings, bedrock, supabase)

    @pytest.mark.asyncio
    async def test_bedrock_llm_failure_raises(self):
        """Bedrock LLM 연결 실패 → RuntimeError."""
        settings = _create_settings()
        bedrock = _create_bedrock(llm_ok=False)
        supabase = _create_supabase()

        with pytest.raises(RuntimeError, match="Bedrock LLM"):
            await validate_startup(settings, bedrock, supabase)

    @pytest.mark.asyncio
    async def test_bedrock_embedding_failure_raises(self):
        """Bedrock Embedding 연결 실패 → RuntimeError."""
        settings = _create_settings()
        bedrock = _create_bedrock(embedding_ok=False)
        supabase = _create_supabase()

        with pytest.raises(RuntimeError, match="Bedrock Embedding"):
            await validate_startup(settings, bedrock, supabase)

    @pytest.mark.asyncio
    async def test_all_healthy_passes(self):
        """모든 의존성 정상 → 예외 없이 완료."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        await validate_startup(settings, bedrock, supabase)

        supabase.health_check.assert_called_once()
        bedrock.health_check_llm.assert_called_once()
        bedrock.health_check_embedding.assert_called_once()
        bedrock.embed_texts.assert_called_once()


# ── 검증 순서 ──


class TestValidationOrder:
    """의존성 검증은 순차적으로 수행되며, 앞선 검증 실패 시 이후 검증을 건너뛴다."""

    @pytest.mark.asyncio
    async def test_supabase_failure_skips_bedrock_checks(self):
        """Supabase 실패 시 Bedrock 헬스체크는 호출되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(db_ok=False)

        with pytest.raises(RuntimeError):
            await validate_startup(settings, bedrock, supabase)

        bedrock.health_check_llm.assert_not_called()
        bedrock.health_check_embedding.assert_not_called()
        bedrock.embed_texts.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_skips_embedding_checks(self):
        """Bedrock LLM 실패 시 임베딩 헬스체크와 차원 검증은 호출되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock(llm_ok=False)
        supabase = _create_supabase()

        with pytest.raises(RuntimeError):
            await validate_startup(settings, bedrock, supabase)

        bedrock.health_check_embedding.assert_not_called()
        bedrock.embed_texts.assert_not_called()
