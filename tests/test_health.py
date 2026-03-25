"""헬스체크 엔드포인트 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.api.health import health_check


# ── 헬퍼 ──


def _create_bedrock(llm_ok: bool = True, embedding_ok: bool = True) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.health_check_llm.return_value = llm_ok
    bedrock.health_check_embedding.return_value = embedding_ok
    return bedrock


def _create_supabase(db_ok: bool = True) -> AsyncMock:
    supabase = AsyncMock()
    supabase.health_check.return_value = db_ok
    return supabase


# ── Property 17: 헬스체크 상태 일관성 ──
# Validates: Requirements 13.3, 13.4


class TestHealthCheckConsistency:
    """Property 17: 의존성 상태 조합에 따른 HTTP 코드 및 status 값 검증."""

    @hyp_settings(max_examples=20)
    @given(
        db_ok=st.booleans(),
        llm_ok=st.booleans(),
        embedding_ok=st.booleans(),
    )
    async def test_status_matches_dependencies(self, db_ok, llm_ok, embedding_ok):
        """모든 의존성 정상이면 healthy/200, 하나라도 비정상이면 unhealthy/503."""
        bedrock = _create_bedrock(llm_ok=llm_ok, embedding_ok=embedding_ok)
        supabase = _create_supabase(db_ok=db_ok)

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        all_ok = db_ok and llm_ok and embedding_ok

        if all_ok:
            assert response.status_code == 200
            assert body["status"] == "healthy"
        else:
            assert response.status_code == 503
            assert body["status"] == "unhealthy"

    @hyp_settings(max_examples=20)
    @given(
        db_ok=st.booleans(),
        llm_ok=st.booleans(),
        embedding_ok=st.booleans(),
    )
    async def test_dependency_fields_match_individual_status(self, db_ok, llm_ok, embedding_ok):
        """각 의존성 필드가 개별 상태와 일치한다."""
        bedrock = _create_bedrock(llm_ok=llm_ok, embedding_ok=embedding_ok)
        supabase = _create_supabase(db_ok=db_ok)

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        deps = body["dependencies"]
        assert deps["vector_db"] == ("healthy" if db_ok else "unhealthy")
        assert deps["bedrock_llm"] == ("healthy" if llm_ok else "unhealthy")
        assert deps["bedrock_embedding"] == ("healthy" if embedding_ok else "unhealthy")


# ── 단위 테스트 ──


class TestHealthCheckUnit:
    async def test_all_healthy(self):
        """모든 의존성 정상 시 200 + healthy."""
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        assert response.status_code == 200
        assert body["status"] == "healthy"
        assert body["dependencies"]["vector_db"] == "healthy"
        assert body["dependencies"]["bedrock_llm"] == "healthy"
        assert body["dependencies"]["bedrock_embedding"] == "healthy"

    async def test_db_unhealthy(self):
        """DB만 비정상이면 503 + unhealthy."""
        bedrock = _create_bedrock()
        supabase = _create_supabase(db_ok=False)

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        assert response.status_code == 503
        assert body["status"] == "unhealthy"
        assert body["dependencies"]["vector_db"] == "unhealthy"
        assert body["dependencies"]["bedrock_llm"] == "healthy"

    async def test_llm_unhealthy(self):
        """LLM만 비정상이면 503 + unhealthy."""
        bedrock = _create_bedrock(llm_ok=False)
        supabase = _create_supabase()

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        assert response.status_code == 503
        assert body["dependencies"]["bedrock_llm"] == "unhealthy"

    async def test_embedding_unhealthy(self):
        """임베딩만 비정상이면 503 + unhealthy."""
        bedrock = _create_bedrock(embedding_ok=False)
        supabase = _create_supabase()

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        assert response.status_code == 503
        assert body["dependencies"]["bedrock_embedding"] == "unhealthy"

    async def test_all_unhealthy(self):
        """모든 의존성 비정상이면 503 + 모두 unhealthy."""
        bedrock = _create_bedrock(llm_ok=False, embedding_ok=False)
        supabase = _create_supabase(db_ok=False)

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        assert response.status_code == 503
        assert body["status"] == "unhealthy"
        assert body["dependencies"]["vector_db"] == "unhealthy"
        assert body["dependencies"]["bedrock_llm"] == "unhealthy"
        assert body["dependencies"]["bedrock_embedding"] == "unhealthy"

    async def test_response_has_required_fields(self):
        """응답에 status, dependencies 필드가 존재한다."""
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        response = await health_check(bedrock, supabase)
        body = json.loads(response.body)

        assert "status" in body
        assert "dependencies" in body
        assert "vector_db" in body["dependencies"]
        assert "bedrock_llm" in body["dependencies"]
        assert "bedrock_embedding" in body["dependencies"]
