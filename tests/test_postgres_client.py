"""PostgreSQL Vector DB 클라이언트 단위 테스트. asyncpg pool 기반 모킹.

D3 (2026-05-13): supabase-py mock 패턴 → asyncpg.create_pool + Pool.fetch/fetchrow/
fetchval/execute 직접 mock 으로 재작성. 호출자 인터페이스 (메서드명 + 반환 타입)
가 동일해서 다른 테스트 파일의 _create_postgres 헬퍼는 AsyncMock 으로 그대로
호환 (test_chat.py / test_documents.py 등 — Task #8 에서 헬퍼명만 리네임).
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings
from app.models.pipeline import AnswerCache, CacheMatch, SearchResult


def _create_settings(**overrides: str) -> Settings:
    """PR #62 헬퍼 패턴 — clear=True 로 셸/CI env leak 차단."""
    env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        **overrides,
    }
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)


@pytest.fixture
def settings():
    return _create_settings()


@pytest.fixture
def postgres_setup(settings):
    """asyncpg.create_pool 을 AsyncMock 으로 패치.

    PostgresVectorClient 는 lazy init — 첫 fetch/execute 호출 시점에 create_pool
    이 await 되며 mock_pool 이 self._pool 에 set. 이후 모든 호출은 mock_pool
    의 메서드를 통과한다.

    yield 3-tuple: ``(client, mock_pool, mock_create_pool)``. 대부분 테스트는
    앞 둘만 사용 (`client, mock_pool, _ = postgres_setup`). create_pool 호출
    횟수 등 lifecycle 검증이 필요한 테스트 (예: ``test_pool_init_only_once``)
    는 세 번째도 unpack 해서 ``await_count`` 단언.
    """
    with patch(
        "app.clients.postgres_client.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create_pool:
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        client = PostgresVectorClient(settings)
        yield client, mock_pool, mock_create_pool


# ── search_documents 테스트 ──


class TestSearchDocuments:
    async def test_success(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = [
            {
                "chunk_id": 1001,
                "doc_id": 1,
                "content": "제1조 내용",
                "metadata": {"doc_name": "학사요람.pdf", "page": 10},
                "similarity": 0.89,
            },
            {
                "chunk_id": 1002,
                "doc_id": 1,
                "content": "제2조 내용",
                "metadata": {"doc_name": "학사요람.pdf", "page": 11},
                "similarity": 0.82,
            },
        ]

        results = await client.search_documents([0.1] * 1024)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == 1001
        assert results[0].doc_id == 1
        assert results[0].similarity_score == 0.89
        assert results[1].chunk_id == 1002
        assert results[1].metadata["page"] == 11

    async def test_empty_results(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        results = await client.search_documents([0.1] * 1024)

        assert results == []

    async def test_custom_params(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        await client.search_documents([0.1] * 1024, top_k=10, threshold=0.8)

        mock_pool.fetch.assert_called_once_with(
            "SELECT chunk_id, doc_id, content, metadata, similarity "
            "FROM match_documents($1, $2, $3)",
            [0.1] * 1024,
            0.8,
            10,
        )


# ── search_answer_cache 테스트 ──


class TestSearchAnswerCache:
    async def test_cache_hit(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchrow.return_value = {
            "question": "휴학 기간",
            "answer": "최대 4년입니다.",
            "similarity": 0.97,
            "sources": [{"doc_name": "학사요람.pdf", "page": 45}],
        }

        result = await client.search_answer_cache([0.1] * 1024)

        assert isinstance(result, CacheMatch)
        assert result.question == "휴학 기간"
        assert result.answer == "최대 4년입니다."
        assert result.similarity_score == 0.97
        assert len(result.sources) == 1

    async def test_cache_miss(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchrow.return_value = None

        result = await client.search_answer_cache([0.1] * 1024)

        assert result is None

    async def test_default_ttl_from_settings(self, postgres_setup):
        """ttl_days 미지정 시 settings.CACHE_TTL_DAYS(기본 90) 사용."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchrow.return_value = None

        await client.search_answer_cache([0.1] * 1024)

        mock_pool.fetchrow.assert_called_once_with(
            "SELECT question, answer, similarity, sources "
            "FROM match_answer_cache($1, $2, $3)",
            [0.1] * 1024,
            0.95,
            90,
        )

    async def test_custom_threshold_and_ttl(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchrow.return_value = None

        await client.search_answer_cache(
            [0.1] * 1024, threshold=0.9, ttl_days=30
        )

        mock_pool.fetchrow.assert_called_once_with(
            "SELECT question, answer, similarity, sources "
            "FROM match_answer_cache($1, $2, $3)",
            [0.1] * 1024,
            0.9,
            30,
        )

    async def test_sources_none_returns_empty_list(self, postgres_setup):
        """sources 값이 None 이면 빈 리스트로 반환."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchrow.return_value = {
            "question": "질문",
            "answer": "답변",
            "similarity": 0.96,
            "sources": None,
        }

        result = await client.search_answer_cache([0.1] * 1024)

        assert result is not None
        assert result.sources == []


# ── search_similar_questions 테스트 ──


class TestSearchSimilarQuestions:
    async def test_success(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = [
            {"question": "휴학 신청 방법"},
            {"question": "복학 절차"},
            {"question": "졸업 요건"},
        ]

        result = await client.search_similar_questions([0.1] * 1024)

        assert result == ["휴학 신청 방법", "복학 절차", "졸업 요건"]

    async def test_empty_results(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        result = await client.search_similar_questions([0.1] * 1024)

        assert result == []

    async def test_threshold_passed_to_rpc(self, postgres_setup):
        """이슈 #46: match_threshold 인자가 RPC 에 그대로 전달된다."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        await client.search_similar_questions(
            [0.1] * 1024, top_k=5, threshold=0.5
        )

        mock_pool.fetch.assert_called_once_with(
            "SELECT question FROM match_similar_questions($1, $2, $3)",
            [0.1] * 1024,
            5,
            0.5,
        )

    async def test_default_threshold_is_zero(self, postgres_setup):
        """후방 호환성: threshold 미지정 시 0.0 전달."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        await client.search_similar_questions([0.1] * 1024)

        # 호출 인자: (sql, embedding, top_k, threshold)
        call_args = mock_pool.fetch.call_args.args
        assert call_args[3] == 0.0


# ── replace_document_chunks 테스트 ──


class TestReplaceDocumentChunks:
    async def test_calls_rpc_with_doc_id_and_chunks(self, postgres_setup):
        """단일 RPC 호출로 doc_id 와 chunks 페이로드를 전달한다."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchval.return_value = 2

        chunks = [
            {
                "chunk_id": 1,
                "content": "내용1",
                "embedding": [0.1] * 1024,
                "metadata": {"page": 1},
            },
            {
                "chunk_id": 2,
                "content": "내용2",
                "embedding": [0.2] * 1024,
                "metadata": {"page": 2},
            },
        ]

        count = await client.replace_document_chunks(99, chunks)

        assert count == 2
        mock_pool.fetchval.assert_called_once_with(
            "SELECT replace_document_chunks($1, $2)", 99, chunks
        )

    async def test_empty_chunks_skips_rpc(self, postgres_setup):
        """빈 chunks 리스트는 RPC 호출 없이 0 반환 (기존 데이터 보존)."""
        client, mock_pool, _ = postgres_setup

        count = await client.replace_document_chunks(99, [])

        assert count == 0
        mock_pool.fetchval.assert_not_called()

    async def test_returns_zero_for_none(self, postgres_setup):
        """RPC 가 None 반환 시 0 (방어)."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchval.return_value = None

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]
        count = await client.replace_document_chunks(1, chunks)

        assert count == 0

    async def test_rpc_failure_propagates(self, postgres_setup):
        """RPC 실패는 호출자로 전파된다 (트랜잭션 롤백은 DB 측 책임)."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchval.side_effect = Exception("RPC failed")

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]

        with pytest.raises(Exception, match="RPC failed"):
            await client.replace_document_chunks(1, chunks)


# ── delete_document_chunks 테스트 ──


class TestDeleteDocumentChunks:
    async def test_success(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = [
            {"chunk_id": 1},
            {"chunk_id": 2},
            {"chunk_id": 3},
        ]

        count = await client.delete_document_chunks(1)

        assert count == 3
        mock_pool.fetch.assert_called_once_with(
            "DELETE FROM documents WHERE doc_id = $1 RETURNING chunk_id", 1
        )

    async def test_no_matching_docs(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        count = await client.delete_document_chunks(12345)

        assert count == 0


# ── invalidate_cache_by_doc_id 테스트 ──


class TestInvalidateCacheByDocId:
    async def test_success(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = [{"id": 10}, {"id": 11}]

        count = await client.invalidate_cache_by_doc_id(1)

        assert count == 2

    async def test_no_matching_cache(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        count = await client.invalidate_cache_by_doc_id(1)

        assert count == 0

    async def test_doc_id_passed_as_native_bigint(self, postgres_setup):
        """asyncpg 는 BIGINT[] native 처리 — supabase-py stringify 우회 불필요."""
        client, mock_pool, _ = postgres_setup
        mock_pool.fetch.return_value = []

        await client.invalidate_cache_by_doc_id(20240001)

        mock_pool.fetch.assert_called_once_with(
            "DELETE FROM answer_cache WHERE $1 = ANY(source_doc_ids) RETURNING id",
            20240001,
        )


# ── upsert_answer_cache 테스트 (이슈 #49) ──


class TestUpsertAnswerCache:
    async def test_success(self, postgres_setup):
        """upsert_answer_cache RPC 호출 시 모든 필드가 정확히 전달된다."""
        client, mock_pool, _ = postgres_setup
        mock_pool.execute.return_value = None

        cache = AnswerCache(
            question="휴학 기간",
            embedding=[0.1] * 1024,
            answer="최대 4년입니다.",
            source_doc_ids=[1],
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )

        await client.upsert_answer_cache(cache)

        mock_pool.execute.assert_called_once_with(
            "SELECT upsert_answer_cache($1, $2, $3, $4, $5)",
            "휴학 기간",
            [0.1] * 1024,
            "최대 4년입니다.",
            [1],
            [{"doc_name": "학사요람.pdf", "page": 45}],
        )

    async def test_does_not_use_raw_table_insert(self, postgres_setup):
        """이슈 #49 잠금: upsert 는 RPC 경로만 사용. raw INSERT/UPDATE 직접 호출 X."""
        client, mock_pool, _ = postgres_setup
        mock_pool.execute.return_value = None

        cache = AnswerCache(
            question="q",
            embedding=[0.1] * 1024,
            answer="a",
            source_doc_ids=[],
            sources=[],
        )
        await client.upsert_answer_cache(cache)

        # execute 가 RPC SELECT 한 번만 호출됐고, fetch/fetchval/fetchrow 미사용
        assert mock_pool.execute.call_count == 1
        assert mock_pool.execute.call_args.args[0].startswith(
            "SELECT upsert_answer_cache"
        )
        mock_pool.fetch.assert_not_called()
        mock_pool.fetchval.assert_not_called()
        mock_pool.fetchrow.assert_not_called()


# ── health_check 테스트 ──


class TestHealthCheck:
    async def test_healthy(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchval.return_value = 1

        assert await client.health_check() is True
        mock_pool.fetchval.assert_called_once_with(
            "SELECT 1 FROM documents LIMIT 1"
        )

    async def test_unhealthy(self, postgres_setup):
        client, mock_pool, _ = postgres_setup
        mock_pool.fetchval.side_effect = Exception("Connection refused")

        assert await client.health_check() is False


# ── close + pool lifecycle 테스트 ──


class TestPoolLifecycle:
    async def test_close_drains_pool(self, postgres_setup):
        """close() 가 pool.close() 를 호출하고 self._pool 을 None 으로 리셋한다."""
        client, mock_pool, _ = postgres_setup
        # _init_pool 발동 — pool 이 set 됨
        mock_pool.fetchval.return_value = 1
        await client.health_check()
        assert client._pool is mock_pool

        await client.close()

        mock_pool.close.assert_awaited_once()
        assert client._pool is None

    async def test_close_no_op_when_pool_not_initialized(self, postgres_setup):
        """pool 이 lazy init 되지 않은 상태에서 close() 호출해도 안전 (no-op)."""
        client, mock_pool, _ = postgres_setup
        # health_check 등 어떤 메서드도 호출 안 함 — pool 이 None

        await client.close()

        # 한 번도 acquire 안 했으니 close 호출 없음
        mock_pool.close.assert_not_awaited()

    async def test_pool_init_only_once(self, postgres_setup):
        """여러 메서드 호출 시 ``asyncpg.create_pool`` 이 최초 1회만 await 된다.

        CodeRabbit CR2 — 기존엔 ``client._pool is mock_pool`` 간접 검증만 했는데
        실제 ``await_count`` 직접 단언으로 잠금 강화. ``asyncio.Lock`` 가드(CR1)
        와 짝을 이루는 회귀 잠금.
        """
        client, mock_pool, mock_create_pool = postgres_setup
        mock_pool.fetchval.return_value = 1
        mock_pool.fetch.return_value = []

        await client.health_check()
        await client.search_documents([0.1] * 1024)
        await client.delete_document_chunks(1)

        # asyncpg.create_pool 이 정확히 1회만 await — race 가드(asyncio.Lock +
        # double-check) 가 두 번째 호출을 차단하는지 명시 검증
        assert mock_create_pool.await_count == 1
        assert client._pool is mock_pool
