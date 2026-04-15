"""Supabase Vector DB 클라이언트 단위 테스트. 모킹 기반."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings
from app.models.pipeline import AnswerCache, CacheMatch, SearchResult


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


@pytest.fixture
def supabase_setup(settings):
    with patch("app.clients.supabase_client.create_client") as mock_create:
        mock_sb = MagicMock()
        mock_create.return_value = mock_sb
        client = SupabaseVectorClient(settings)
    return client, mock_sb


# ── search_documents 테스트 ──


class TestSearchDocuments:
    async def test_success(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data=[
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
        )

        results = await client.search_documents([0.1] * 1024)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == 1001
        assert results[0].doc_id == 1
        assert results[0].similarity_score == 0.89
        assert results[1].chunk_id == 1002
        assert results[1].metadata["page"] == 11

    async def test_empty_results(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        results = await client.search_documents([0.1] * 1024)

        assert results == []

    async def test_custom_params(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        await client.search_documents(
            [0.1] * 1024, top_k=10, threshold=0.8
        )

        mock_sb.rpc.assert_called_once_with(
            "match_documents",
            {
                "query_embedding": [0.1] * 1024,
                "match_threshold": 0.8,
                "match_count": 10,
            },
        )


# ── search_answer_cache 테스트 ──


class TestSearchAnswerCache:
    async def test_cache_hit(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "question": "휴학 기간",
                    "answer": "최대 4년입니다.",
                    "similarity": 0.97,
                    "sources": [{"doc_name": "학사요람.pdf", "page": 45}],
                }
            ]
        )

        result = await client.search_answer_cache([0.1] * 1024)

        assert isinstance(result, CacheMatch)
        assert result.question == "휴학 기간"
        assert result.answer == "최대 4년입니다."
        assert result.similarity_score == 0.97
        assert len(result.sources) == 1

    async def test_cache_miss(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        result = await client.search_answer_cache([0.1] * 1024)

        assert result is None

    async def test_default_ttl_from_settings(self, supabase_setup):
        """ttl_days 미지정 시 settings.CACHE_TTL_DAYS(기본 90) 사용."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        await client.search_answer_cache([0.1] * 1024)

        mock_sb.rpc.assert_called_once_with(
            "match_answer_cache",
            {
                "query_embedding": [0.1] * 1024,
                "match_threshold": 0.95,
                "ttl_days": 90,  # settings 기본값
            },
        )

    async def test_custom_threshold_and_ttl(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        await client.search_answer_cache(
            [0.1] * 1024, threshold=0.9, ttl_days=30
        )

        mock_sb.rpc.assert_called_once_with(
            "match_answer_cache",
            {
                "query_embedding": [0.1] * 1024,
                "match_threshold": 0.9,
                "ttl_days": 30,
            },
        )

    async def test_sources_none_returns_empty_list(self, supabase_setup):
        """sources 값이 None이면 빈 리스트로 반환."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "question": "질문",
                    "answer": "답변",
                    "similarity": 0.96,
                    "sources": None,
                }
            ]
        )

        result = await client.search_answer_cache([0.1] * 1024)

        assert result is not None
        assert result.sources == []


# ── search_similar_questions 테스트 ──


class TestSearchSimilarQuestions:
    async def test_success(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {"question": "휴학 신청 방법"},
                {"question": "복학 절차"},
                {"question": "졸업 요건"},
            ]
        )

        result = await client.search_similar_questions([0.1] * 1024)

        assert result == ["휴학 신청 방법", "복학 절차", "졸업 요건"]

    async def test_empty_results(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        result = await client.search_similar_questions([0.1] * 1024)

        assert result == []


# ── insert_document_chunks 테스트 ──


class TestInsertDocumentChunks:
    async def test_success(self, supabase_setup):
        client, mock_sb = supabase_setup
        chunks = [
            {"content": "내용1", "embedding": [0.1] * 1024},
            {"content": "내용2", "embedding": [0.2] * 1024},
        ]
        mock_sb.table.return_value.insert.return_value.execute.return_value = (
            MagicMock(data=[{"id": 1}, {"id": 2}])
        )

        count = await client.insert_document_chunks(1, chunks)

        assert count == 2

    async def test_empty_chunks_returns_zero(self, supabase_setup):
        """빈 chunks 리스트는 DB 호출 없이 0 반환."""
        client, mock_sb = supabase_setup

        count = await client.insert_document_chunks(1, [])

        assert count == 0
        mock_sb.table.return_value.insert.assert_not_called()

    async def test_doc_id_injected_into_chunks(self, supabase_setup):
        """chunks에 doc_id가 없어도 메서드가 자동으로 주입."""
        client, mock_sb = supabase_setup
        chunks = [{"content": "내용", "embedding": [0.1] * 1024}]
        mock_sb.table.return_value.insert.return_value.execute.return_value = (
            MagicMock(data=[{"id": 1}])
        )

        await client.insert_document_chunks(99, chunks)

        inserted_payload = mock_sb.table.return_value.insert.call_args[0][0]
        assert all(c["doc_id"] == 99 for c in inserted_payload)


# ── delete_document_chunks 테스트 ──


class TestDeleteDocumentChunks:
    async def test_success(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}, {"id": 2}, {"id": 3}]
        )

        count = await client.delete_document_chunks(1)

        assert count == 3

    async def test_no_matching_docs(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[]
        )

        count = await client.delete_document_chunks(12345)

        assert count == 0


# ── invalidate_cache_by_doc_id 테스트 ──


class TestInvalidateCacheByDocId:
    async def test_success(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.delete.return_value.contains.return_value.execute.return_value = MagicMock(
            data=[{"id": 10}, {"id": 11}]
        )

        count = await client.invalidate_cache_by_doc_id(1)

        assert count == 2

    async def test_no_matching_cache(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.delete.return_value.contains.return_value.execute.return_value = MagicMock(
            data=[]
        )

        count = await client.invalidate_cache_by_doc_id(1)

        assert count == 0


# ── insert_answer_cache 테스트 ──


class TestInsertAnswerCache:
    async def test_success(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.insert.return_value.execute.return_value = (
            MagicMock(data=[{"id": 1}])
        )

        cache = AnswerCache(
            question="휴학 기간",
            embedding=[0.1] * 1024,
            answer="최대 4년입니다.",
            source_doc_ids=[1],
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )

        await client.insert_answer_cache(cache)

        mock_sb.table.assert_called_with("answer_cache")


# ── health_check 테스트 ──


class TestHealthCheck:
    async def test_healthy(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        assert await client.health_check() is True

    async def test_unhealthy(self, supabase_setup):
        client, mock_sb = supabase_setup
        mock_sb.table.return_value.select.return_value.limit.return_value.execute.side_effect = (
            Exception("Connection refused")
        )

        assert await client.health_check() is False

    async def test_timeout_applied(self, supabase_setup):
        """타임아웃 초과 시 _run_with_timeout이 TimeoutError를 발생시킨다."""
        client, mock_sb = supabase_setup
        client._timeout = 0.01  # 10ms

        def _slow_fn():
            import time
            time.sleep(1)
            return MagicMock(data=[])

        mock_sb.rpc.return_value.execute = _slow_fn

        with pytest.raises(asyncio.TimeoutError):
            await client.search_documents([0.1] * 1024)
