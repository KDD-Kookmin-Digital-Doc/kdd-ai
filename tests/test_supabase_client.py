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

    async def test_threshold_passed_to_rpc(self, supabase_setup):
        """이슈 #46: match_threshold 인자가 RPC 에 그대로 전달된다."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        await client.search_similar_questions(
            [0.1] * 1024, top_k=5, threshold=0.5
        )

        mock_sb.rpc.assert_called_once_with(
            "match_similar_questions",
            {
                "query_embedding": [0.1] * 1024,
                "match_count": 5,
                "match_threshold": 0.5,
            },
        )

    async def test_default_threshold_is_zero(self, supabase_setup):
        """후방 호환성: threshold 미지정 시 0.0 전달."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        await client.search_similar_questions([0.1] * 1024)

        call_kwargs = mock_sb.rpc.call_args[0][1]
        assert call_kwargs["match_threshold"] == 0.0


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


# ── replace_document_chunks 테스트 ──


class TestReplaceDocumentChunks:
    async def test_calls_rpc_with_doc_id_and_chunks(self, supabase_setup):
        """단일 RPC 호출로 doc_id와 chunks 페이로드를 전달한다."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=2)

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
        mock_sb.rpc.assert_called_once_with(
            "replace_document_chunks",
            {"p_doc_id": 99, "p_chunks": chunks},
        )

    async def test_empty_chunks_skips_rpc(self, supabase_setup):
        """빈 chunks 리스트는 RPC 호출 없이 0 반환 (기존 데이터 보존)."""
        client, mock_sb = supabase_setup

        count = await client.replace_document_chunks(99, [])

        assert count == 0
        mock_sb.rpc.assert_not_called()

    async def test_returns_count_from_list_of_dict(self, supabase_setup):
        """RPC가 [{"replace_document_chunks": N}] 형태로 반환할 때 명시 키로 N을 추출한다."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data=[{"replace_document_chunks": 5}]
        )

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]
        count = await client.replace_document_chunks(1, chunks)

        assert count == 5

    async def test_returns_count_from_list_of_int(self, supabase_setup):
        """RPC가 [N] (단일 정수 리스트) 형태로 반환해도 N을 추출한다."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[7])

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]
        count = await client.replace_document_chunks(1, chunks)

        assert count == 7

    async def test_explicit_key_takes_precedence_over_dict_order(self, supabase_setup):
        """dict에 다른 키가 섞여 있어도 'replace_document_chunks' 키가 우선 추출된다."""
        client, mock_sb = supabase_setup
        # 메타 키가 먼저 와도 명시 키 우선 — PostgREST 직렬화 순서 변동에 견고
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data=[{"_meta": "ignored", "replace_document_chunks": 9}]
        )

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]
        count = await client.replace_document_chunks(1, chunks)

        assert count == 9

    async def test_unknown_response_shape_warns_and_returns_zero(
        self, supabase_setup, caplog
    ):
        """알 수 없는 응답 형태는 silent 0가 아니라 경고 로그 + 0을 반환한다."""
        import logging

        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(
            data={"unexpected": "shape"}
        )

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]

        with caplog.at_level(logging.WARNING, logger="app.clients.supabase_client"):
            count = await client.replace_document_chunks(1, chunks)

        assert count == 0
        assert any(
            "응답 형태가 예상과 다릅니다" in rec.message for rec in caplog.records
        )

    async def test_returns_zero_for_empty_list(self, supabase_setup):
        """빈 리스트 응답은 0을 반환한다 (RPC가 0 row 반환한 정상 경로)."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]
        count = await client.replace_document_chunks(1, chunks)

        assert count == 0

    async def test_returns_zero_for_none_data(self, supabase_setup):
        """RPC가 None 또는 빈 데이터를 반환하면 0."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=None)

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]
        count = await client.replace_document_chunks(1, chunks)

        assert count == 0

    async def test_rpc_failure_propagates(self, supabase_setup):
        """RPC 실패는 호출자로 전파된다 (트랜잭션 롤백은 DB 측 책임)."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.side_effect = Exception("RPC failed")

        chunks = [{"chunk_id": 1, "content": "x", "embedding": [0.1] * 1024, "metadata": {}}]

        with pytest.raises(Exception, match="RPC failed"):
            await client.replace_document_chunks(1, chunks)


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

    async def test_contains_argument_is_stringified(self, supabase_setup):
        """postgrest-py의 .contains()는 array 원소를 ",".join()하므로 str 필수.
        BIGINT[] 컬럼이라도 호출 측에서 stringify해야 TypeError 방지됨.
        """
        client, mock_sb = supabase_setup
        contains_mock = mock_sb.table.return_value.delete.return_value.contains
        contains_mock.return_value.execute.return_value = MagicMock(data=[])

        await client.invalidate_cache_by_doc_id(20240001)

        contains_mock.assert_called_once_with("source_doc_ids", ["20240001"])


# ── upsert_answer_cache 테스트 (이슈 #49) ──


class TestUpsertAnswerCache:
    async def test_success(self, supabase_setup):
        """upsert_answer_cache RPC 호출 시 모든 필드가 정확히 전달된다."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=None)

        cache = AnswerCache(
            question="휴학 기간",
            embedding=[0.1] * 1024,
            answer="최대 4년입니다.",
            source_doc_ids=[1],
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )

        await client.upsert_answer_cache(cache)

        mock_sb.rpc.assert_called_once_with(
            "upsert_answer_cache",
            {
                "p_question": "휴학 기간",
                "p_embedding": [0.1] * 1024,
                "p_answer": "최대 4년입니다.",
                "p_source_doc_ids": [1],
                "p_sources": [{"doc_name": "학사요람.pdf", "page": 45}],
            },
        )

    async def test_does_not_use_table_insert(self, supabase_setup):
        """이슈 #49: 더 이상 .table().insert() 직접 호출 X — RPC 만 사용."""
        client, mock_sb = supabase_setup
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=None)

        cache = AnswerCache(
            question="q", embedding=[0.1] * 1024, answer="a",
            source_doc_ids=[], sources=[],
        )
        await client.upsert_answer_cache(cache)

        # .table("answer_cache").insert(...) 패턴이 호출되지 않음
        # (호출자는 mock_sb.rpc 만 사용)
        # mock_sb.table 은 다른 메서드(invalidate 등)에서도 쓰일 수 있어 호출 자체는 막지 않음.
        # 핵심은 답변 캐시 저장이 RPC 로 처리됨.
        mock_sb.rpc.assert_called_once()
        assert mock_sb.rpc.call_args[0][0] == "upsert_answer_cache"


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
