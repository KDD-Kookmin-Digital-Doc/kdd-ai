"""문서 관리 API 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.schemas import DocumentChunk, DocumentMetadata, EmbedRequest
from app.api.documents import delete_document, embed_document


# ── 헬퍼 ──


def _create_settings() -> Settings:
    with patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
    }):
        return Settings(_env_file=None)


def _create_bedrock(fail_indices: set[int] | None = None) -> AsyncMock:
    """임베딩 모킹. fail_indices에 포함된 인덱스는 예외 발생."""
    bedrock = AsyncMock()
    call_count = {"n": 0}

    async def _mock_embed(texts, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if fail_indices and idx in fail_indices:
            raise RuntimeError(f"Bedrock API timeout (chunk {idx})")
        return [[0.1] * 1024]

    bedrock.embed_texts.side_effect = _mock_embed
    return bedrock


def _create_supabase(inserted_count: int | None = None) -> AsyncMock:
    supabase = AsyncMock()
    supabase.delete_document_chunks.return_value = 0
    supabase.invalidate_cache_by_doc_id.return_value = 0
    if inserted_count is not None:
        supabase.insert_document_chunks.return_value = inserted_count
    else:
        # 기본: 전달된 chunks 수만큼 반환
        supabase.insert_document_chunks.side_effect = (
            lambda doc_id, chunks: len(chunks)
        )
    return supabase


def _make_embed_request(
    doc_id: int = 1,
    chunk_count: int = 3,
    doc_name: str = "학사요람.pdf",
) -> EmbedRequest:
    return EmbedRequest(
        doc_id=doc_id,
        metadata=DocumentMetadata(
            doc_name=doc_name,
            category="학사",
            enforcement_date="2026-03-01",
        ),
        chunks=[
            DocumentChunk(chunk_id=i + 1, content=f"제{i+1}조 내용", page=i + 1)
            for i in range(chunk_count)
        ],
    )


# ── Property 12: 문서 적재 응답 정합성 ──
# Validates: Requirements 9.3


class TestEmbedResponseConsistency:
    """Property 12: embedded_chunk_count가 입력 chunks 길이와 같고 doc_id 일치."""

    @hyp_settings(max_examples=30)
    @given(
        chunk_count=st.integers(min_value=1, max_value=20),
        doc_id=st.integers(min_value=1, max_value=2**31),
    )
    async def test_count_matches_input(self, chunk_count, doc_id):
        """성공 시 embedded_chunk_count == len(chunks)."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(doc_id=doc_id, chunk_count=chunk_count)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["doc_id"] == doc_id
        assert result["embedded_chunk_count"] == chunk_count
        assert result["status"] == "success"

    async def test_partial_failure_count(self):
        """부분 실패 시 embedded_chunk_count + len(failed_chunks) == len(chunks)."""
        settings = _create_settings()
        bedrock = _create_bedrock(fail_indices={1})  # 2번째 청크 실패
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=3)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["status"] == "partial_failure"
        assert result["embedded_chunk_count"] + len(result["failed_chunks"]) == 3
        assert result["failed_chunks"][0]["index"] == 1


# ── Property 13: 문서 적재 라운드트립 ──
# Validates: Requirements 9.2, 9.5


class TestEmbedRoundtrip:
    """Property 13: 적재 시 content, page 메타데이터가 보존된다."""

    async def test_metadata_preserved_in_chunks(self):
        """삽입 요청에 content, page, doc_name, enforcement_date가 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=2, doc_name="학칙.pdf")

        await embed_document(request, settings, bedrock, supabase)

        call_args = supabase.insert_document_chunks.call_args
        doc_id_arg = call_args[0][0]
        chunks_arg = call_args[0][1]

        assert doc_id_arg == 1
        assert len(chunks_arg) == 2
        assert chunks_arg[0]["content"] == "제1조 내용"
        assert chunks_arg[0]["metadata"]["page"] == 1
        assert chunks_arg[0]["metadata"]["doc_name"] == "학칙.pdf"
        assert chunks_arg[0]["metadata"]["enforcement_date"] == "2026-03-01"
        assert chunks_arg[1]["content"] == "제2조 내용"
        assert chunks_arg[1]["metadata"]["page"] == 2

    @hyp_settings(max_examples=20)
    @given(
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        page=st.integers(min_value=1, max_value=1000),
    )
    async def test_content_and_page_always_preserved(self, content, page):
        """임의 content와 page가 삽입 요청에 보존된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        request = EmbedRequest(
            doc_id=999,
            metadata=DocumentMetadata(
                doc_name="test.pdf",
                category="학사",
                enforcement_date="2026-01-01",
            ),
            chunks=[DocumentChunk(chunk_id=77, content=content, page=page)],
        )

        await embed_document(request, settings, bedrock, supabase)

        chunks_arg = supabase.insert_document_chunks.call_args[0][1]
        assert chunks_arg[0]["chunk_id"] == 77
        assert chunks_arg[0]["content"] == content
        assert chunks_arg[0]["metadata"]["page"] == page


# ── 단위 테스트 ──


class TestEmbedDocumentUnit:
    async def test_deletes_existing_before_insert(self):
        """적재 전 기존 청크 삭제 + 캐시 무효화가 수행된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request()

        await embed_document(request, settings, bedrock, supabase)

        supabase.delete_document_chunks.assert_called_once_with(1)
        supabase.invalidate_cache_by_doc_id.assert_called_once_with(1)

    async def test_embedding_uses_search_document_type(self):
        """임베딩 호출 시 input_type이 search_document이다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=1)

        await embed_document(request, settings, bedrock, supabase)

        call_kwargs = bedrock.embed_texts.call_args
        assert call_kwargs.kwargs.get("input_type") == "search_document" or \
            (len(call_kwargs.args) > 1 and call_kwargs.args[1] == "search_document")

    async def test_embedding_includes_vector(self):
        """삽입 청크에 1024차원 embedding이 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=1)

        await embed_document(request, settings, bedrock, supabase)

        chunks_arg = supabase.insert_document_chunks.call_args[0][1]
        assert len(chunks_arg[0]["embedding"]) == 1024

    async def test_all_chunks_fail(self):
        """모든 청크가 실패하면 embedded_chunk_count=0, 삭제/삽입 미호출."""
        settings = _create_settings()
        bedrock = _create_bedrock(fail_indices={0, 1, 2})
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=3)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["status"] == "partial_failure"
        assert result["embedded_chunk_count"] == 0
        assert len(result["failed_chunks"]) == 3
        supabase.delete_document_chunks.assert_not_called()
        supabase.invalidate_cache_by_doc_id.assert_not_called()
        supabase.insert_document_chunks.assert_not_called()

    async def test_failed_chunk_has_index_and_error(self):
        """실패 청크에 index와 error 메시지가 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(fail_indices={2})
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=3)

        result = await embed_document(request, settings, bedrock, supabase)

        failed = result["failed_chunks"][0]
        assert failed["index"] == 2
        assert failed["error"] == "embedding_failed"

    async def test_doc_id_in_response(self):
        """응답에 요청한 doc_id가 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(doc_id=99)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["doc_id"] == 99

    async def test_re_upload_same_doc_id(self):
        """동일 doc_id 재적재 시 삭제 → 적재 순서로 실행된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.delete_document_chunks.return_value = 5  # 기존 5개 삭제

        request = _make_embed_request(doc_id=1, chunk_count=2)

        result = await embed_document(request, settings, bedrock, supabase)

        # 삭제 먼저
        supabase.delete_document_chunks.assert_called_once_with(1)
        supabase.invalidate_cache_by_doc_id.assert_called_once_with(1)
        # 새로 적재
        assert result["embedded_chunk_count"] == 2
        assert result["status"] == "success"


# ── Property 10: 문서 삭제 및 캐시 연쇄 무효화 ──
# Validates: Requirements 10.1, 10.2


class TestDeleteCascade:
    """Property 10: DELETE 후 documents, answer_cache에서 해당 doc_id 관련 레코드 0개."""

    async def test_delete_calls_both_tables(self):
        """삭제 시 documents와 answer_cache 모두에서 삭제가 수행된다."""
        supabase = _create_supabase()
        supabase.delete_document_chunks.return_value = 5
        supabase.invalidate_cache_by_doc_id.return_value = 2

        result = await delete_document(1, supabase)

        supabase.delete_document_chunks.assert_called_once_with(1)
        supabase.invalidate_cache_by_doc_id.assert_called_once_with(1)
        assert result["deleted_chunk_count"] == 5
        assert result["invalidated_cache_count"] == 2

    async def test_response_includes_all_fields(self):
        """응답에 status, doc_id, deleted_chunk_count, invalidated_cache_count, message 포함."""
        supabase = _create_supabase()
        supabase.delete_document_chunks.return_value = 3
        supabase.invalidate_cache_by_doc_id.return_value = 1

        result = await delete_document(99, supabase)

        assert result["status"] == "success"
        assert result["doc_id"] == 99
        assert result["deleted_chunk_count"] == 3
        assert result["invalidated_cache_count"] == 1
        assert "message" in result

    @hyp_settings(max_examples=30)
    @given(
        doc_id=st.integers(min_value=1, max_value=2**31),
        chunk_count=st.integers(min_value=0, max_value=100),
        cache_count=st.integers(min_value=0, max_value=50),
    )
    async def test_counts_match_db_response(self, doc_id, chunk_count, cache_count):
        """응답 카운트가 DB 삭제 결과와 일치한다."""
        supabase = _create_supabase()
        supabase.delete_document_chunks.return_value = chunk_count
        supabase.invalidate_cache_by_doc_id.return_value = cache_count

        result = await delete_document(doc_id, supabase)

        assert result["deleted_chunk_count"] == chunk_count
        assert result["invalidated_cache_count"] == cache_count
        assert result["status"] == "success"


# ── Property 11: 문서 삭제 멱등성 ──
# Validates: Requirements 10.4


class TestDeleteIdempotency:
    """Property 11: 여러 번 DELETE 수행 시 항상 성공, 존재하지 않는 doc_id는 카운트 0."""

    async def test_nonexistent_doc_returns_zero_counts(self):
        """존재하지 않는 doc_id도 카운트 0으로 성공 응답."""
        supabase = _create_supabase()
        supabase.delete_document_chunks.return_value = 0
        supabase.invalidate_cache_by_doc_id.return_value = 0

        result = await delete_document(12345, supabase)

        assert result["status"] == "success"
        assert result["deleted_chunk_count"] == 0
        assert result["invalidated_cache_count"] == 0

    async def test_double_delete_always_succeeds(self):
        """같은 doc_id를 2번 삭제해도 항상 성공."""
        supabase = _create_supabase()
        supabase.delete_document_chunks.side_effect = [5, 0]
        supabase.invalidate_cache_by_doc_id.side_effect = [2, 0]

        result1 = await delete_document(1, supabase)
        result2 = await delete_document(1, supabase)

        assert result1["status"] == "success"
        assert result1["deleted_chunk_count"] == 5
        assert result2["status"] == "success"
        assert result2["deleted_chunk_count"] == 0

    @hyp_settings(max_examples=20)
    @given(
        repeat=st.integers(min_value=1, max_value=5),
    )
    async def test_multiple_deletes_always_success(self, repeat):
        """N번 삭제해도 항상 status=success."""
        supabase = _create_supabase()
        supabase.delete_document_chunks.return_value = 0
        supabase.invalidate_cache_by_doc_id.return_value = 0

        for _ in range(repeat):
            result = await delete_document(1, supabase)
            assert result["status"] == "success"
