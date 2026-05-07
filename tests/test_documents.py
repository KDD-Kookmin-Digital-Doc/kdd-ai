"""문서 관리 API 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
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
    }, clear=True):
        return Settings(_env_file=None)


def _create_bedrock(fail_indices: set[int] | None = None) -> AsyncMock:
    """배치 임베딩 모킹.

    fail_indices에 포함된 청크 인덱스가 배치에 **하나라도** 속하면
    해당 배치 호출 전체가 예외로 실패한다 (Option B 시맨틱).
    """
    bedrock = AsyncMock()
    state = {"offset": 0}

    async def _mock_embed(texts, **kwargs):
        offset = state["offset"]
        batch_range = range(offset, offset + len(texts))
        state["offset"] = offset + len(texts)
        if fail_indices and any(i in fail_indices for i in batch_range):
            raise RuntimeError(
                f"Bedrock batch failed (chunks {offset}~{offset + len(texts) - 1})"
            )
        return [[0.1] * 1024 for _ in texts]

    bedrock.embed_texts.side_effect = _mock_embed
    return bedrock


def _create_supabase(inserted_count: int | None = None) -> AsyncMock:
    supabase = AsyncMock()
    supabase.delete_document_chunks.return_value = 0
    supabase.invalidate_cache_by_doc_id.return_value = 0
    if inserted_count is not None:
        supabase.replace_document_chunks.return_value = inserted_count
    else:
        # 기본: 전달된 chunks 수만큼 반환
        supabase.replace_document_chunks.side_effect = (
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
        """배치 실패 시 해당 배치의 모든 청크가 failed_chunks에 기록된다 (Option B)."""
        settings = _create_settings()
        bedrock = _create_bedrock(fail_indices={1})  # 단일 배치에 포함 → 배치 전체 실패
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=3)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["status"] == "partial_failure"
        assert result["embedded_chunk_count"] + len(result["failed_chunks"]) == 3
        assert {f["index"] for f in result["failed_chunks"]} == {0, 1, 2}


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

        call_args = supabase.replace_document_chunks.call_args
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

    async def test_enforcement_date_none_omits_metadata_key(self):
        """enforcement_date=None일 때 metadata에 키 자체가 들어가지 않는다 (옵션 A)."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = EmbedRequest(
            doc_id=1,
            metadata=DocumentMetadata(
                doc_name="test.pdf",
                category="학사",
                enforcement_date=None,
            ),
            chunks=[DocumentChunk(chunk_id=1, content="내용", page=1)],
        )

        await embed_document(request, settings, bedrock, supabase)

        chunks_arg = supabase.replace_document_chunks.call_args[0][1]
        assert "enforcement_date" not in chunks_arg[0]["metadata"]
        assert chunks_arg[0]["metadata"]["doc_name"] == "test.pdf"
        assert chunks_arg[0]["metadata"]["category"] == "학사"
        assert chunks_arg[0]["metadata"]["page"] == 1

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

        chunks_arg = supabase.replace_document_chunks.call_args[0][1]
        assert chunks_arg[0]["chunk_id"] == 77
        assert chunks_arg[0]["content"] == content
        assert chunks_arg[0]["metadata"]["page"] == page


# ── 단위 테스트 ──


class TestEmbedDocumentUnit:
    async def test_replaces_via_single_rpc(self):
        """적재 시 RPC 단일 호출로 처리된다 (delete/invalidate 별도 호출 X — 이슈 #43)."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request()

        await embed_document(request, settings, bedrock, supabase)

        supabase.replace_document_chunks.assert_called_once()
        # 비원자 3단계 호출 경로는 사용되지 않음
        supabase.delete_document_chunks.assert_not_called()
        supabase.invalidate_cache_by_doc_id.assert_not_called()
        supabase.insert_document_chunks.assert_not_called()

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

        chunks_arg = supabase.replace_document_chunks.call_args[0][1]
        assert len(chunks_arg[0]["embedding"]) == 1024

    async def test_all_chunks_fail(self):
        """모든 청크가 실패하면 embedded_chunk_count=0, RPC 미호출 (기존 데이터 보존)."""
        settings = _create_settings()
        bedrock = _create_bedrock(fail_indices={0, 1, 2})
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=3)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["status"] == "partial_failure"
        assert result["embedded_chunk_count"] == 0
        assert len(result["failed_chunks"]) == 3
        supabase.replace_document_chunks.assert_not_called()
        supabase.delete_document_chunks.assert_not_called()
        supabase.invalidate_cache_by_doc_id.assert_not_called()
        supabase.insert_document_chunks.assert_not_called()

    async def test_failed_chunk_has_index_and_error(self):
        """실패한 배치의 각 청크가 index와 error 메시지로 기록된다 (Option B)."""
        settings = _create_settings()
        bedrock = _create_bedrock(fail_indices={2})  # 단일 배치 포함 → 배치 전체 실패
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=3)

        result = await embed_document(request, settings, bedrock, supabase)

        assert len(result["failed_chunks"]) == 3
        assert all(f["error"] == "embedding_failed" for f in result["failed_chunks"])
        assert [f["index"] for f in result["failed_chunks"]] == [0, 1, 2]

    async def test_doc_id_in_response(self):
        """응답에 요청한 doc_id가 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(doc_id=99)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["doc_id"] == 99

    async def test_single_batch_call_for_small_doc(self):
        """청크 수가 BATCH_SIZE 이하면 embed_texts를 1회만 호출한다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        chunk_count = max(1, settings.EMBED_BATCH_SIZE - 1)
        request = _make_embed_request(chunk_count=chunk_count)

        result = await embed_document(request, settings, bedrock, supabase)

        assert bedrock.embed_texts.await_count == 1
        assert result["embedded_chunk_count"] == chunk_count
        assert result["status"] == "success"

    async def test_batches_at_batch_size_boundary(self):
        """정확히 settings.EMBED_BATCH_SIZE 청크는 1회 호출로 끝난다."""
        settings = _create_settings()
        batch_size = settings.EMBED_BATCH_SIZE
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=batch_size)

        await embed_document(request, settings, bedrock, supabase)

        assert bedrock.embed_texts.await_count == 1
        call_args = bedrock.embed_texts.call_args_list[0]
        assert len(call_args.args[0]) == batch_size

    async def test_multiple_batches_all_success(self):
        """settings.EMBED_BATCH_SIZE 초과 시 여러 배치로 나뉘고 모두 성공한다."""
        settings = _create_settings()
        batch_size = settings.EMBED_BATCH_SIZE
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        remainder = 8
        chunk_count = batch_size * 2 + remainder
        request = _make_embed_request(chunk_count=chunk_count)

        result = await embed_document(request, settings, bedrock, supabase)

        assert bedrock.embed_texts.await_count == 3
        batch_sizes = [
            len(c.args[0]) for c in bedrock.embed_texts.call_args_list
        ]
        assert batch_sizes == [batch_size, batch_size, remainder]
        assert result["embedded_chunk_count"] == chunk_count
        assert result["status"] == "success"

    async def test_partial_batch_failure(self):
        """배치 2개 중 1개만 실패 시 실패 배치의 청크만 failed_chunks에 기록된다."""
        settings = _create_settings()
        batch_size = settings.EMBED_BATCH_SIZE
        remainder = 4
        chunk_count = batch_size + remainder
        # 배치 0 (0 ~ batch_size-1) 성공, 배치 1 (batch_size ~ chunk_count-1) 실패
        fail_index = batch_size + 1
        bedrock = _create_bedrock(fail_indices={fail_index})
        supabase = _create_supabase()
        request = _make_embed_request(chunk_count=chunk_count)

        result = await embed_document(request, settings, bedrock, supabase)

        assert result["status"] == "partial_failure"
        assert result["embedded_chunk_count"] == batch_size
        assert len(result["failed_chunks"]) == remainder
        expected_failed_indices = list(
            range(batch_size, batch_size + remainder)
        )
        assert [
            f["index"] for f in result["failed_chunks"]
        ] == expected_failed_indices
        # 성공분은 단일 RPC로 atomic 적재됨
        supabase.replace_document_chunks.assert_called_once()
        rpc_args = supabase.replace_document_chunks.call_args[0]
        assert rpc_args[0] == 1
        assert len(rpc_args[1]) == batch_size

    async def test_re_upload_same_doc_id(self):
        """동일 doc_id 재적재 시 단일 RPC로 atomic하게 처리된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        request = _make_embed_request(doc_id=1, chunk_count=2)

        result = await embed_document(request, settings, bedrock, supabase)

        # RPC 단일 호출 — DELETE+INSERT가 트랜잭션 내부에서 처리
        supabase.replace_document_chunks.assert_called_once()
        assert supabase.replace_document_chunks.call_args[0][0] == 1
        assert result["embedded_chunk_count"] == 2
        assert result["status"] == "success"

    async def test_rpc_failure_does_not_call_legacy_methods(self):
        """RPC 실패 시 비원자 3단계 경로로 폴백하지 않는다 (데이터 손실 방지 — 이슈 #43)."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.replace_document_chunks.side_effect = RuntimeError(
            "Supabase RPC unavailable"
        )
        request = _make_embed_request(chunk_count=2)

        with pytest.raises(RuntimeError, match="Supabase RPC unavailable"):
            await embed_document(request, settings, bedrock, supabase)

        # RPC가 실패해도 레거시 비원자 경로는 호출되지 않음
        supabase.delete_document_chunks.assert_not_called()
        supabase.invalidate_cache_by_doc_id.assert_not_called()
        supabase.insert_document_chunks.assert_not_called()


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
