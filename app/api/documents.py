"""문서 관리 API 엔드포인트. 벡터화 적재 및 삭제."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.dependencies import get_bedrock_client, get_supabase_client
from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings, get_settings
from app.models.schemas import EmbedRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/documents/embed")
async def embed_document(
    request: EmbedRequest,
    settings: Settings = Depends(get_settings),
    bedrock: BedrockClient = Depends(get_bedrock_client),
    supabase: SupabaseVectorClient = Depends(get_supabase_client),
) -> dict:
    """문서 청크를 벡터화하여 DB에 적재한다.

    동일 doc_id 재적재 시 기존 청크 삭제 + 캐시 무효화 후 새 청크 적재 (last-write-wins).
    """
    # 1. 청크별 임베딩 생성 (기존 데이터 삭제 전에 먼저 준비)
    embedded_chunks: list[dict] = []
    failed_chunks: list[dict] = []

    for i, chunk in enumerate(request.chunks):
        try:
            embeddings = await bedrock.embed_texts(
                [chunk.content], input_type="search_document"
            )
            embedded_chunks.append({
                "content": chunk.content,
                "embedding": embeddings[0],
                "metadata": {
                    "doc_name": request.metadata.doc_name,
                    "page": chunk.page,
                    "category": request.metadata.category,
                    "enforcement_date": str(request.metadata.enforcement_date),
                },
            })
        except Exception as exc:
            logger.exception("청크 %d 임베딩 실패", i)
            failed_chunks.append({"index": i, "error": "embedding_failed"})

    # 2. 성공분이 있을 때만 기존 삭제 + 새 청크 삽입
    inserted_count = 0
    if embedded_chunks:
        await supabase.delete_document_chunks(request.doc_id)
        await supabase.invalidate_cache_by_doc_id(request.doc_id)
        inserted_count = await supabase.insert_document_chunks(
            request.doc_id, embedded_chunks
        )

    # 4. 응답
    if failed_chunks:
        logger.warning(
            "문서 적재 부분 실패: doc_id=%s, 성공=%d, 실패=%d",
            request.doc_id,
            inserted_count,
            len(failed_chunks),
        )
        return {
            "status": "partial_failure",
            "doc_id": request.doc_id,
            "embedded_chunk_count": inserted_count,
            "failed_chunks": failed_chunks,
            "message": "일부 청크의 벡터화에 실패했습니다.",
        }

    logger.info(
        "문서 적재 완료: doc_id=%s, chunk_count=%d",
        request.doc_id,
        inserted_count,
    )
    return {
        "status": "success",
        "doc_id": request.doc_id,
        "embedded_chunk_count": inserted_count,
    }


@router.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    supabase: SupabaseVectorClient = Depends(get_supabase_client),
) -> dict:
    """문서 벡터 데이터를 삭제하고 관련 캐시를 무효화한다.

    존재하지 않는 doc_id에 대해서도 카운트 0으로 성공 응답을 반환한다 (멱등성).
    """
    deleted_chunk_count = await supabase.delete_document_chunks(doc_id)
    invalidated_cache_count = await supabase.invalidate_cache_by_doc_id(doc_id)

    logger.info(
        "문서 삭제 완료: doc_id=%s, deleted_chunks=%d, invalidated_caches=%d",
        doc_id,
        deleted_chunk_count,
        invalidated_cache_count,
    )
    return {
        "status": "success",
        "doc_id": doc_id,
        "deleted_chunk_count": deleted_chunk_count,
        "invalidated_cache_count": invalidated_cache_count,
        "message": "문서 삭제가 완료되었습니다.",
    }
