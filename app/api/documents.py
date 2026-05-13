"""문서 관리 API 엔드포인트. 벡터화 적재 및 삭제."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Path

from app.api.dependencies import get_bedrock_client, get_postgres_client
from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings, get_settings
from app.models.schemas import EmbedRequest, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/documents/embed",
    tags=["Documents"],
    summary="문서 청크 벡터화 및 적재",
    operation_id="embed_document",
    description=(
        "문서 청크 목록을 받아 임베딩을 생성하고 벡터 DB에 적재합니다.\n\n"
        "- 동일 `doc_id` 재적재 시 **단일 트랜잭션**(Postgres RPC)으로 "
        "기존 청크 삭제 + 관련 캐시 무효화 + 새 청크 삽입 (last-write-wins, atomic)\n"
        "- 일부 청크 임베딩 실패 시 `status=partial_failure` 로 응답합니다. "
        "이때도 RPC는 호출되며, 동일 `doc_id`의 기존 청크 전체가 새 성공분으로 "
        "교체됩니다 (atomic). 즉 `partial_failure`라도 기존 데이터는 그대로가 아닙니다.\n"
        "- 모든 청크 임베딩 실패 시 RPC 미호출 → 기존 데이터 그대로 보존.\n"
        "- DB 적재 단계 실패 시 트랜잭션 롤백으로 기존 데이터가 보존됩니다."
    ),
    responses={
        200: {
            "description": "적재 성공 또는 부분 실패",
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "전체 성공",
                            "value": {
                                "status": "success",
                                "doc_id": 20240001,
                                "embedded_chunk_count": 12,
                                "message": "문서 벡터화 및 적재가 완료되었습니다.",
                            },
                        },
                        "partial_failure": {
                            "summary": "부분 실패",
                            "value": {
                                "status": "partial_failure",
                                "doc_id": 20240001,
                                "embedded_chunk_count": 10,
                                "failed_chunks": [{"index": 5, "error": "embedding_failed"}],
                                "message": "일부 청크의 벡터화에 실패했습니다.",
                            },
                        },
                    }
                }
            },
        },
        400: {"model": ErrorResponse, "description": "필수 파라미터 누락"},
        422: {"model": ErrorResponse, "description": "타입 불일치 / 제약조건 위반"},
        503: {"model": ErrorResponse, "description": "외부 서비스 장애"},
    },
)
async def embed_document(
    request: EmbedRequest,
    settings: Settings = Depends(get_settings),
    bedrock: BedrockClient = Depends(get_bedrock_client),
    postgres: PostgresVectorClient = Depends(get_postgres_client),
) -> dict:
    """문서 청크를 벡터화하여 DB에 적재한다.

    동일 doc_id 재적재 시 RPC 단일 트랜잭션으로 기존 청크 삭제 + 캐시 무효화 + 새 청크 적재
    (last-write-wins, atomic — 이슈 #43).
    """
    # 1. 청크 배치 임베딩 (기존 데이터 삭제 전에 먼저 준비)
    embedded_chunks: list[dict] = []
    failed_chunks: list[dict] = []

    batch_size = settings.EMBED_BATCH_SIZE
    for batch_start in range(0, len(request.chunks), batch_size):
        batch = request.chunks[batch_start : batch_start + batch_size]
        try:
            embeddings = await bedrock.embed_texts(
                [c.content for c in batch], input_type="search_document"
            )
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"embedding_count_mismatch: expected={len(batch)}, "
                    f"got={len(embeddings)}"
                )
            for chunk, embedding in zip(batch, embeddings):
                metadata_dict = {
                    "doc_name": request.metadata.doc_name,
                    "page": chunk.page,
                    "category": request.metadata.category,
                }
                if request.metadata.enforcement_date is not None:
                    metadata_dict["enforcement_date"] = str(
                        request.metadata.enforcement_date
                    )
                embedded_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "embedding": embedding,
                    "metadata": metadata_dict,
                })
        except Exception:
            logger.exception(
                "배치 임베딩 실패: offset=%d, size=%d", batch_start, len(batch)
            )
            for i in range(len(batch)):
                failed_chunks.append(
                    {"index": batch_start + i, "error": "embedding_failed"}
                )

    # 2. 성공분이 있을 때만 기존 청크/캐시 삭제 + 새 청크 삽입
    #    (RPC 단일 트랜잭션 — INSERT 실패 시 DELETE 자동 롤백, 이슈 #43)
    inserted_count = 0
    if embedded_chunks:
        inserted_count = await postgres.replace_document_chunks(
            request.doc_id, embedded_chunks
        )

    # 3. 응답
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
        "message": "문서 벡터화 및 적재가 완료되었습니다.",
    }


@router.delete(
    "/api/documents/{doc_id}",
    tags=["Documents"],
    summary="문서 삭제 및 캐시 무효화",
    operation_id="delete_document",
    description=(
        "지정 문서(`doc_id`)의 벡터 청크를 전부 삭제하고 관련 답변 캐시를 무효화합니다.\n\n"
        "- **멱등성 보장**: 존재하지 않는 `doc_id` 에 대해서도 `deleted_chunk_count=0` 으로 200 응답\n"
        "- 재색인 전 clean-up 용도로 사용 가능"
    ),
    responses={
        200: {
            "description": "삭제 성공 (멱등)",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "doc_id": 20240001,
                        "deleted_chunk_count": 12,
                        "invalidated_cache_count": 3,
                        "message": "해당 문서의 벡터 데이터 및 관련 캐시가 정상적으로 삭제되었습니다.",
                    }
                }
            },
        },
        503: {"model": ErrorResponse, "description": "외부 서비스 장애"},
    },
)
async def delete_document(
    doc_id: int = Path(..., ge=1),
    postgres: PostgresVectorClient = Depends(get_postgres_client),
) -> dict:
    """문서 벡터 데이터를 삭제하고 관련 캐시를 무효화한다.

    존재하지 않는 doc_id에 대해서도 카운트 0으로 성공 응답을 반환한다 (멱등성).
    """
    deleted_chunk_count = await postgres.delete_document_chunks(doc_id)
    invalidated_cache_count = await postgres.invalidate_cache_by_doc_id(doc_id)

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
        "message": "해당 문서의 벡터 데이터 및 관련 캐시가 정상적으로 삭제되었습니다.",
    }
