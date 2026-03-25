"""GET /api/health 엔드포인트. 외부 의존성 헬스체크."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_bedrock_client, get_supabase_client
from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/health", response_model=None)
async def health_check(
    bedrock: BedrockClient = Depends(get_bedrock_client),
    supabase: SupabaseVectorClient = Depends(get_supabase_client),
):
    """서버 및 외부 의존성 상태를 확인한다.

    모든 의존성 정상 → HTTP 200, 하나 이상 비정상 → HTTP 503.
    """
    vector_db_ok = await supabase.health_check()
    bedrock_llm_ok = await bedrock.health_check_llm()
    bedrock_embedding_ok = await bedrock.health_check_embedding()

    dependencies = {
        "vector_db": "healthy" if vector_db_ok else "unhealthy",
        "bedrock_llm": "healthy" if bedrock_llm_ok else "unhealthy",
        "bedrock_embedding": "healthy" if bedrock_embedding_ok else "unhealthy",
    }

    all_healthy = vector_db_ok and bedrock_llm_ok and bedrock_embedding_ok
    status = "healthy" if all_healthy else "unhealthy"
    status_code = 200 if all_healthy else 503

    body = {
        "status": status,
        "dependencies": dependencies,
    }

    if not all_healthy:
        logger.warning("헬스체크 비정상: %s", dependencies)

    return JSONResponse(status_code=status_code, content=body)
