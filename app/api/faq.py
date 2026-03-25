"""POST /api/faq/analyze 엔드포인트. FAQ 후보 자동 추출."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_bedrock_client, get_supabase_client
from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings, get_settings
from app.models.schemas import FAQAnalyzeRequest
from app.services.faq_analyzer import InsufficientDataError, analyze_faq

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/faq/analyze", response_model=None)
async def faq_analyze(
    request: FAQAnalyzeRequest,
    settings: Settings = Depends(get_settings),
    bedrock: BedrockClient = Depends(get_bedrock_client),
    supabase: SupabaseVectorClient = Depends(get_supabase_client),
):
    """누적 질문을 클러스터링하여 FAQ 후보를 추출한다."""
    try:
        candidates = await analyze_faq(
            questions=request.questions,
            top_k=request.top_k,
            min_cluster_size=request.min_cluster_size,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        return {
            "status": "success",
            "candidates": candidates,
        }

    except InsufficientDataError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error_code": "INSUFFICIENT_DATA",
                "message": str(exc),
            },
        )
