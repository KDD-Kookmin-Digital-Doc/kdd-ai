"""POST /api/faq/analyze 엔드포인트. FAQ 후보 자동 추출."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_bedrock_client, get_postgres_client
from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings, get_settings
from app.models.schemas import ErrorResponse, FAQAnalyzeRequest
from app.services.faq_analyzer import InsufficientDataError, analyze_faq

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/faq/analyze",
    response_model=None,
    tags=["FAQ"],
    summary="FAQ 후보 자동 추출",
    operation_id="faq_analyze",
    description=(
        "누적된 사용자 질문 목록을 임베딩 후 클러스터링하여 상위 FAQ 후보를 반환합니다.\n\n"
        "- `top_k`: 반환할 상위 클러스터 개수 (기본 5, 최대 50)\n"
        "- `min_cluster_size`: 하나의 클러스터로 인정하기 위한 최소 질문 수 (기본 2)\n"
        "- 데이터가 충분하지 않으면 `INSUFFICIENT_DATA` 에러 코드로 400 반환"
    ),
    responses={
        200: {
            "description": "FAQ 후보 추출 성공",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "candidates": [
                            {
                                "question": "휴학 신청 기간 및 방법",
                                "draft_answer": "휴학은 매 학기 지정된 기간 내에 신청해야 하며...",
                                "category": "enrollment_status",
                                "frequency": 42,
                            },
                            {
                                "question": "학부별 졸업 학점 요건",
                                "draft_answer": "졸업을 위해서는 총 130학점 이상을 이수해야 하며...",
                                "category": "graduation",
                                "frequency": 28,
                            },
                        ],
                    }
                }
            },
        },
        400: {"model": ErrorResponse, "description": "필수 파라미터 누락 / 분석 데이터 부족"},
        422: {"model": ErrorResponse, "description": "타입 불일치 / 제약조건 위반"},
        503: {"model": ErrorResponse, "description": "외부 서비스 장애"},
    },
)
async def faq_analyze(
    request: FAQAnalyzeRequest,
    settings: Settings = Depends(get_settings),
    bedrock: BedrockClient = Depends(get_bedrock_client),
    postgres: PostgresVectorClient = Depends(get_postgres_client),
):
    """누적 질문을 클러스터링하여 FAQ 후보를 추출한다."""
    try:
        candidates = await analyze_faq(
            questions=request.questions,
            top_k=request.top_k,
            min_cluster_size=request.min_cluster_size,
            bedrock=bedrock,
            postgres=postgres,
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
