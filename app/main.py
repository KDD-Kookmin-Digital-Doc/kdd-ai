"""FastAPI 앱 엔트리포인트. 라우터, 예외 핸들러, lifespan 이벤트를 통합한다."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, faq, health
from app.api.dependencies import get_bedrock_client, get_supabase_client
from app.api.error_handlers import register_error_handlers
from app.config import get_settings
from app.startup import validate_startup

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 이벤트를 관리한다."""
    settings = get_settings()
    bedrock = get_bedrock_client()
    supabase = get_supabase_client()

    await validate_startup(settings, bedrock, supabase)
    logger.info("AI 서버 시작 완료")

    yield

    logger.info("AI 서버 종료")


def create_app() -> FastAPI:
    """FastAPI 앱 인스턴스를 생성하고 설정한다."""
    app = FastAPI(
        title="학사규정 RAG AI 챗봇",
        description="학사규정 RAG(검색 증강 생��) AI 챗봇 서버",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 라우터 등록
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(faq.router)
    app.include_router(health.router)

    # 글로벌 예외 핸들러 등록
    register_error_handlers(app)

    # CORS 미들웨어 (CORS_ORIGINS 환경변수 설정 시에만 활성화)
    settings = get_settings()
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS 활성화: %s", settings.CORS_ORIGINS)

    return app


app = create_app()
