"""FastAPI 앱 엔트리포인트. 라우터, 예외 핸들러, lifespan 이벤트를 통합한다."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, faq, health
from app.api.chat import wait_pending_cache_writes
from app.api.dependencies import get_bedrock_client, get_postgres_client
from app.api.error_handlers import register_error_handlers
from app.config import get_settings
from app.logging_context import SessionIdFilter
from app.startup import validate_startup

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 이벤트를 관리한다."""
    settings = get_settings()

    # asyncio 기본 ThreadPoolExecutor 를 명시 크기로 교체. Python 기본
    # (min(32, cpu+4)) 은 2 vCPU Lightsail 에서 6 → invoke_llm_stream 이
    # 요청당 1 스레드를 점유하므로 7번째 동시 LLM 스트림부터 큐잉됨.
    # validate_startup 의 embed_texts 도 to_thread 를 쓰므로 교체를 먼저 수행.
    executor = ThreadPoolExecutor(
        max_workers=settings.THREAD_POOL_MAX_WORKERS,
        thread_name_prefix="kdd-ai-io",
    )
    asyncio.get_running_loop().set_default_executor(executor)
    logger.info(
        "기본 ThreadPoolExecutor 확장 완료: max_workers=%d",
        settings.THREAD_POOL_MAX_WORKERS,
    )

    bedrock = get_bedrock_client()
    postgres = get_postgres_client()

    try:
        await validate_startup(settings, bedrock, postgres)
        logger.info("AI 서버 시작 완료")
        yield
    finally:
        # startup 실패 시에도 cleanup 보장 — executor / postgres pool 좀비 차단.
        # PostgresVectorClient.close() 는 _pool is None 가드로 idempotent.
        await wait_pending_cache_writes()
        await postgres.close()
        executor.shutdown(wait=True)
        logger.info("AI 서버 종료")


OPENAPI_TAGS = [
    {
        "name": "Chat",
        "description": "RAG 파이프라인 기반 대화 엔드포인트. SSE 스트리밍 응답.",
    },
    {
        "name": "Documents",
        "description": "문서 청크 벡터화 적재 및 삭제. 동일 doc_id 재적재 시 last-write-wins.",
    },
    {
        "name": "FAQ",
        "description": "누적 질문 클러스터링을 통한 FAQ 후보 자동 추출.",
    },
    {
        "name": "Health",
        "description": "서버 기동 및 외부 의존성(Vector DB, Bedrock) 상태 점검.",
    },
]

API_DESCRIPTION = """
학사규정 RAG(검색 증강 생성) AI 챗봇 서버의 REST API 명세입니다.

### 주요 기능
- **Chat**: SSE 스트리밍 기반 RAG 대화
- **Documents**: 규정 문서 벡터화 적재 / 삭제
- **FAQ**: 누적 질문 클러스터링으로 FAQ 후보 추출
- **Health**: 서버 및 외부 의존성 상태 체크

### 에러 응답 스키마
`/api/health` 를 제외한 모든 엔드포인트의 실패 응답은 아래 공통 스키마를 따릅니다.
(`/api/health` 는 503 상황에서도 `ErrorResponse` 가 아닌 동일한 `HealthResponse`
본문을 반환합니다.)

```json
{ "status": "error", "error_code": "VALIDATION_ERROR", "message": "..." }
```

| HTTP | error_code                            | 설명                       |
|------|---------------------------------------|----------------------------|
| 400  | `BAD_REQUEST`, `INSUFFICIENT_DATA`    | 필수 파라미터 누락 등      |
| 422  | `VALIDATION_ERROR`                    | 타입·제약조건 위반         |
| 500  | `INTERNAL_ERROR`                      | 예상치 못한 서버 내부 오류 |
| 503  | `SERVICE_UNAVAILABLE`                 | Bedrock/Vector DB 장애     |
"""


def create_app() -> FastAPI:
    """FastAPI 앱 인스턴스를 생성하고 설정한다."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s][session=%(session_id)s] %(message)s",
    )
    # 모든 핸들러에 SessionIdFilter 부착 — 모든 LogRecord 에 session_id 자동 첨부 (이슈 #50).
    # default="-" 라 startup/lifespan 시점 로그도 안전하게 출력됨.
    session_filter = SessionIdFilter()
    for handler in logging.root.handlers:
        handler.addFilter(session_filter)

    app = FastAPI(
        title="학사규정 RAG AI 챗봇",
        description=API_DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        contact={
            "name": "KDD Kookmin Digital Doc",
            "url": "https://github.com/KDD-Kookmin-Digital-Doc/kdd-ai",
        },
        license_info={"name": "MIT"},
        swagger_ui_parameters={
            "docExpansion": "list",
            "defaultModelsExpandDepth": 1,
            "defaultModelExpandDepth": 2,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "syntaxHighlight.theme": "monokai",
        },
    )

    # 라우터 등록
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(faq.router)
    app.include_router(health.router)

    # 글로벌 예외 핸들러 등록
    register_error_handlers(app)

    # CORS 미들웨어 (CORS_ORIGINS 환경변수 설정 시에만 활성화)
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
