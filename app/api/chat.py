"""POST /api/chat 엔드포인트. RAG 파이프라인 오케스트레이션."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_bedrock_client, get_postgres_client
from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings, get_settings
from app.logging_context import set_session_id
from app.models.pipeline import AnswerCache, PipelineContext
from app.models.schemas import ChatRequest, ErrorResponse
from app.pipeline.intent_router import classify_intent
from app.pipeline.query_rewriter import rewrite_query
from app.pipeline.semantic_cache import check_cache
from app.pipeline.vector_search import search_documents
from app.streaming.sse import determine_confidence, stream_sse_response

logger = logging.getLogger(__name__)

_pending_cache_writes: set[asyncio.Task] = set()


async def wait_pending_cache_writes() -> None:
    """진행 중인 캐시 쓰기 태스크를 모두 기다린다. lifespan 종료 시 호출."""
    while _pending_cache_writes:
        pending = tuple(_pending_cache_writes)
        logger.info("캐시 쓰기 태스크 %d개 완료 대기 중...", len(pending))
        await asyncio.gather(*pending, return_exceptions=True)
        _pending_cache_writes.difference_update(pending)
    logger.info("캐시 쓰기 태스크 완료")

router = APIRouter()


async def _run_pipeline(
    request: ChatRequest,
    bedrock: BedrockClient,
    postgres: PostgresVectorClient,
    settings: Settings,
) -> PipelineContext:
    """RAG 파이프라인을 순차 실행하고 PipelineContext를 반환한다."""
    logger.info("파이프라인 시작: session=%s, question=%r", request.session_id, request.message)

    context = PipelineContext(
        original_question=request.message,
        user_context=request.user_context,
        session_id=request.session_id,
        history=[
            {"role": m.role, "content": m.content}
            for m in request.history
        ],
    )

    # 1. 시맨틱 캐시
    context = await check_cache(
        context, request.is_first_message, bedrock, postgres, settings
    )
    if context.cache_hit:
        return context

    # 2. 의도 분류 (rewrite 이전 — history 톤이 잡담을 학사로 비트는 것을 차단)
    context = await classify_intent(context, bedrock, settings)

    # 3. 잡담이면 rewrite·벡터 검색 모두 우회
    if context.intent == "chitchat":
        return context

    # 4. 질문 재작성 (academic 경로에서만)
    context = await rewrite_query(context, bedrock, settings)

    # 5. 벡터 검색
    context = await search_documents(context, bedrock, postgres, settings)

    return context


async def _streaming_wrapper(
    context: PipelineContext,
    bedrock: BedrockClient,
    postgres: PostgresVectorClient,
    settings: Settings,
    http_request: Request,
    is_first_message: bool,
) -> AsyncGenerator[str, None]:
    """SSE 스트리밍을 래핑하여 답변 버퍼링 + 캐시 저장 + disconnect 감지를 처리한다."""
    answer_buffer: list[str] = []
    stream_completed = False
    should_cache = (
        is_first_message
        and not context.cache_hit
        and context.intent == "academic"
        and context.search_results
    )

    stream = stream_sse_response(context, bedrock, settings)
    try:
        async for chunk in stream:
            if await http_request.is_disconnected():
                logger.info("클라이언트 disconnect 감지 — 스트리밍 중단")
                return

            # 이벤트 파싱
            event = None
            if chunk.startswith("data: "):
                try:
                    event = json.loads(chunk.removeprefix("data: ").strip())
                except (json.JSONDecodeError, AttributeError):
                    pass

            # 답변 텍스트 버퍼링 (캐시 저장용)
            if event and event.get("type") == "text" and should_cache and event.get("content"):
                answer_buffer.append(event["content"])
            elif event and event.get("type") == "done":
                stream_completed = True

            yield chunk
    finally:
        await stream.aclose()

    # 스트리밍 완료 후 캐시 저장 (정상 완료 + 학사규정 답변 시에만)
    if should_cache and stream_completed and answer_buffer:
        task = asyncio.create_task(
            _save_answer_cache(context, answer_buffer, bedrock, postgres, settings)
        )
        _pending_cache_writes.add(task)
        task.add_done_callback(_pending_cache_writes.discard)


async def _save_answer_cache(
    context: PipelineContext,
    answer_parts: list[str],
    bedrock: BedrockClient,
    postgres: PostgresVectorClient,
    settings: Settings,
) -> None:
    """답변 캐시를 비동기로 저장한다. 실패 시 로그만 남긴다.

    context 는 read-only 가정 — background task 진입 시점에 호출자(chat handler)는
    이미 SSE 응답을 완료해 context 를 더 이상 변조하지 않는다. set-once + lifecycle
    분리 패턴이라 동시 변조 race 없음. 미래에 분석/메트릭 등 응답 완료 후에도
    context 를 만지는 코드가 추가되면 snapshot 패턴 도입 검토.
    """
    try:
        full_answer = "".join(answer_parts)
        # Task 16: semantic_cache가 original_question을 search_query로 임베딩한 결과를 재사용.
        # vector_search가 rewrite 분기에서 새 임베딩을 만들어도 context.question_embedding은
        # 갱신되지 않으므로 여기서도 original 측 임베딩이 안전하게 유지됨.
        # input_type 까지 비교해 미래에 cache 측 input_type 이 바뀌면 자동으로 새로 호출.
        if (
            context.question_embedding is not None
            and context.embedded_question_text == context.original_question
            and context.embedded_question_input_type == "search_query"
        ):
            question_embedding = context.question_embedding
            logger.debug("_save_answer_cache 임베딩 재사용")
        else:
            embeddings = await bedrock.embed_texts(
                [context.original_question], input_type="search_query"
            )
            question_embedding = embeddings[0]
            logger.debug("_save_answer_cache 임베딩 신규 호출")

        source_doc_ids = list({r.doc_id for r in (context.search_results or [])})
        sources = [
            {"doc_id": s.doc_id, "chunk_id": s.chunk_id, "doc_name": s.doc_name, "page": s.page}
            for s in context.source_docs
        ]
        # 저장 시점의 confidence 를 박제 — cache hit 시 SSE meta(시나리오 C)에
        # 동일 값을 노출해 cache miss(시나리오 A)와 UX 정합 유지.
        confidence = determine_confidence(context.search_results, settings)

        cache = AnswerCache(
            question=context.original_question,
            embedding=question_embedding,
            answer=full_answer,
            source_doc_ids=source_doc_ids,
            sources=sources,
            confidence=confidence,
        )
        await postgres.upsert_answer_cache(cache)
        logger.info("답변 캐시 저장 완료: %r", context.original_question)
    except Exception:
        logger.warning("답변 캐시 저장 실패 — 사용자 응답에 영향 없음", exc_info=True)


@router.post(
    "/api/chat",
    tags=["Chat"],
    summary="RAG 챗봇 대화 (SSE 스트리밍)",
    operation_id="chat_stream",
    description=(
        "학사규정 RAG 파이프라인을 실행하고 **SSE(Server-Sent Events)** 스트림으로 답변을 반환합니다.\n\n"
        "### 파이프라인 단계\n"
        "1. 시맨틱 캐시 조회 (`is_first_message=true`인 경우)\n"
        "2. 의도 분류 (`academic` / `chitchat`) — history 톤이 분류를 비트는 것을 막기 위해 rewrite 이전에 수행\n"
        "3. 질문 재작성 (academic 경로에서만, 히스토리 기반 문맥화)\n"
        "4. 벡터 검색 (academic 인 경우)\n"
        "5. LLM 스트리밍 답변 생성\n\n"
        "### SSE 이벤트 포맷\n"
        "응답은 `text/event-stream` 이며 각 이벤트는 `data: <JSON>\\n\\n` 형태입니다. "
        "정상(academic) 시나리오는 `meta`(문서/신뢰도) → `text*`(스트리밍) → `done`(usage) 순으로 흐릅니다.\n"
        "```\n"
        "data: {\"type\": \"meta\", \"subtype\": \"document\", \"confidence\": \"high\", \"sources\": [...]}\n\n"
        "data: {\"type\": \"text\", \"content\": \"휴학은 신청서를 제출합니다{{1}}. \"}\n\n"
        "data: {\"type\": \"done\", \"usage\": {\"prompt_tokens\": 0, \"completion_tokens\": 0, \"total_tokens\": 0}}\n\n"
        "```\n\n"
        "캐시 히트 시에는 `meta`(subtype=cache) 에 저장 시점의 `confidence` 박제값이 노출되며 `text` 1회 + `done` 으로 종료됩니다.\n"
        "```\n"
        "data: {\"type\": \"meta\", \"subtype\": \"cache\", \"cache_hit\": true, \"confidence\": \"high\", \"sources\": [...]}\n\n"
        "data: {\"type\": \"text\", \"content\": \"휴학은 신청서를 제출합니다{{1}}.\"}\n\n"
        "data: {\"type\": \"done\", \"usage\": {\"prompt_tokens\": 0, \"completion_tokens\": 0, \"total_tokens\": 0}}\n\n"
        "```\n\n"
        "답변 본문 내 `{{N}}` 패턴은 `meta.sources[N-1]` 출처를 가리키는 인용 마커입니다. "
        "FE 는 마커를 클릭 가능한 footnote 로 렌더링하고, 마커가 없거나 범위 밖이면 평문으로 graceful degrade 합니다.\n\n"
        "### 테스트 (curl)\n"
        "```bash\n"
        "curl -N -X POST http://localhost:8000/api/chat \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        '  -d \'{"message":"휴학 신청 방법","session_id":"s1","user_context":"학부생","is_first_message":true,"history":[]}\'\n'
        "```\n\n"
        "> ⚠️ Swagger UI 의 *Try it out* 은 SSE 스트림을 온전히 표시하지 못할 수 있습니다. curl 또는 전용 클라이언트 사용을 권장합니다."
    ),
    responses={
        200: {
            "description": "SSE 스트림 응답 (text/event-stream)",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type": "meta", "subtype": "document", "confidence": "high", "sources": [{"doc_id": 20240001, "chunk_id": 1012, "doc_name": "학사규정_2024", "page": 12}]}\n\n'
                        'data: {"type": "text", "content": "휴학은 "}\n\n'
                        'data: {"type": "text", "content": "신청서를 제출하면 됩니다{{1}}."}\n\n'
                        'data: {"type": "done", "usage": {"prompt_tokens": 320, "completion_tokens": 48, "total_tokens": 368}}\n\n'
                    )
                }
            },
        },
        400: {"model": ErrorResponse, "description": "필수 파라미터 누락"},
        422: {"model": ErrorResponse, "description": "타입 불일치 / 제약조건 위반"},
        503: {"model": ErrorResponse, "description": "외부 서비스 장애"},
    },
)
async def chat(
    request: ChatRequest,
    http_request: Request,
    settings: Settings = Depends(get_settings),
    bedrock: BedrockClient = Depends(get_bedrock_client),
    postgres: PostgresVectorClient = Depends(get_postgres_client),
) -> StreamingResponse:
    """RAG 챗봇 대화 엔드포인트. SSE 스트리밍 응답을 반환한다."""
    # PR-50: contextvars 에 session_id 설정. 같은 task 내 모든 sub-coroutine 과
    # asyncio.create_task 로 분기되는 _save_answer_cache 까지 자동 전파.
    # FastAPI 는 요청마다 새 task 를 만들므로 reset 없이도 다른 요청과 격리됨.
    set_session_id(request.session_id)
    context = await _run_pipeline(request, bedrock, postgres, settings)

    return StreamingResponse(
        _streaming_wrapper(context, bedrock, postgres, settings, http_request, request.is_first_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
