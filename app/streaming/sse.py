"""SSE 스트리밍 모듈. PipelineContext 상태에 따라 적절한 SSE 청크 시퀀스를 생성한다."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from app.clients.bedrock import BedrockClient
from app.config import Settings
from app.models.pipeline import PipelineContext
from app.pipeline.llm_generator import generate_response

logger = logging.getLogger(__name__)


_WHITESPACE_CHARS = (" ", "\n", "\t", "\r")


async def _buffer_by_word(
    token_stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """토큰 스트림을 whitespace 기준으로 묶어서 단어 단위로 yield한다.

    upstream이 예외로 종료되어도 잔여 버퍼를 먼저 flush한 뒤 예외를 재전파한다.
    """
    buf = ""
    try:
        async for token in token_stream:
            buf += token
            # 버퍼의 마지막 whitespace까지 flush (공백/개행/탭 모두 경계로 인식)
            last_ws = max(buf.rfind(c) for c in _WHITESPACE_CHARS)
            if last_ws != -1:
                yield buf[: last_ws + 1]
                buf = buf[last_ws + 1 :]
        if buf:
            yield buf
    except Exception:
        if buf:
            yield buf
        raise


def _format_sse_event(data: dict) -> str:
    """SSE 포맷 문자열을 생성한다."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def determine_confidence(
    search_results: list,
    settings: Settings,
) -> str:
    """검색 결과의 최고 유사도 기반으로 confidence 레벨을 반환한다."""
    if not search_results:
        return "low"

    max_score = max(r.similarity_score for r in search_results)

    if max_score >= settings.CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if max_score >= settings.CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


async def stream_sse_response(
    context: PipelineContext,
    bedrock: BedrockClient,
    settings: Settings,
) -> AsyncGenerator[str, None]:
    """PipelineContext 상태에 따라 적절한 SSE 시나리오를 선택하여 스트리밍한다.

    시나리오:
    - A (정상): meta(document) → text* → done
    - B (폴백): fallback → done
    - C (캐시 히트): meta(cache) → text → done
    - D (잡담): meta(chitchat) → text* → done
    - 에러: error → 종료 (done 없음)
    """
    try:
        # 시나리오 C: 캐시 히트
        if context.cache_hit:
            if not context.cached_answer:
                raise ValueError("cache_hit=True이지만 cached_answer가 없습니다")
            # Citation 컨트랙트: sources[N-1] 이 답변 본문의 {{N}} 마커와 매핑된다.
            # 캐시 답변엔 본 변경 이전에 저장된 답변(마커 0개)이 섞일 수 있고,
            # 본 변경 이후 저장 답변엔 마커가 포함된다. FE 는 마커 없으면 평문 렌더로
            # graceful 처리.
            sources = [
                {"doc_id": s.doc_id, "chunk_id": s.chunk_id, "doc_name": s.doc_name, "page": s.page}
                for s in context.cached_sources
            ]
            yield _format_sse_event({
                "type": "meta",
                "subtype": "cache",
                "cache_hit": True,
                "confidence": context.cached_confidence,
                "sources": sources,
            })
            yield _format_sse_event({
                "type": "text",
                "content": context.cached_answer,
            })
            yield _format_sse_event({
                "type": "done",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            })
            return

        # 시나리오 D: 잡담
        if context.intent == "chitchat":
            yield _format_sse_event({
                "type": "meta",
                "subtype": "chitchat",
                "intent": "chitchat",
            })
            async for chunk in _buffer_by_word(
                generate_response(context, bedrock, settings)
            ):
                yield _format_sse_event({
                    "type": "text",
                    "content": chunk,
                })
            yield _format_sse_event({
                "type": "done",
                "usage": {
                    "prompt_tokens": context.token_usage.prompt_tokens,
                    "completion_tokens": context.token_usage.completion_tokens,
                    "total_tokens": context.token_usage.total_tokens,
                },
            })
            return

        # 시나리오 B: 폴백 (검색 결과 없음)
        if not context.search_results:
            yield _format_sse_event({
                "type": "fallback",
                "message": "관련 문서를 찾을 수 없습니다. 아래 유사한 질문을 참고해 주세요.",
                "suggested_questions": context.suggested_questions,
            })
            yield _format_sse_event({
                "type": "done",
                "usage": {
                    "prompt_tokens": context.token_usage.prompt_tokens,
                    "completion_tokens": context.token_usage.completion_tokens,
                    "total_tokens": context.token_usage.total_tokens,
                },
            })
            return

        # 시나리오 A: 정상 (문서 검색 성공)
        confidence = determine_confidence(context.search_results, settings)
        # Citation 컨트랙트: sources[N-1] 이 답변 본문의 {{N}} 마커와 매핑된다.
        # 순서는 context.source_docs (검색 결과 / rerank 결과) 순서 그대로이며,
        # vector_search.py 에서 search_results 와 source_docs 가 동일 results 리스트로
        # 1:1 set 되어 LLM 프롬프트의 [문서 N] 인덱스와 자동 정합한다.
        sources = [
            {"doc_id": s.doc_id, "chunk_id": s.chunk_id, "doc_name": s.doc_name, "page": s.page}
            for s in context.source_docs
        ]
        yield _format_sse_event({
            "type": "meta",
            "subtype": "document",
            "confidence": confidence,
            "sources": sources,
        })
        async for chunk in _buffer_by_word(
            generate_response(context, bedrock, settings)
        ):
            yield _format_sse_event({
                "type": "text",
                "content": chunk,
            })
        yield _format_sse_event({
            "type": "done",
            "usage": {
                "prompt_tokens": context.token_usage.prompt_tokens,
                "completion_tokens": context.token_usage.completion_tokens,
                "total_tokens": context.token_usage.total_tokens,
            },
        })

    except Exception as exc:
        logger.error("SSE 스트리밍 중 오류 발생: %s", exc, exc_info=True)
        yield _format_sse_event({
            "type": "error",
            "message": "서비스 일시 장애가 발생했습니다.",
        })
