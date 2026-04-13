"""시맨틱 캐시 모듈. 파이프라인 최선두에서 캐시 히트 여부를 판별한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext, SourceDoc

logger = logging.getLogger(__name__)


async def check_cache(
    context: PipelineContext,
    is_first_message: bool,
    bedrock: BedrockClient,
    supabase: SupabaseVectorClient,
    settings: Settings,
) -> PipelineContext:
    """시맨틱 캐시를 탐색하여 PipelineContext를 갱신한다.

    - ``is_first_message=True`` 일 때만 answer_cache 테이블에서 탐색.
    - ``is_first_message=False`` 이면 캐시 탐색을 건너뛴다.
    - 캐시 탐색 중 예외 발생 시 graceful degradation: 로그만 남기고
      캐시를 건너뛰어 다음 파이프라인 단계로 진행한다.
    """
    if not is_first_message:
        logger.debug("is_first_message=False — 캐시 탐색 건너뜀")
        return context

    try:
        embeddings = await bedrock.embed_texts(
            [context.original_question], input_type="search_query"
        )
        question_embedding = embeddings[0]

        cache_match = await supabase.search_answer_cache(
            embedding=question_embedding,
            threshold=settings.CACHE_SIMILARITY_THRESHOLD,
        )

        if cache_match is None:
            logger.debug("시맨틱 캐시 미스")
            return context

        context.cache_hit = True
        context.cached_answer = cache_match.answer
        context.cached_sources = [
            SourceDoc(
                doc_id=s.get("doc_id", ""),
                chunk_id=s["chunk_id"],
                doc_name=s["doc_name"],
                page=s["page"],
            )
            for s in cache_match.sources
        ]
        logger.info(
            "시맨틱 캐시 히트 (유사도=%.4f, 질문=%r)",
            cache_match.similarity_score,
            cache_match.question,
        )

    except Exception:
        logger.warning("시맨틱 캐시 탐색 실패 — 캐시 건너뛰고 계속 진행", exc_info=True)

    return context
