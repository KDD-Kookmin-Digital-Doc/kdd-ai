"""시맨틱 캐시 모듈. 파이프라인 최선두에서 캐시 히트 여부를 판별한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext, SourceDoc

logger = logging.getLogger(__name__)


async def check_cache(
    context: PipelineContext,
    is_first_message: bool,
    bedrock: BedrockClient,
    postgres: PostgresVectorClient,
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

        # Task 16: 임베딩 재사용을 위해 컨텍스트에 보관 (cache hit/miss 무관).
        # vector_search / _save_answer_cache 가 텍스트 + input_type 일치 시 재사용.
        # input_type 도 함께 저장해 미래에 cache 측 input_type 이 바뀌어도
        # 하위 단계가 silent 하게 잘못 재사용하지 않도록 가드.
        context.question_embedding = question_embedding
        context.embedded_question_text = context.original_question
        context.embedded_question_input_type = "search_query"
        logger.debug(
            "질문 임베딩 컨텍스트 보관: text=%r, input_type=%s",
            context.original_question,
            "search_query",
        )

        cache_match = await postgres.search_answer_cache(
            embedding=question_embedding,
            threshold=settings.CACHE_SIMILARITY_THRESHOLD,
        )

        if cache_match is None:
            logger.debug("시맨틱 캐시 미스")
            return context

        parsed_sources = [SourceDoc.from_cache_dict(s) for s in cache_match.sources]
        context.cache_hit = True
        context.cached_answer = cache_match.answer
        context.cached_sources = parsed_sources
        logger.info(
            "시맨틱 캐시 히트 (유사도=%.4f, 질문=%r)",
            cache_match.similarity_score,
            cache_match.question,
        )

    except Exception:
        logger.warning("시맨틱 캐시 탐색 실패 — 캐시 건너뛰고 계속 진행", exc_info=True)

    return context
