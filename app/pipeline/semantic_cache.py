"""시맨틱 캐시 모듈. 파이프라인 최선두에서 캐시 히트 여부를 판별한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext, SourceDoc

logger = logging.getLogger(__name__)


def _build_cache_key(user_context: str, question: str) -> str:
    """캐시 키 텍스트를 빌드한다.

    user_context 를 prefix 로 포함시켜 같은 user_context 사용자들끼리만 캐시를
    공유하도록 분리한다. 다른 학과/학년/학적상태 사용자에게 환각 답변이
    cross-누설되는 사고 (id=10 케이스) 를 임베딩 키 공간 분리로 구조적으로 차단.

    user_context 가 비어있으면 question 만 반환 — BE 폴백 경로 (프로필 미등록
    사용자에게 이름만 보내는 케이스) 호환.
    """
    if not user_context:
        return question
    return f"[사용자] {user_context}\n[질문] {question}"


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
        cache_key_text = _build_cache_key(
            context.user_context, context.original_question
        )

        # Batch 임베딩 — 캐시 키 + 검색 키 한 호출에 처리. cache miss 시
        # vector_search 가 question_embedding 을 재사용해 Task 16 효과 유지.
        # cache hit 시 question_embedding 은 사용되지 않으나 batch 비용이
        # 단건과 거의 동일해 분기 복잡도 회피가 더 가치 있다고 판단.
        embeddings = await bedrock.embed_texts(
            [cache_key_text, context.original_question],
            input_type="search_query",
        )
        cache_key_embedding, question_embedding = embeddings[0], embeddings[1]

        # 캐시 키 임베딩 보관 (_save_answer_cache 가 read-only 재사용)
        context.cache_key_embedding = cache_key_embedding

        # 검색 키 임베딩 + Task 16 메타 (vector_search 가 재사용)
        # input_type 도 함께 저장해 미래에 cache 측 input_type 이 바뀌어도
        # 하위 단계가 silent 하게 잘못 재사용하지 않도록 가드.
        context.question_embedding = question_embedding
        context.embedded_question_text = context.original_question
        context.embedded_question_input_type = "search_query"
        logger.debug(
            "질문 임베딩 컨텍스트 보관: text=%r, input_type=%s, cache_key_has_prefix=%s",
            context.original_question,
            "search_query",
            bool(context.user_context),
        )

        cache_match = await postgres.search_answer_cache(
            embedding=cache_key_embedding,
            threshold=settings.CACHE_SIMILARITY_THRESHOLD,
        )

        if cache_match is None:
            logger.debug("시맨틱 캐시 미스")
            return context

        parsed_sources = [SourceDoc.from_cache_dict(s) for s in cache_match.sources]
        context.cache_hit = True
        context.cached_answer = cache_match.answer
        context.cached_sources = parsed_sources
        context.cached_confidence = cache_match.confidence
        logger.info(
            "시맨틱 캐시 히트 (유사도=%.4f, 질문=%r)",
            cache_match.similarity_score,
            cache_match.question,
        )

    except Exception:
        logger.warning("시맨틱 캐시 탐색 실패 — 캐시 건너뛰고 계속 진행", exc_info=True)

    return context
