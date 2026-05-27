"""시맨틱 캐시 모듈. 파이프라인 최선두에서 캐시 히트 여부를 판별한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext, SourceDoc

logger = logging.getLogger(__name__)

# 캐시 키 임베딩의 input_type 상수. semantic_cache.check_cache (search) 와
# chat._save_answer_cache (save) 양쪽이 같은 상수를 참조해 invariant 를
# 구조적으로 성립시킨다. Task 16 패턴의 (text, input_type) 가드를 두는 대신
# 단일 상수로 일원화 — 미래에 input_type 을 바꾸려면 이 상수 하나만 수정하면
# search/save 가 자동 동기.
CACHE_EMBED_INPUT_TYPE = "search_query"

# _build_cache_key 의 마커 — _extract_question_from_cache_key 의 역연산
# 마커도 동일 상수 참조.
_CACHE_KEY_USER_MARKER = "[사용자] "
_CACHE_KEY_QUESTION_MARKER = "\n[질문] "


def _build_cache_key(user_context: str, question: str) -> str:
    """캐시 키 텍스트를 빌드한다.

    user_context 를 prefix 로 포함시켜 같은 user_context 사용자들끼리만 캐시를
    공유하도록 분리한다. 다른 학과/학년/학적상태 사용자에게 환각 답변이
    cross-누설되는 사고 (id=10 케이스) 를 임베딩 키 공간 분리 + dedup 키 분리로
    구조적으로 차단.

    이 결과가 ``answer_cache.question`` 컬럼에 저장되어 RPC ``upsert_answer_cache``
    의 ``DELETE WHERE question = ...`` dedup 키 + ``pg_advisory_xact_lock`` 락 키로
    동시에 사용된다. 즉 cohort 별 row 가 따로 보존되고 락도 cohort 별 직렬화.

    user_context 가 비어있으면 question 만 반환 — BE 폴백 경로 (프로필 미등록
    사용자에게 이름만 보내는 케이스) 호환.
    """
    if not user_context:
        return question
    return f"{_CACHE_KEY_USER_MARKER}{user_context}{_CACHE_KEY_QUESTION_MARKER}{question}"


def _extract_question_from_cache_key(cache_key_text: str) -> str:
    """_build_cache_key 의 역연산. answer_cache.question 컬럼의 prefix 포함
    텍스트에서 사용자 노출용 원문 질문만 추출한다.

    fallback 경로 (suggested_questions) 가 사용자 화면에 prefix 텍스트 그대로
    노출되지 않도록 변환한다. prefix 없는 텍스트 (BE 폴백 경로 / 레거시 row) 는
    그대로 반환 — _build_cache_key 가 user_context 빈 경우 question 만 반환하는
    contract 와 정합.
    """
    idx = cache_key_text.find(_CACHE_KEY_QUESTION_MARKER)
    if idx == -1:
        return cache_key_text
    return cache_key_text[idx + len(_CACHE_KEY_QUESTION_MARKER):]


async def fetch_similar_questions_for_user(
    postgres: PostgresVectorClient,
    embedding: list[float],
    top_k: int,
    threshold: float,
) -> list[str]:
    """fallback 추천용 유사 질문 조회 — extract + dedup + over-fetch 캡슐화.

    ``answer_cache.question`` 컬럼에 cohort prefix 텍스트가 저장된 후, 같은
    원문 질문이 cohort 별 여러 row 로 등록될 수 있다. 그대로 노출하면 사용자
    화면에 동일 추천이 중복 표시되므로 다음을 처리:

    1. over-fetch: ``top_k * 2`` 개를 조회해 dedup 여유 확보
    2. extract: 각 결과에서 prefix 제거해 원문만 복원
    3. order-preserving dedup: 동일 원문 중복 제거 (유사도 순서 유지)
    4. slice: 최대 ``top_k`` 개 반환

    이 함수가 단일 진입점이라 미래의 consumer 가 prefix 제거를 잊는
    구조적 사고를 차단 — fetch+extract 가 강결합 (외부 리뷰 7.6).
    """
    raw = await postgres.search_similar_questions(
        embedding=embedding,
        top_k=top_k * 2,
        threshold=threshold,
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for cache_key_text in raw:
        original = _extract_question_from_cache_key(cache_key_text)
        if original in seen:
            continue
        seen.add(original)
        deduped.append(original)
        if len(deduped) >= top_k:
            break
    return deduped


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
        # input_type 은 CACHE_EMBED_INPUT_TYPE 상수로 일원화 — _save_answer_cache
        # 가 같은 상수 참조해 search/save invariant 구조적 성립.
        embeddings = await bedrock.embed_texts(
            [cache_key_text, context.original_question],
            input_type=CACHE_EMBED_INPUT_TYPE,
        )
        cache_key_embedding, question_embedding = embeddings[0], embeddings[1]

        # 캐시 키 임베딩 보관 (_save_answer_cache 가 read-only 재사용)
        context.cache_key_embedding = cache_key_embedding

        # 검색 키 임베딩 + Task 16 메타 (vector_search 가 재사용)
        # input_type 도 함께 저장해 미래에 cache 측 input_type 이 바뀌어도
        # 하위 단계가 silent 하게 잘못 재사용하지 않도록 가드.
        context.question_embedding = question_embedding
        context.embedded_question_text = context.original_question
        context.embedded_question_input_type = CACHE_EMBED_INPUT_TYPE
        logger.debug(
            "질문 임베딩 컨텍스트 보관: text=%r, input_type=%s, cache_key_has_prefix=%s",
            context.original_question,
            CACHE_EMBED_INPUT_TYPE,
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
