"""서버 시작 검증 모듈. lifespan 이벤트에서 호출하여 외부 의존성과 설정을 검증한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings

logger = logging.getLogger(__name__)


async def validate_startup(
    settings: Settings,
    bedrock: BedrockClient,
    supabase: SupabaseVectorClient,
) -> None:
    """서버 시작 시 외부 의존성 연결과 임베딩 차원 정합성을 검증한다.

    검증 실패 시 RuntimeError를 발생시켜 서버 시작을 중단한다.
    """
    logger.info("서버 시작 검증 시작")

    # 1. Supabase 연결 확인
    if not await supabase.health_check():
        raise RuntimeError("서버 시작 실패: Supabase Vector DB 연결에 실패했습니다.")
    logger.info("Supabase 연결 확인 완료")

    # 2. Bedrock LLM 연결 확인
    if not await bedrock.health_check_llm():
        raise RuntimeError("서버 시작 실패: Bedrock LLM 서비스 연결에 실패했습니다.")
    logger.info("Bedrock LLM 연결 확인 완료")

    # 3. Bedrock Embedding 연결 확인
    if not await bedrock.health_check_embedding():
        raise RuntimeError("서버 시작 실패: Bedrock Embedding 서비스 연결에 실패했습니다.")
    logger.info("Bedrock Embedding 연결 확인 완료")

    # 4. 임베딩 차원 정합성 검증
    await _validate_embedding_dimension(settings, bedrock)

    logger.info("서버 시작 검증 완료 — 모든 의존성 정상")


async def _validate_embedding_dimension(
    settings: Settings,
    bedrock: BedrockClient,
) -> None:
    """임베딩 모델의 출력 차원이 환경변수 EMBEDDING_DIMENSION과 일치하는지 검증한다."""
    try:
        embeddings = await bedrock.embed_texts(["차원 검증 테스트"], input_type="search_query")
    except Exception as exc:
        raise RuntimeError(
            f"서버 시작 실패: 임베딩 차원 검증 중 Bedrock 호출 실패 — {exc}"
        ) from exc

    actual_dimension = len(embeddings[0])
    expected_dimension = settings.EMBEDDING_DIMENSION

    if actual_dimension != expected_dimension:
        raise RuntimeError(
            f"서버 시작 실패: 임베딩 차원 불일치 — "
            f"모델 출력={actual_dimension}, 설정(EMBEDDING_DIMENSION)={expected_dimension}"
        )
    logger.info(
        "임베딩 차원 정합성 확인 완료: %d차원", actual_dimension
    )
