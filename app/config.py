"""애플리케이션 설정 모듈. Pydantic Settings를 활용한 환경 변수 로드."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 변수 기반 애플리케이션 설정."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS Bedrock 설정
    BEDROCK_LIGHT_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
    BEDROCK_ANSWER_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    BEDROCK_EMBEDDING_MODEL_ID: str = "cohere.embed-multilingual-v3"
    AWS_REGION: str = "us-east-1"

    # 임베딩 및 LLM 설정
    EMBEDDING_DIMENSION: int = 1024
    LLM_CONTEXT_WINDOW: int = 200000
    LLM_MAX_TOKENS: int = 1024
    # 문서 적재 시 한 번의 Bedrock embed 호출에 묶을 청크 수
    # (Cohere v4 read_timeout 초과 회피 + Throttling 균형). 양수 강제.
    EMBED_BATCH_SIZE: int = Field(default=48, gt=0)

    # Supabase 설정
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # 유사도 임계값
    SIMILARITY_THRESHOLD: float = 0.75
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # confidence 임계값
    CONFIDENCE_HIGH_THRESHOLD: float = 0.9
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.8

    @model_validator(mode="after")
    def _validate_confidence_thresholds(self) -> "Settings":
        med = self.CONFIDENCE_MEDIUM_THRESHOLD
        high = self.CONFIDENCE_HIGH_THRESHOLD
        if not (0 <= med <= high <= 1):
            raise ValueError(
                f"confidence 임계값이 유효하지 않습니다: "
                f"0 <= MEDIUM({med}) <= HIGH({high}) <= 1 이어야 합니다."
            )
        return self

    # CORS 설정 (미설정 시 CORS 비활성)
    CORS_ORIGINS: list[str] = []

    # 캐시 만료 정책
    CACHE_TTL_DAYS: int = 90

    # 타임아웃 (초)
    BEDROCK_LLM_TIMEOUT: int = 30
    BEDROCK_EMBEDDING_TIMEOUT: int = 30
    # 임베딩 connect 타임아웃 (read와 분리 — connect 실패는 빠르게 감지)
    BEDROCK_EMBEDDING_CONNECT_TIMEOUT: int = 10
    SUPABASE_TIMEOUT: int = 10


@lru_cache
def get_settings() -> Settings:
    """싱글턴 Settings 인스턴스를 반환한다."""
    return Settings()
