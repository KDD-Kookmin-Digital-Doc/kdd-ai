"""애플리케이션 설정 모듈. Pydantic Settings를 활용한 환경 변수 로드."""

from functools import lru_cache

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

    # Supabase 설정
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # 유사도 임계값
    SIMILARITY_THRESHOLD: float = 0.75
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # 캐시 만료 정책
    CACHE_TTL_DAYS: int = 90

    # 타임아웃 (초)
    BEDROCK_LLM_TIMEOUT: int = 30
    BEDROCK_EMBEDDING_TIMEOUT: int = 15
    SUPABASE_TIMEOUT: int = 10


@lru_cache
def get_settings() -> Settings:
    """싱글턴 Settings 인스턴스를 반환한다."""
    return Settings()
