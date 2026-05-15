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

    # AWS Bedrock 설정 (기본값은 운영 환경 .env 와 일치 — 이슈 #45)
    BEDROCK_LIGHT_MODEL_ID: str = "apac.anthropic.claude-3-haiku-20240307-v1:0"
    BEDROCK_ANSWER_MODEL_ID: str = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
    BEDROCK_EMBEDDING_MODEL_ID: str = "global.cohere.embed-v4:0"
    AWS_REGION: str = "ap-northeast-2"

    # 임베딩 및 LLM 설정
    EMBEDDING_DIMENSION: int = 1024
    LLM_CONTEXT_WINDOW: int = 200000
    LLM_MAX_TOKENS: int = 1024
    # 문서 적재 시 한 번의 Bedrock embed 호출에 묶을 청크 수
    # (Cohere v4 read_timeout 초과 회피 + Throttling 균형). 양수 강제.
    EMBED_BATCH_SIZE: int = Field(default=48, gt=0)

    # PostgreSQL (RDS) 설정 — D3 마이그레이션 (Supabase → RDS asyncpg)
    DATABASE_URL: str
    POSTGRES_POOL_MIN_SIZE: int = Field(default=2, gt=0)
    POSTGRES_POOL_MAX_SIZE: int = Field(default=10, gt=0)

    # 유사도 임계값
    SIMILARITY_THRESHOLD: float = 0.75
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    # 벡터 검색 폴백(answer_cache 유사 질문 추출)에 적용할 floor 임계값.
    # 너무 낮으면 무관한 질문이 fallback 으로 노출됨 (이슈 #46).
    FALLBACK_SIMILARITY_THRESHOLD: float = 0.5

    # 검색/LLM 매직넘버 (PR-R5)
    VECTOR_SEARCH_TOP_K: int = Field(default=5, gt=0)
    FALLBACK_SUGGESTED_COUNT: int = Field(default=3, gt=0)
    REWRITE_MAX_TOKENS: int = Field(default=512, gt=0)
    INTENT_MAX_TOKENS: int = Field(default=16, gt=0)
    CHITCHAT_MAX_TOKENS: int = Field(default=256, gt=0)
    FAQ_LLM_MAX_TOKENS: int = Field(default=512, gt=0)
    BEDROCK_MAX_RETRIES: int = Field(default=2, ge=0)
    # FAQ 답변 초안 생성 시 동시 LLM 호출 수 (Bedrock throttling 가드)
    FAQ_CONCURRENCY: int = Field(default=5, gt=0)

    # confidence 임계값
    CONFIDENCE_HIGH_THRESHOLD: float = 0.9
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.8

    # Rerank 설정 (D1, Task 13)
    # Cohere Rerank 3.5 는 Single-region only — 도쿄(ap-northeast-1) 선택 (서울 RTT ~30ms).
    # RERANK_ENABLED=False 시 두 단계 retrieval 미사용, VECTOR_SEARCH_TOP_K 만 사용.
    RERANK_ENABLED: bool = True
    RERANK_REGION: str = "ap-northeast-1"
    RERANK_MODEL_ID: str = "cohere.rerank-v3-5:0"
    # Stage 1 (임베딩) 후보 수. RERANK_ENABLED=True 일 때만 사용.
    RETRIEVE_TOP_K_RERANK: int = Field(default=30, gt=0)
    # Stage 2 (리랭크) 최종 컨텍스트 개수.
    RERANK_TOP_N: int = Field(default=5, gt=0)
    # Rerank API timeout (도쿄 RTT + 모델 처리 여유).
    BEDROCK_RERANK_TIMEOUT: int = 15

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

    @model_validator(mode="after")
    def _validate_rerank_top_n(self) -> "Settings":
        # 역전 시 stage 2 슬라이스가 후보 수보다 커져 의미 없음.
        if self.RERANK_TOP_N > self.RETRIEVE_TOP_K_RERANK:
            raise ValueError(
                f"RERANK_TOP_N({self.RERANK_TOP_N}) 은 "
                f"RETRIEVE_TOP_K_RERANK({self.RETRIEVE_TOP_K_RERANK}) 이하여야 합니다."
            )
        return self

    @model_validator(mode="after")
    def _validate_postgres_pool_size(self) -> "Settings":
        # min > max 면 asyncpg.create_pool 이 런타임에 ValueError 를 던짐 — fail-fast.
        if self.POSTGRES_POOL_MIN_SIZE > self.POSTGRES_POOL_MAX_SIZE:
            raise ValueError(
                f"POSTGRES_POOL_MIN_SIZE({self.POSTGRES_POOL_MIN_SIZE}) 은 "
                f"POSTGRES_POOL_MAX_SIZE({self.POSTGRES_POOL_MAX_SIZE}) 이하여야 합니다."
            )
        return self

    # CORS 설정 (미설정 시 CORS 비활성)
    CORS_ORIGINS: list[str] = []

    # 캐시 만료 정책
    CACHE_TTL_DAYS: int = 90

    # 의도 분류 보조 컨텍스트 — BE가 history 최대 10턴까지 던지는 만큼 분류용으로도 넉넉히
    # 너무 적으면 멀티턴 학사 후속 질문이 chitchat으로 빠질 수 있고,
    # 너무 많으면 분류 LLM 입력 토큰이 커진다. 기본 6턴 = user/assistant 약 3쌍.
    INTENT_HISTORY_TURNS: int = Field(default=6, gt=0)
    # 분류 입력에 첨부할 history 메시지 1개의 최대 글자 수 (토큰 폭주 방지)
    INTENT_HISTORY_CHARS_PER_TURN: int = Field(default=200, gt=0)

    # 타임아웃 (초)
    BEDROCK_LLM_TIMEOUT: int = 30
    BEDROCK_EMBEDDING_TIMEOUT: int = 30
    # 임베딩 connect 타임아웃 (read와 분리 — connect 실패는 빠르게 감지)
    BEDROCK_EMBEDDING_CONNECT_TIMEOUT: int = 10
    POSTGRES_TIMEOUT: int = 10

    # asyncio 기본 ThreadPoolExecutor 크기. invoke_llm_stream / embed_texts /
    # rerank / stream close 가 잠시 점유하는 I/O 스레드 풀. Python 기본
    # (min(32, cpu+4)) 은 2 vCPU 박스에서 6 → 7번째 동시 LLM 스트림부터 큐잉.
    # 32 = Python 자체 상한과 동일, 2GB Lightsail 에서도 스택 ~256MB 안전 마진.
    THREAD_POOL_MAX_WORKERS: int = Field(default=32, gt=0)

    # 로깅 레벨 (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    # 알파테스트 동안엔 DEBUG 권장 (임베딩 재사용 등 최적화 효과 검증용),
    # 운영 안정화 후 INFO로 복귀해 노이즈 정리.
    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def _validate_log_level(self) -> "Settings":
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = self.LOG_LEVEL.upper()
        if normalized not in valid:
            raise ValueError(
                f"LOG_LEVEL 값이 유효하지 않습니다: {self.LOG_LEVEL!r}. "
                f"허용값: {sorted(valid)}"
            )
        # 정규화된 값으로 일관 유지 (대소문자 혼재 방지)
        object.__setattr__(self, "LOG_LEVEL", normalized)
        return self


@lru_cache
def get_settings() -> Settings:
    """싱글턴 Settings 인스턴스를 반환한다."""
    return Settings()
