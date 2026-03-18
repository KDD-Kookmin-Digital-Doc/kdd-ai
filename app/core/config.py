import os
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY 환경변수가 설정되지 않았습니다. "
        ".env 파일에 GOOGLE_API_KEY=your_key 를 추가하세요."
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# DB 연결 설정 - PG_CONNECTION_STRING에서 파싱하거나 개별 변수 사용
_pg_url = os.getenv("PG_CONNECTION_STRING", "")
if _pg_url:
    _parsed = urlparse(_pg_url.replace("postgresql+psycopg2://", "postgresql://"))
    PG_USER = _parsed.username or ""
    PG_PASSWORD = _parsed.password or ""
    PG_HOST = _parsed.hostname or "localhost"
    PG_PORT = _parsed.port or 5432
    PG_DATABASE = _parsed.path.lstrip("/") or "postgres"
    if _parsed.hostname is None:
        raise ValueError(
            "PG_CONNECTION_STRING에서 호스트를 파싱할 수 없습니다. "
            "형식: postgresql+psycopg2://user:pass@host:port/db"
        )
else:
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")
    PG_DATABASE = os.getenv("PG_DATABASE", "postgres")

COLLECTION_NAME = "documents"
CACHE_COLLECTION_NAME = "question_logs"

# 시맨틱 캐시 임계값 (코사인 거리 기준, 낮을수록 엄격)
CACHE_THRESHOLD = 0.35

# 멀티턴 대화 설정
MAX_TURNS_BEFORE_SUMMARY = 5  # 이 턴 수 이상이면 대화 요약 실행
