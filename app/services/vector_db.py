"""
Supabase PGVector 기반 문서 검색 모듈
- documents 테이블에서 코사인 유사도 검색
- 커스텀 스키마에 맞춰 직접 쿼리
- DB 연결 실패 시 재시도 가능
"""
from sqlalchemy import create_engine, text, URL
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE

print("[INIT] Google Gemini Embedding 모델 로딩 중...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="SEMANTIC_SIMILARITY",
    output_dimensionality=768,
)

# DB 상태 (모듈 레벨)
engine = None
_db_available = False


def _ensure_db_connection() -> bool:
    """DB 연결을 확인하고, 미연결 시 재시도합니다."""
    global engine, _db_available
    if _db_available and engine is not None:
        return True
    try:
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
        )
        engine = create_engine(
            url,
            connect_args={"connect_timeout": 10},
            pool_pre_ping=True,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[INIT] Supabase PostgreSQL 연결 성공!")
        _db_available = True
        return True
    except Exception as e:
        engine = None
        _db_available = False
        print(f"[INIT] PostgreSQL 연결 실패 (더미 모드): {e}")
        return False


# 최초 연결 시도
_ensure_db_connection()


def search_documents(query: str, k: int = 3, threshold: float = 1.2) -> str:
    """documents 테이블에서 코사인 유사도 기반 검색"""
    if not _ensure_db_connection():
        return ""

    try:
        query_embedding = embeddings.embed_query(query)
    except Exception:
        print("[DB ERROR] 임베딩 생성 실패")
        return ""

    try:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with engine.connect() as conn:
            conn.execute(text("SET statement_timeout = '10s'"))
            result = conn.execute(
                text("""
                    SELECT content, embedding <=> CAST(:embedding AS vector) AS distance
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :k
                """),
                {"embedding": embedding_str, "k": k}
            )
            rows = result.fetchall()

        filtered = [row[0] for row in rows if row[1] <= threshold]
        return "\n\n".join(filtered) if filtered else ""
    except Exception:
        print("[DB ERROR] 문서 검색 실패")
        return ""
