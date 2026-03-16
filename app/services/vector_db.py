"""
Supabase PGVector 기반 문서 검색 모듈
- documents 테이블에서 코사인 유사도 검색
- 커스텀 스키마에 맞춰 직접 쿼리
"""
from sqlalchemy import create_engine, text, URL
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE

print("[INIT] Google Gemini Embedding 모델 로딩 중...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    task_type="SEMANTIC_SIMILARITY",
    output_dimensionality=768,
)

# DB 엔진 생성 (URL.create로 유저명의 . 파싱 문제 방지)
try:
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
    )
    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[INIT] Supabase PostgreSQL 연결 성공!")
    _db_available = True
except Exception as e:
    engine = None
    _db_available = False
    print(f"[INIT] PostgreSQL 연결 실패 (더미 모드): {e}")


def search_documents(query: str, k: int = 3, threshold: float = 1.2) -> str:
    """documents 테이블에서 코사인 유사도 기반 검색"""
    if not _db_available:
        return ""

    try:
        query_embedding = embeddings.embed_query(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT content, embedding <=> :embedding AS distance
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> :embedding
                    LIMIT :k
                """),
                {"embedding": embedding_str, "k": k}
            )
            rows = result.fetchall()

        filtered = [row[0] for row in rows if row[1] <= threshold]

        if filtered:
            return "\n\n".join(filtered)
        else:
            return ""
    except Exception as e:
        print(f"[DB ERROR] 문서 검색 실패: {e}")
        return ""
