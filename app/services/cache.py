"""
Supabase PGVector 기반 시맨틱 캐시
- question_logs 테이블에서 유사 질문 검색
- 질문-답변 쌍을 벡터와 함께 저장
"""
from sqlalchemy import text
from app.services.vector_db import embeddings, engine, _db_available
from app.core.config import CACHE_THRESHOLD


def check_cache(user_msg: str):
    """question_logs에서 유사한 질문이 있는지 확인"""
    if not _db_available:
        return None

    normalized_msg = user_msg.replace(" ", "").replace("?", "").strip()

    try:
        query_embedding = embeddings.embed_query(normalized_msg)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT answer, embedding <=> :embedding AS distance
                    FROM question_logs
                    WHERE embedding IS NOT NULL AND answer IS NOT NULL
                    ORDER BY embedding <=> :embedding
                    LIMIT 1
                """),
                {"embedding": embedding_str}
            )
            row = result.fetchone()

        if row and row[1] < CACHE_THRESHOLD:
            print(f"[CACHE HIT] 입력: '{user_msg}' | 거리: {row[1]:.4f}")
            return row[0]
        elif row:
            print(f"[CACHE MISS] 입력: '{user_msg}' | 거리: {row[1]:.4f}")

    except Exception as e:
        print(f"[CACHE ERROR] 캐시 조회 실패: {e}")

    return None


def update_cache(user_msg: str, answer: str):
    """질문-답변 쌍을 question_logs에 저장"""
    if not _db_available:
        return

    normalized_msg = user_msg.replace(" ", "").replace("?", "").strip()

    try:
        query_embedding = embeddings.embed_query(normalized_msg)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO question_logs (question, embedding, answer, intent)
                    VALUES (:question, :embedding, :answer, :intent)
                """),
                {
                    "question": user_msg,
                    "embedding": embedding_str,
                    "answer": answer,
                    "intent": "학사규정",
                }
            )
            conn.commit()
        print("[CACHE UPDATE] 질문-답변이 DB에 저장되었습니다.")

    except Exception as e:
        print(f"[CACHE ERROR] 캐시 저장 실패: {e}")
