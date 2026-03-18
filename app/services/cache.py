"""
Supabase PGVector 기반 시맨틱 캐시
- question_logs 테이블에서 유사 질문 검색
- 질문-답변 쌍을 벡터와 함께 저장
"""
import hashlib
import app.services.vector_db as vector_db
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import CACHE_THRESHOLD


def _hash_msg(msg: str) -> str:
    """사용자 메시지의 SHA-256 해시를 반환"""
    return hashlib.sha256(msg.encode()).hexdigest()


def check_cache(user_msg: str):
    """question_logs에서 유사한 질문이 있는지 확인"""
    if not vector_db._db_available:
        return None

    normalized_msg = user_msg.replace(" ", "").replace("?", "").strip()
    if not normalized_msg:
        return None

    msg_hash = _hash_msg(user_msg)
    log_id = msg_hash[:12]

    try:
        query_embedding = vector_db.embeddings.embed_query(normalized_msg)
    except Exception:
        print(f"[CACHE ERROR] 임베딩 생성 실패 (id={log_id})")
        return None

    try:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with vector_db.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT answer, embedding <=> CAST(:embedding AS vector) AS distance
                    FROM question_logs
                    WHERE embedding IS NOT NULL AND answer IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT 1
                """),
                {"embedding": embedding_str}
            )
            row = result.fetchone()

        if row and row[1] < CACHE_THRESHOLD:
            print(f"[CACHE HIT] id={log_id} | 거리: {row[1]:.4f}")
            return row[0]
        elif row:
            print(f"[CACHE MISS] id={log_id} | 거리: {row[1]:.4f}")
    except SQLAlchemyError:
        print(f"[CACHE ERROR] DB 조회 실패 (id={log_id})")
    except Exception:
        print(f"[CACHE ERROR] 캐시 조회 중 예기치 않은 오류 (id={log_id})")

    return None


def update_cache(user_msg: str, answer: str):
    """질문-답변 쌍을 question_logs에 저장 (원문 대신 해시 저장)"""
    if not vector_db._db_available:
        return

    normalized_msg = user_msg.replace(" ", "").replace("?", "").strip()
    if not normalized_msg:
        return

    msg_hash = _hash_msg(user_msg)
    log_id = msg_hash[:12]

    try:
        query_embedding = vector_db.embeddings.embed_query(normalized_msg)
    except Exception:
        print(f"[CACHE ERROR] 임베딩 생성 실패 (id={log_id})")
        return

    try:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with vector_db.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO question_logs (question, embedding, answer, intent)
                    VALUES (:question, CAST(:embedding AS vector), :answer, :intent)
                """),
                {
                    "question": msg_hash,
                    "embedding": embedding_str,
                    "answer": answer,
                    "intent": "학사규정",
                }
            )
            conn.commit()
        print(f"[CACHE UPDATE] 캐시 저장 완료 (id={log_id})")
    except SQLAlchemyError:
        print(f"[CACHE ERROR] DB 저장 실패 (id={log_id})")
    except Exception:
        print(f"[CACHE ERROR] 캐시 저장 중 예기치 않은 오류 (id={log_id})")
