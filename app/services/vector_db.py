from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from app.core.config import PG_CONNECTION_STRING, COLLECTION_NAME

print("[INIT] 로컬 임베딩 모델 로딩 중...")
# 정규화(Normalization) 옵션을 켜서 거리를 0.0 ~ 2.0 으로 예쁘게 맞춥니다.
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    encode_kwargs={'normalize_embeddings': True}
)

# PostgreSQL (PGVector) 연결
try:
    vector_db = PGVector(
        connection_string=PG_CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    print("[INIT] PostgreSQL Vector DB 연결 성공!")
except Exception as e:
    vector_db = None
    print(f"[INIT] PostgreSQL 연결 대기 중 (더미 모드로 작동합니다): {e}")

def search_documents(query: str, k: int = 3) -> str:
    """DB에서 관련 규정을 검색하여 텍스트로 반환합니다."""
    if vector_db:
        docs = vector_db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    else:
        # DB 세팅 전 테스트용 하드코딩 데이터
        return "제22조(휴학) ① 휴학은 학기 또는 1년 단위로 가능하며, 통산 4년(8학기)을 초과할 수 없다."