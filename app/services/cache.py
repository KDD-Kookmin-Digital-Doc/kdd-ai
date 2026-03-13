from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.services.vector_db import embeddings
from app.core.config import CACHE_THRESHOLD
#질문이 들어오면 가장 먼저 살펴보는 FAISS 시맨틱 캐시
# 초기화 (메모리 DB)
dummy_faq = [Document(page_content="초기화용 더미 데이터", metadata={"answer": "더미"})]
semantic_cache = FAISS.from_documents(dummy_faq, embeddings)

def check_cache(user_msg: str):
    """캐시에 비슷한 질문이 있는지 확인"""
    # 정규화: 띄어쓰기, 물음표 제거
    normalized_msg = user_msg.replace(" ", "").replace("?", "").strip()
    
    results = semantic_cache.similarity_search_with_score(normalized_msg, k=1)
    if results:
        best_match, score = results[0]
        print(f"[CACHE CHECK] 입력: '{user_msg}' | 거리: {score:.4f}")
        
        if score < CACHE_THRESHOLD and best_match.page_content != "초기화용 더미 데이터":
            return best_match.metadata["answer"]
    return None

def update_cache(user_msg: str, answer: str):
    """새로운 질의응답을 캐시에 학습시킵니다."""
    normalized_msg = user_msg.replace(" ", "").replace("?", "").strip()
    new_doc = Document(page_content=normalized_msg, metadata={"answer": answer})
    semantic_cache.add_documents([new_doc])
    print("[CACHE UPDATE] 새로운 답변이 캐시에 저장되었습니다.")