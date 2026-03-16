# 프로젝트 개선점 분석

## 1. 에러 핸들링 부재

현재 LLM 호출, DB 검색, 캐시 조회 어디에도 try-except가 없다.
Gemini API 장애, 네트워크 타임아웃, 토큰 한도 초과 등이 발생하면 500 에러가 그대로 클라이언트에 노출된다.

**개선 방향:**
- `chat.py`의 스트리밍 제너레이터 내부에 try-except 추가
- LLM 호출 실패 시 `data: [ERROR] 일시적인 오류가 발생했습니다\n\n` 형태로 SSE 에러 이벤트 전송
- FastAPI의 `@app.exception_handler`로 글로벌 에러 핸들러 등록
- `vector_db.py`의 `search_documents`에서 DB 쿼리 실패 시 빈 컨텍스트 반환 처리

```python
# 예시: chat.py 스트리밍 에러 핸들링
async def streamer():
    try:
        for chunk in rag_chain.stream(...):
            yield f"data: {chunk.content}\n\n"
    except Exception as e:
        print(f"[ERROR] LLM 스트리밍 실패: {e}")
        yield "data: [ERROR] 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n"
    yield "data: [DONE]\n\n"
```

---

## 2. 로깅 체계 미흡

모든 로그가 `print()`로 처리되고 있다. 운영 환경에서는 로그 레벨 구분, 파일 출력, 구조화된 로그가 필요하다.

**개선 방향:**
- Python `logging` 모듈 또는 `loguru` 도입
- DEBUG / INFO / WARNING / ERROR 레벨 구분
- 요청별 trace ID를 로그에 포함하여 추적 가능하게 구성

```python
import logging
logger = logging.getLogger("rag-server")

# print("[ROUTER] 규정 검색 진행 중...") 대신:
logger.info(f"[session={session_id}] 규정 검색 진행 중: '{user_msg[:30]}'")
```

---

## 3. 세션 메모리 누수 위험

`memory.py`의 `_session_store`와 `_summary_store`가 dict로 무한히 쌓인다.
서버가 오래 돌면 메모리가 계속 증가한다.

**개선 방향:**
- TTL(Time-To-Live) 기반 세션 만료 처리 (예: 30분 미사용 시 자동 삭제)
- `cachetools.TTLCache` 또는 Redis로 세션 저장소 교체
- 세션 수 상한선 설정 (LRU 방식으로 오래된 세션 제거)

```python
from cachetools import TTLCache

# 최대 1000개 세션, 30분 TTL
_session_store = TTLCache(maxsize=1000, ttl=1800)
_summary_store = TTLCache(maxsize=1000, ttl=1800)
```

---

## 4. 시맨틱 캐시 한계

현재 FAISS 캐시가 인메모리라서 서버 재시작 시 모든 캐시가 사라진다.
또한 캐시가 무한히 커질 수 있다.

**개선 방향:**
- FAISS 인덱스를 주기적으로 디스크에 저장/복원 (`faiss.write_index` / `faiss.read_index`)
- 캐시 항목 수 제한 (최대 N개 초과 시 오래된 항목 제거)
- 캐시 히트율 모니터링 로그 추가
- LangChain의 `GPTCache` 또는 Redis 기반 시맨틱 캐시로 전환 고려

---

## 5. 멀티턴에서 Vector DB 검색 쿼리 개선

현재 멀티턴 대화에서도 사용자의 마지막 메시지만으로 Vector DB를 검색한다.
"그러면 복학은?" 같은 대명사/생략형 질문은 검색 품질이 떨어진다.

**개선 방향:**
- 대화 히스토리를 기반으로 검색 쿼리를 재작성(Query Rewriting)하는 단계 추가
- LLM에게 "이전 대화 맥락을 고려하여 검색에 적합한 독립적인 질문으로 바꿔줘"라고 요청

```python
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "이전 대화 맥락을 참고하여, 사용자의 마지막 질문을 "
               "검색에 적합한 독립적인 질문으로 재작성하세요. 재작성된 질문만 출력하세요."),
    ("human", "대화 맥락:\n{chat_history}\n\n마지막 질문: {question}")
])
# 재작성된 쿼리로 Vector DB 검색
rewritten_query = query_rewrite_chain.invoke(...)
context_text = search_documents(rewritten_query)
```

---

## 6. 경량 라우터 로직 취약

인사말 감지 조건이 `len(user_msg) < 4`로 하드코딩되어 있어서,
"안녕하세요"(5글자)는 인사로 인식 못하고 LLM을 호출하게 된다.

**개선 방향:**
- 길이 제한 완화 또는 제거
- 인사말 패턴을 정규식이나 키워드 매칭으로 확장
- LLM 기반 의도 분류(Intent Classification)를 경량 모델로 처리하는 것도 고려

```python
import re
greeting_pattern = re.compile(r"^(안녕|반가워|누구야|고마워|하이|헬로|감사|안녕하세요|반갑습니다)[\s?!.]*$")
if greeting_pattern.match(user_msg):
    # 인사 응답
```

---

## 7. CORS 설정 없음

Spring Boot 백엔드에서 이 서버를 호출한다면 CORS 이슈가 발생할 수 있다.
현재 아무런 CORS 미들웨어가 없다.

**개선 방향:**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Spring Boot 주소
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 8. 헬스체크 엔드포인트 없음

서버 상태 확인, DB 연결 상태, 모델 로딩 상태를 확인할 수 있는 엔드포인트가 없다.
운영 환경에서 로드밸런서나 모니터링 도구가 서버 상태를 확인할 수 없다.

**개선 방향:**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "db_connected": vector_db is not None,
        "model_loaded": True,
    }
```

---

## 9. 비동기 처리 미흡

`rag_chain.stream()`과 `summarize_and_compress()`가 동기 호출이라
이벤트 루프를 블로킹할 수 있다. 동시 요청이 많아지면 성능 병목이 된다.

**개선 방향:**
- LangChain의 `.astream()` (비동기 스트리밍) 사용
- 요약 작업을 `asyncio.to_thread()`로 감싸거나 `BackgroundTasks`로 처리

```python
# 동기 stream → 비동기 astream
async for chunk in multiturn_rag_chain.astream({...}):
    yield f"data: {chunk.content}\n\n"
```

---

## 10. 환경별 설정 분리 없음

개발/스테이징/운영 환경에 따라 DB 주소, 모델, 캐시 임계값 등이 달라져야 하는데
현재는 단일 `.env`로만 관리된다.

**개선 방향:**
- Pydantic `BaseSettings`를 활용한 설정 클래스 도입
- 환경별 `.env.dev`, `.env.prod` 분리
- `APP_ENV` 환경변수로 어떤 설정을 로드할지 결정

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str = ""
    pg_connection_string: str = ""
    cache_threshold: float = 0.35
    max_turns_before_summary: int = 5
    app_env: str = "dev"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 우선순위 요약

| 순위 | 항목 | 난이도 | 영향도 |
|------|------|--------|--------|
| 1 | 에러 핸들링 | 낮음 | 높음 |
| 2 | 멀티턴 쿼리 재작성 | 중간 | 높음 |
| 3 | CORS 설정 | 낮음 | 중간 |
| 4 | 헬스체크 | 낮음 | 중간 |
| 5 | 세션 메모리 관리 | 중간 | 중간 |
| 6 | 비동기 전환 | 중간 | 중간 |
| 7 | 로깅 체계 | 낮음 | 중간 |
| 8 | 경량 라우터 개선 | 낮음 | 낮음 |
| 9 | 캐시 영속화 | 중간 | 낮음 |
| 10 | 환경별 설정 분리 | 중간 | 낮음 |
