# 🎓 국민대학교 학사규정 RAG 챗봇 API 서버

국민대학교 학사규정을 기반으로 질의응답하는 RAG(Retrieval-Augmented Generation) API 서버입니다.  
Vector DB에서 관련 규정을 검색하고, Google Gemini LLM을 통해 답변을 실시간 스트리밍(SSE)으로 반환합니다.

## 아키텍처

```
사용자 질문 → 시맨틱 캐시(FAISS) 확인
                ├─ 캐시 히트 → 캐시된 답변 반환
                └─ 캐시 미스 → PGVector 검색 → Gemini LLM 스트리밍 생성 → 캐시 저장
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 프레임워크 | FastAPI |
| LLM | Google Gemini 2.5 Flash |
| 임베딩 | `jhgan/ko-sroberta-multitask` (HuggingFace) |
| Vector DB | PostgreSQL + PGVector |
| 시맨틱 캐시 | FAISS (인메모리) |
| 오케스트레이션 | LangChain |

## 프로젝트 구조

```
app/
├── main.py              # FastAPI 앱 진입점
├── api/
│   └── chat.py          # /api/chat 엔드포인트 (캐시 → DB → LLM 파이프라인)
├── core/
│   └── config.py        # 환경변수 및 설정 관리
├── schemas/
│   └── request.py       # Pydantic 요청 스키마
└── services/
    ├── cache.py          # FAISS 기반 시맨틱 캐시
    ├── llm.py            # Gemini LLM 및 RAG 프롬프트
    └── vector_db.py      # PGVector 연결 및 문서 검색
```

## 시작하기

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 복사하여 `.env` 파일을 생성하고 값을 채워주세요.

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_google_api_key_here
PG_CONNECTION_STRING=postgresql+psycopg2://user:password@localhost:5432/rag_db
```

### 3. PostgreSQL + PGVector 준비

PGVector 확장이 설치된 PostgreSQL이 필요합니다. `rag_db` 데이터베이스에 학사규정 문서가 임베딩되어 있어야 합니다.

> DB가 없어도 더미 데이터로 기본 동작을 확인할 수 있습니다.

### 4. 서버 실행

```bash
uvicorn app.main:app --reload
```

서버가 실행되면 Swagger UI에서 API를 확인할 수 있습니다: http://127.0.0.1:8000/docs

## API 사용 예시

```bash
# 일상 인사
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕"}'

# 학사규정 질의
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "휴학 규정 알려줘"}'

# 시맨틱 캐시 테스트 (유사 질문)
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "휴학 규정을 알고싶어"}'
```

## 주요 기능

- SSE 스트리밍 응답: 답변을 실시간으로 토큰 단위 스트리밍
- 시맨틱 캐시: FAISS 기반으로 유사 질문에 대해 캐시된 답변 즉시 반환 (임계값: 0.35)
- 경량 라우터: 간단한 인사말은 LLM 호출 없이 즉시 응답
- RAG 파이프라인: PGVector 검색 → Gemini LLM 생성의 표준 RAG 흐름
