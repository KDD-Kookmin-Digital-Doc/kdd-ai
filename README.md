# 🎓 국민대학교 학사규정 RAG 챗봇 API 서버

국민대학교 학사규정을 기반으로 질의응답하는 RAG(Retrieval-Augmented Generation) API 서버입니다.  
Supabase PGVector에서 관련 규정을 검색하고, Google Gemini LLM을 통해 답변을 실시간 스트리밍(SSE)으로 반환합니다.

## 아키텍처

```
사용자 질문 → 경량 라우터 (인사말 필터링)
                │
                ├─ 인사말 → 즉시 응답 (LLM 호출 없음)
                │
                └─ 학사규정 질의
                     │
                     ├─ [싱글턴] 시맨틱 캐시(PGVector question_logs) 확인
                     │     ├─ 캐시 히트 → 캐시된 답변 반환
                     │     └─ 캐시 미스 → PGVector 검색 → Gemini LLM 스트리밍 → 캐시 저장
                     │
                     └─ [멀티턴] 대화 히스토리 관리
                           ├─ 5턴 이상 → LLM 요약 압축
                           └─ PGVector 검색 → 대화 맥락 포함 → Gemini LLM 스트리밍
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 프레임워크 | FastAPI |
| LLM | Google Gemini 2.5 Flash |
| 임베딩 | Google Gemini Embedding (`gemini-embedding-exp-03-07`, 768차원) |
| Vector DB | Supabase PostgreSQL + PGVector (documents 테이블) |
| 시맨틱 캐시 | Supabase PGVector (question_logs 테이블, 코사인 거리 임계값 0.35) |
| 대화 관리 | LangChain InMemoryChatMessageHistory + LLM 요약 압축 |
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
│   └── request.py       # Pydantic 요청 스키마 (message, session_id)
└── services/
    ├── cache.py          # PGVector 기반 시맨틱 캐시 (question_logs 테이블)
    ├── llm.py            # Gemini LLM 및 RAG 프롬프트 (싱글턴/멀티턴)
    ├── memory.py         # 세션 기반 멀티턴 대화 관리 및 요약 압축
    └── vector_db.py      # Supabase PGVector 연결 및 문서 검색 (documents 테이블)
```

## 주요 기능

- **SSE 스트리밍 응답**: 답변을 실시간으로 토큰 단위 스트리밍
- **시맨틱 캐시**: PGVector 기반으로 유사 질문에 대해 캐시된 답변 즉시 반환 (코사인 거리 임계값: 0.35)
- **멀티턴 대화**: `session_id`를 통한 세션 기반 대화 히스토리 관리
- **대화 요약 압축**: 5턴 이상 대화 시 LLM(Gemini 2.0 Flash)으로 자동 요약하여 컨텍스트 절약
- **경량 라우터**: 간단한 인사말은 LLM 호출 없이 즉시 응답
- **RAG 파이프라인**: PGVector 문서 검색 → Gemini LLM 생성의 표준 RAG 흐름

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

두 가지 방식으로 DB 연결을 설정할 수 있습니다:

**방법 A: 연결 문자열 (PG_CONNECTION_STRING)**
```env
GOOGLE_API_KEY=your_google_api_key_here
PG_CONNECTION_STRING=postgresql+psycopg2://user:password@host:port/database
```

**방법 B: 개별 변수**
```env
GOOGLE_API_KEY=your_google_api_key_here
PG_HOST=your_host
PG_PORT=6543
PG_USER=postgres.your_project_id
PG_PASSWORD=your_password
PG_DATABASE=postgres
```

> Supabase Pooler 사용 시 유저명에 `.`이 포함되므로 개별 변수 방식을 권장합니다.

### 3. Supabase DB 준비

Supabase 프로젝트에서 SQL Editor를 열고 다음 스키마를 실행하세요:

```sql
-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- RAG 문서 벡터 테이블
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 질문 로그 테이블 (시맨틱 캐시)
CREATE TABLE question_logs (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    embedding VECTOR(768),
    answer TEXT,
    intent VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- HNSW 인덱스 (코사인 유사도 검색 최적화)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON question_logs USING hnsw (embedding vector_cosine_ops);
```

### 4. 서버 실행

```bash
uvicorn app.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

## API 사용 예시

### 싱글턴 대화 (session_id 없음)

```bash
# 인사말
curl -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"안녕\"}"

# 학사규정 질의
curl -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학 규정 알려줘\"}"

# 시맨틱 캐시 테스트 (유사 질문)
curl -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학 규정을 알고싶어\"}"
```

### 멀티턴 대화 (session_id 포함)

```bash
# 첫 번째 질문
curl -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학 규정 알려줘\", \"session_id\": \"user-123\"}"

# 후속 질문 (이전 맥락 참조)
curl -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"기간 제한은?\", \"session_id\": \"user-123\"}"
```

> Windows CMD에서는 curl 명령을 한 줄로 작성하고, JSON 내부 큰따옴표를 `\"` 로 이스케이프해야 합니다.
