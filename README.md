# 학사규정 RAG AI 챗봇 서버

학사규정 RAG(검색 증강 생성) AI 챗봇 서버. FastAPI 기반 독립 마이크로서비스로, 메인 백엔드(Spring)와 완전히 분리되어 무상태(Stateless)로 동작합니다.

## 기술 스택

| 분류 | 기술 |
|---|---|
| 런타임 | Python 3.11+, FastAPI, uvicorn |
| LLM (답변 생성) | AWS Bedrock — Claude 3.5 Sonnet |
| LLM (재작성/의도분류/잡담) | AWS Bedrock — Claude 3 Haiku |
| 임베딩 | AWS Bedrock — Cohere Embed Multilingual v3 (1024차원) |
| 벡터 DB | Supabase (PostgreSQL + pgvector) |
| 스트리밍 | SSE (Server-Sent Events) |
| 테스트 | pytest, pytest-asyncio, Hypothesis (속성 기반 테스트) |

## 아키텍처

```text
┌─────────────────────────────────────────────────────────┐
│                 프론트엔드 / 메인 백엔드 (Spring)          │
└──────┬──────────┬──────────────┬──────────────┬─────────┘
       │POST      │POST          │DELETE        │POST     │GET
       │/api/chat │/api/docs/embed │/api/docs/{id} │/api/faq │/api/health
       ▼          ▼              ▼              ▼         ▼
┌─────────────────────────────────────────────────────────┐
│                  AI Server (FastAPI)                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │                   API Router                       │ │
│  └──┬──────────┬──────────────┬──────────────┬────────┘ │
│     │          │              │              │          │
│     ▼          ▼              ▼              ▼          │
│  RAG        Document        FAQ          Health         │
│  Pipeline   Service         Service       Service       │
└─────┬──────────┬──────────────┬─────────────────────────┘
      │          │              │
      ▼          ▼              ▼
┌───────────┐ ┌──────────────┐ ┌──────────────────────┐
│ Bedrock   │ │ Bedrock      │ │ Supabase Vector DB   │
│ Claude    │ │ Cohere Embed │ │ (pgvector)           │
│ Haiku/    │ │ v3 (1024d)   │ │                      │
│ Sonnet    │ │              │ │                      │
└───────────┘ └──────────────┘ └──────────────────────┘
```

## RAG 파이프라인 흐름

```text
요청 수신 → 입력 검증
    │
    ▼
is_first_message?
    ├─(true)──▶ 시맨틱 캐시 탐색
    │               ├─(히트)──▶ 캐시 응답 반환 (토큰 소모=0)
    │               └─(미스)──┐
    └─(false)─────────────────┘
                              │
                              ▼
                      질문 재작성 (history 있으면 LLM 재작성)
                              │
                              ▼
                        의도 분류
              ┌───────────────┴───────────────┐
              │(학사규정)                      │(잡담)
              ▼                               ▼
        벡터 검색 (코사인 유사도)          잡담 LLM 응답
              │                               │
      ┌───────┴───────┐                       │
      │(≥ 임계값)      │(< 임계값)              │
      ▼               ▼                       │
  LLM 답변 생성   유사 질문 추천                │
  + 출처 포함     (Fallback)                   │
      │               │                       │
      └───────────────┴───────────────────────┘
                              │
                              ▼
                      SSE 스트리밍 응답
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 | 응답 형식 |
|---|---|---|---|
| `/api/chat` | POST | RAG 챗봇 대화 | SSE Stream |
| `/api/documents/embed` | POST | 문서 벡터화 및 적재 | JSON |
| `/api/documents/{doc_id}` | DELETE | 문서 벡터 삭제 | JSON |
| `/api/faq/analyze` | POST | FAQ 후보 자동 추출 | JSON |
| `/api/health` | GET | 헬스체크 | JSON |

### SSE 스트리밍 응답 시나리오 (POST /api/chat)

| 시나리오 | 조건 | 청크 시퀀스 |
|---|---|---|
| A. 정상 | 문서 검색 성공 | `meta(document)` → `text*` → `done(usage)` |
| B. 폴백 | 문서 검색 실패 | `fallback(suggested_questions)` → `done(usage)` |
| C. 캐시 히트 | 시맨틱 캐시 일치 | `meta(cache)` → `text` → `done(usage=0)` |
| D. 잡담 | 의도 분류 chitchat | `meta(chitchat)` → `text*` → `done(usage)` |
| 에러 | 스트리밍 중 장애 | `error(message)` |

### 에러 응답 형식 (통일)

```json
{
  "status": "error",
  "error_code": "BAD_REQUEST | VALIDATION_ERROR | SERVICE_UNAVAILABLE | INTERNAL_ERROR",
  "message": "사용자 친화적 에러 메시지"
}
```

| HTTP 코드 | error_code | 원인 |
|---|---|---|
| 400 | BAD_REQUEST | 필수 파라미터 누락, 비즈니스 검증 실패 |
| 422 | VALIDATION_ERROR | 타입 불일치, 제약조건 위반 |
| 503 | SERVICE_UNAVAILABLE | Bedrock / Supabase 장애 |
| 500 | INTERNAL_ERROR | 예상치 못한 내부 오류 |

## 프로젝트 구조

```text
app/
├── main.py                  # FastAPI 앱 엔트리포인트 (라우터, 에러 핸들러, lifespan)
├── config.py                # 환경 변수 로드 (Pydantic Settings)
├── startup.py               # 서버 시작 검증 (의존성 연결, 임베딩 차원 정합성)
├── exceptions.py            # 커스텀 예외 (ServiceUnavailableError)
│
├── api/                     # API 엔드포인트
│   ├── chat.py              # POST /api/chat (RAG 파이프라인 오케스트레이션)
│   ├── documents.py         # POST /api/documents/embed, DELETE /api/documents/{doc_id}
│   ├── faq.py               # POST /api/faq/analyze
│   ├── health.py            # GET /api/health
│   ├── error_handlers.py    # 글로벌 예외 핸들러 (400/422/503/500)
│   └── dependencies.py      # 의존성 주입 (클라이언트 싱글턴)
│
├── pipeline/                # RAG 파이프라인 모듈
│   ├── semantic_cache.py    # 시맨틱 캐싱 (answer_cache 기반)
│   ├── query_rewriter.py    # 질문 재작성 (히스토리 컨텍스트)
│   ├── intent_router.py     # 의도 분류 (academic / chitchat)
│   ├── vector_search.py     # 벡터 검색 + 임계값 폴백
│   └── llm_generator.py     # LLM 답변 생성 (스트리밍)
│
├── streaming/
│   └── sse.py               # SSE 스트리밍 모듈 (시나리오별 청크 생성)
│
├── clients/                 # 외부 서비스 클라이언트
│   ├── bedrock.py           # AWS Bedrock (LLM + 임베딩)
│   └── supabase_client.py   # Supabase Vector DB
│
├── models/
│   ├── schemas.py           # 요청/응답 Pydantic 모델
│   └── pipeline.py          # 내부 데이터 모델 (PipelineContext 등)
│
└── services/
    └── faq_analyzer.py      # FAQ 클러스터링 + 답변 초안 생성

tests/
├── test_bedrock.py          # Bedrock 클라이언트 단위 테스트
├── test_supabase_client.py  # Supabase 클라이언트 단위 테스트
├── test_semantic_cache.py   # 시맨틱 캐시 속성 테스트
├── test_query_rewriter.py   # 질문 재작성 속성 테스트
├── test_intent_router.py    # 의도 분류 속성 테스트
├── test_vector_search.py    # 벡터 검색 속성 테스트
├── test_llm_generator.py    # LLM 생성 + truncation 테스트
├── test_sse.py              # SSE 스트리밍 구조 테스트
├── test_chat.py             # 채팅 API + 파이프라인 통합 테스트
├── test_documents.py        # 문서 관리 API 테스트
├── test_faq.py              # FAQ 분석 테스트
├── test_health.py           # 헬스체크 테스트
├── test_error_handlers.py   # 글로벌 에러 핸들러 테스트
├── test_startup.py          # 서버 시작 검증 테스트
└── test_models.py           # Pydantic 모델 테스트
```

## 환경 변수

`.env.example`을 `.env`로 복사한 뒤 실제 값을 설정합니다.

```bash
cp .env.example .env
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SUPABASE_URL` | (필수) | Supabase 프로젝트 URL |
| `SUPABASE_KEY` | (필수) | Supabase service role key |
| `AWS_REGION` | `us-east-1` | AWS 리전 |
| `BEDROCK_LIGHT_MODEL_ID` | `anthropic.claude-3-haiku-20240307-v1:0` | 질문 재작성, 의도 분류, 잡담용 LLM |
| `BEDROCK_ANSWER_MODEL_ID` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | 학사규정 답변 생성용 LLM |
| `BEDROCK_EMBEDDING_MODEL_ID` | `cohere.embed-multilingual-v3` | 임베딩 모델 |
| `EMBEDDING_DIMENSION` | `1024` | 임베딩 벡터 차원 |
| `LLM_CONTEXT_WINDOW` | `200000` | LLM 컨텍스트 윈도우 (토큰) |
| `LLM_MAX_TOKENS` | `1024` | LLM 최대 출력 토큰 |
| `SIMILARITY_THRESHOLD` | `0.75` | 벡터 검색 유사도 임계값 |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | 시맨틱 캐시 유사도 임계값 |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.9` | confidence "high" 기준 |
| `CONFIDENCE_MEDIUM_THRESHOLD` | `0.8` | confidence "medium" 기준 |
| `CACHE_TTL_DAYS` | `90` | 캐시 만료 기간 (일) |
| `BEDROCK_LLM_TIMEOUT` | `30` | LLM 호출 타임아웃 (초) |
| `BEDROCK_EMBEDDING_TIMEOUT` | `15` | 임베딩 호출 타임아웃 (초) |
| `SUPABASE_TIMEOUT` | `10` | Supabase 호출 타임아웃 (초) |
| `CORS_ORIGINS` | `[]` (비활성) | CORS 허용 오리진 (예: `["http://localhost:3000"]`) |

## 로컬 실행

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -e ".[dev]"

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일에 실제 SUPABASE_URL, SUPABASE_KEY 등 입력

# 4. 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버 시작 시 자동으로 외부 의존성(Supabase, Bedrock) 연결과 임베딩 차원 정합성을 검증합니다. 검증 실패 시 서버가 시작되지 않습니다.

## Docker 실행

```bash
# 이미지 빌드
docker build -t kdd-ai .

# 컨테이너 실행
docker run -d --name kdd-ai -p 8000:8000 --env-file .env kdd-ai
```

### API 문서

서버 실행 후 아래 주소에서 OpenAPI 문서를 확인할 수 있습니다:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 특정 모듈만
pytest tests/test_chat.py -v

# 커버리지 포함
pytest tests/ -v --cov=app
```

모든 테스트는 외부 의존성(AWS Bedrock, Supabase)을 모킹하여 실행되므로 네트워크 연결 없이 동작합니다.

## 설계 원칙

- **무상태성**: 세션 상태를 서버에 저장하지 않으며, 매 요청마다 `history` 배열로 대화 문맥을 전달받음
- **비용 최적화**: 시맨틱 캐싱을 파이프라인 최선두에 배치하여 LLM 호출 최소화
- **할루시네이션 방지**: 검색된 문서 컨텍스트 내에서만 답변하도록 LLM 프롬프트 제한
- **모델 교체 용이성**: LLM/임베딩 모델 ID를 환경 변수로 관리하여 코드 변경 없이 교체 가능
- **방어적 설계**: 컨텍스트 윈도우 기반 동적 history truncation, 타임아웃, 리트라이
