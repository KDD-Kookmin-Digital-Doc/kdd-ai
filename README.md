# 학사규정 RAG AI 챗봇 서버

학사규정 RAG(검색 증강 생성) AI 챗봇 서버. FastAPI 기반 독립 마이크로서비스로, 메인 백엔드(Spring)와 완전히 분리되어 무상태(Stateless)로 동작합니다.

## 기술 스택

| 분류 | 기술 |
|---|---|
| 런타임 | Python 3.11+, FastAPI, uvicorn |
| LLM (답변 생성) | AWS Bedrock — Claude 3.5 Sonnet v2 (`claude-3-5-sonnet-20241022-v2:0`) |
| LLM (재작성/의도분류/잡담) | AWS Bedrock — Claude 3 Haiku (`claude-3-haiku-20240307-v1:0`) |
| 임베딩 | AWS Bedrock — Cohere Embed Multilingual v4 (1024차원, 서울) |
| 리랭커 | AWS Bedrock — Cohere Rerank 3.5 (`cohere.rerank-v3-5:0`, 도쿄) |
| 벡터 DB | AWS RDS PostgreSQL 16 + pgvector (asyncpg) |
| 스트리밍 | SSE (Server-Sent Events) |
| 관측성 | session_id 자동 트레이싱(`ContextVar`) + CloudWatch Logs (awslogs driver) |
| 테스트 | pytest, pytest-asyncio, Hypothesis (속성 기반 테스트) |

## 아키텍처

```mermaid
flowchart TB
    Client["프론트엔드 / 메인 백엔드 (Spring)"]

    Client -->|"POST /api/chat<br/>POST /api/documents/embed<br/>DELETE /api/documents/{doc_id}<br/>POST /api/faq/analyze<br/>GET /api/health"| Router

    subgraph AIServer["AI Server (FastAPI)"]
        Router["API Router"]
        Router --> RAG["RAG Pipeline"]
        Router --> Doc["Document Service"]
        Router --> FAQ["FAQ Service"]
        Router --> Health["Health Service"]
    end

    subgraph AWS["AWS"]
        Bedrock["Bedrock Claude<br/>Haiku · Sonnet"]
        Embed["Bedrock Cohere<br/>Embed v4 (1024d)<br/>Seoul"]
        Rerank["Bedrock Cohere<br/>Rerank 3.5<br/>Tokyo"]
        RDS[("RDS PostgreSQL<br/>+ pgvector")]
    end

    RAG --> Bedrock
    RAG --> Embed
    RAG --> Rerank
    RAG --> RDS
    Doc --> Embed
    Doc --> RDS
    FAQ --> Bedrock
    FAQ --> Embed
    FAQ --> RDS
    Health --> RDS
```

## RAG 파이프라인 흐름

```mermaid
flowchart TD
    Start(["요청 수신"]) --> Validate["입력 검증"]
    Validate --> FirstMsg{"is_first_message?"}

    FirstMsg -->|"true"| Cache["시맨틱 캐시 탐색"]
    Cache -->|"히트"| CacheHit["캐시 응답 반환<br/>(토큰 소모 = 0)"]
    Cache -->|"미스"| Classify

    FirstMsg -->|"false"| Classify["의도 분류 (history-aware)<br/>rewrite 이전에 수행 —<br/>history 톤이 잡담을 학사로<br/>비트는 회귀 차단"]

    Classify -->|"학사규정"| Rewrite["질문 재작성<br/>(history 있으면 LLM 재작성)"]
    Classify -->|"잡담"| ChitChat["잡담 LLM 응답"]

    Rewrite --> VecSearch["벡터 검색<br/>(코사인 유사도, top_k=30)"]
    VecSearch --> RerankStage["Cohere Rerank 3.5<br/>(30 → 5건 재정렬, 도쿄)"]
    RerankStage --> Threshold{"최고 score<br/>≥ 임계값?"}
    Threshold -->|"예"| LLMGen["LLM 답변 생성<br/>출처 + 인용 마커 포함"]
    Threshold -->|"아니오"| Fallback["유사 질문 추천<br/>(Fallback)"]

    CacheHit --> SSE["SSE 스트리밍 응답"]
    ChitChat --> SSE
    LLMGen --> SSE
    Fallback --> SSE
```

`RERANK_ENABLED=false` 일 때는 임베딩 검색 결과 상위 `VECTOR_SEARCH_TOP_K` 만 사용 (Rerank 단계 우회, graceful degradation 경로도 동일).

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

**인용 마커 (시나리오 A · C 본문)**: 답변 본문의 각 주장 끝에 `{{N}}` 형식의 마커가 박힙니다(`N` = 1-based 출처 인덱스). 클라이언트는 `meta` 의 `sources` 배열 순서(`sources[N-1]`)와 매핑하여 인용 UI를 렌더링합니다. 마커는 `_buffer_by_word` whitespace 누적 패턴으로 단일 `text` 청크 내 atomic 전송이 보장됩니다.

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
| 503 | SERVICE_UNAVAILABLE | Bedrock / RDS PostgreSQL 장애 |
| 500 | INTERNAL_ERROR | 예상치 못한 내부 오류 |

## 프로젝트 구조

```text
app/
├── main.py                  # FastAPI 앱 엔트리포인트 (라우터, 에러 핸들러, lifespan)
├── config.py                # 환경 변수 로드 (Pydantic Settings)
├── startup.py               # 서버 시작 검증 (의존성 연결, 임베딩 차원 정합성)
├── exceptions.py            # 커스텀 예외 (ServiceUnavailableError)
├── logging_context.py       # session_id ContextVar + LoggerFilter (요청별 자동 트레이싱)
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
│   ├── vector_search.py     # 벡터 검색 + Cohere Rerank + 임계값 폴백
│   ├── llm_generator.py     # LLM 답변 생성 (스트리밍, 인용 마커 포함)
│   └── _messages.py         # Bedrock messages 빌더 헬퍼 (make_user_message 등)
│
├── streaming/
│   └── sse.py               # SSE 스트리밍 모듈 (시나리오별 청크 생성)
│
├── clients/                 # 외부 서비스 클라이언트
│   ├── bedrock.py           # AWS Bedrock (LLM + 임베딩 + Rerank, 도쿄 Rerank 클라이언트 분리)
│   └── postgres_client.py   # RDS PostgreSQL Vector DB (asyncpg + pgvector)
│
├── models/
│   ├── schemas.py           # 요청/응답 Pydantic 모델
│   └── pipeline.py          # 내부 데이터 모델 (PipelineContext 등)
│
└── services/
    └── faq_analyzer.py      # FAQ 클러스터링 + 답변 초안 생성 (asyncio.gather + Semaphore)

tests/
├── test_bedrock.py          # Bedrock 클라이언트 단위 테스트
├── test_postgres_client.py  # PostgreSQL (asyncpg) 클라이언트 단위 테스트 (race 가드 회귀 포함)
├── test_semantic_cache.py   # 시맨틱 캐시 속성 테스트
├── test_query_rewriter.py   # 질문 재작성 속성 테스트
├── test_intent_router.py    # 의도 분류 속성 테스트
├── test_vector_search.py    # 벡터 검색 + Rerank graceful degradation 테스트
├── test_llm_generator.py    # LLM 생성 + truncation + 인용 마커 회귀 테스트
├── test_sse.py              # SSE 스트리밍 구조 + 인용 마커 atomic flush 테스트
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

### 데이터베이스 (RDS PostgreSQL)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DATABASE_URL` | (필수) | RDS PostgreSQL DSN (예: `postgresql://user:pass@host:5432/db?sslmode=require`) |
| `POSTGRES_POOL_MIN_SIZE` | `2` | asyncpg connection pool 최소 크기 |
| `POSTGRES_POOL_MAX_SIZE` | `10` | asyncpg connection pool 최대 크기 |
| `POSTGRES_TIMEOUT` | `10` | asyncpg 쿼리/풀 초기화 타임아웃 (초) |

### AWS Bedrock — 모델 ID

| 변수 | 기본값 | 설명 |
|---|---|---|
| `AWS_REGION` | `ap-northeast-2` | AWS 기본 리전 (LLM + 임베딩, Rerank 는 별도) |
| `BEDROCK_LIGHT_MODEL_ID` | `apac.anthropic.claude-3-haiku-20240307-v1:0` | 질문 재작성 / 의도 분류 / 잡담 LLM |
| `BEDROCK_ANSWER_MODEL_ID` | `apac.anthropic.claude-3-5-sonnet-20241022-v2:0` | 학사규정 답변 생성 LLM |
| `BEDROCK_EMBEDDING_MODEL_ID` | `global.cohere.embed-v4:0` | 임베딩 모델 |
| `EMBEDDING_DIMENSION` | `1024` | 임베딩 벡터 차원 |

### AWS Bedrock — Rerank (Cohere Rerank 3.5, 도쿄)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `RERANK_ENABLED` | `True` | 두 단계 retrieval 활성 여부 (False 시 임베딩 검색만) |
| `RERANK_REGION` | `ap-northeast-1` | Rerank API 리전 (Cohere Rerank 3.5 single-region only) |
| `RERANK_MODEL_ID` | `cohere.rerank-v3-5:0` | Rerank 모델 ID |
| `RETRIEVE_TOP_K_RERANK` | `30` | Stage 1 (임베딩 검색) 후보 수 |
| `RERANK_TOP_N` | `5` | Stage 2 (리랭크) 최종 컨텍스트 개수 (≤ `RETRIEVE_TOP_K_RERANK`) |
| `BEDROCK_RERANK_TIMEOUT` | `15` | Rerank API 타임아웃 (초) |

### RAG 동작 파라미터

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.75` | 벡터 검색 유사도 임계값 (운영 `.env` 와 다를 수 있음 — 이슈 #45) |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | 시맨틱 캐시 유사도 임계값 |
| `FALLBACK_SIMILARITY_THRESHOLD` | `0.5` | 폴백(유사 질문 추천) floor 임계값 |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.9` | confidence "high" 기준 |
| `CONFIDENCE_MEDIUM_THRESHOLD` | `0.8` | confidence "medium" 기준 |
| `VECTOR_SEARCH_TOP_K` | `5` | Rerank 미사용 시 또는 fallback 시 검색 결과 수 |
| `FALLBACK_SUGGESTED_COUNT` | `3` | 폴백 시 추천 질문 개수 |
| `INTENT_HISTORY_TURNS` | `6` | 의도 분류에 첨부할 history 메시지 개수 |
| `INTENT_HISTORY_CHARS_PER_TURN` | `200` | history 메시지 1개당 최대 글자 수 |
| `REWRITE_MAX_TOKENS` | `512` | 질문 재작성 LLM 최대 출력 토큰 |
| `INTENT_MAX_TOKENS` | `16` | 의도 분류 LLM 최대 출력 토큰 |
| `CHITCHAT_MAX_TOKENS` | `256` | 잡담 LLM 최대 출력 토큰 |
| `FAQ_LLM_MAX_TOKENS` | `512` | FAQ 답변 초안 LLM 최대 출력 토큰 |
| `LLM_CONTEXT_WINDOW` | `200000` | LLM 컨텍스트 윈도우 (토큰) |
| `LLM_MAX_TOKENS` | `1024` | 답변 LLM 최대 출력 토큰 |

### 동시성 · 캐시 · 타임아웃

| 변수 | 기본값 | 설명 |
|---|---|---|
| `FAQ_CONCURRENCY` | `5` | FAQ 답변 초안 생성 시 동시 Bedrock 호출 수 (throttling 가드) |
| `EMBED_BATCH_SIZE` | `48` | 문서 적재 시 한 번의 embed 호출에 묶을 청크 수 |
| `CACHE_TTL_DAYS` | `90` | 시맨틱 캐시 TTL (일) |
| `BEDROCK_LLM_TIMEOUT` | `30` | LLM 호출 타임아웃 (초) |
| `BEDROCK_EMBEDDING_TIMEOUT` | `30` | 임베딩 read 타임아웃 (초) |
| `BEDROCK_EMBEDDING_CONNECT_TIMEOUT` | `10` | 임베딩 connect 타임아웃 (초) |
| `BEDROCK_MAX_RETRIES` | `2` | Bedrock SDK 자동 재시도 횟수 |

### 운영

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LOG_LEVEL` | `INFO` | 로깅 레벨 (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) |
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
# .env 파일에 실제 DATABASE_URL, AWS 자격증명 등 입력

# 4. 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버 시작 시 자동으로 외부 의존성(RDS PostgreSQL, Bedrock) 연결과 임베딩 차원 정합성을 검증합니다. 검증 실패 시 서버가 시작되지 않습니다.

## Docker 실행

```bash
# 환경 파일 준비
cp .env.example .env
# .env에 DATABASE_URL, AWS 자격증명 등 필수값 설정

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

모든 테스트는 외부 의존성(AWS Bedrock, RDS PostgreSQL)을 모킹하여 실행되므로 네트워크 연결 없이 동작합니다. `patch.dict(os.environ, ..., clear=True)` 패턴으로 셸/CI 환경변수와 격리되어 있어 결과가 재현 가능합니다.

## 설계 원칙

- **무상태성**: 세션 상태를 서버에 저장하지 않으며, 매 요청마다 `history` 배열로 대화 문맥을 전달받음
- **비용 최적화**: 시맨틱 캐싱을 파이프라인 최선두에 배치하여 LLM 호출 최소화 + 동일 요청 내 임베딩 중복 호출 제거(`PipelineContext.question_embedding` 보관)
- **할루시네이션 방지**: 검색된 문서 컨텍스트 내에서만 답변하도록 LLM 프롬프트 제한, 답변 본문 각 주장 끝에 `{{N}}` 인용 마커 강제로 출처 가시성 확보
- **두 단계 retrieval**: 1024차원 cosine 임베딩 후보 30건 → Cohere Rerank 3.5 로 5건 재정렬. Rerank 실패/타임아웃 시 임베딩 결과로 graceful degradation
- **모델 교체 용이성**: LLM/임베딩/Rerank 모델 ID를 환경 변수로 관리하여 코드 변경 없이 교체 가능
- **방어적 설계**: 컨텍스트 윈도우 기반 동적 history truncation, 타임아웃, 리트라이, `asyncio.Lock` 기반 lazy pool 초기화 race 가드
- **관측성**: `session_id` ContextVar 자동 트레이싱으로 모든 로그에 세션 식별자 첨부, CloudWatch Logs Insights 에서 단일 세션 흐름 추적 가능
