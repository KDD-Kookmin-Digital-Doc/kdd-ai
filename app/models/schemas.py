"""요청/응답 Pydantic 모델 정의."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _check_not_blank(v: str, field_name: str) -> str:
    """문자열이 공백만으로 이루어져 있지 않은지 검증."""
    if not v.strip():
        raise ValueError(f"{field_name}이(가) 비어있거나 공백만 포함합니다.")
    return v


# ── Chat 관련 ──


class HistoryMessage(BaseModel):
    """대화 내역 메시지."""

    role: Literal["user", "assistant"] = Field(
        ..., description="메시지 발화자. `user` 또는 `assistant`.", examples=["user"]
    )
    content: str = Field(
        ...,
        min_length=1,
        description="메시지 본문 (공백만 있는 문자열 불가).",
        examples=["휴학 신청은 어떻게 하나요?"],
    )

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        return _check_not_blank(v, "content")


class ChatRequest(BaseModel):
    """POST /api/chat 요청 모델."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="사용자 질문. 1~2000자, 공백만 있는 문자열 불가.",
        examples=["휴학 신청은 어떻게 하나요?"],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="대화 세션 식별자. 클라이언트에서 발급한 UUID 권장.",
        examples=["sess-2026-04-13-abc123"],
    )
    user_context: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="사용자 상황 정보 (학부/학과/학년 등). 답변 톤·범위 판단에 사용.",
        examples=["컴퓨터공학과 3학년 학부생"],
    )
    is_first_message: bool = Field(
        ...,
        description="세션 내 첫 메시지 여부. `true` 인 경우에만 시맨틱 캐시 조회/저장이 수행됨.",
        examples=[True],
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="직전 대화 히스토리. 질문 재작성(문맥화) 단계에서 참조됨.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "휴학은 몇 학기까지 가능한가요?",
                    "session_id": "sess-2026-04-13-abc123",
                    "user_context": "컴퓨터공학과 3학년 학부생",
                    "is_first_message": True,
                    "history": [],
                }
            ]
        }
    )

    @field_validator("message", "session_id", "user_context")
    @classmethod
    def fields_not_blank(cls, v: str, info) -> str:
        return _check_not_blank(v, info.field_name)


# ── Document Embed 관련 ──


class DocumentMetadata(BaseModel):
    """문서 메타데이터."""

    doc_name: str = Field(
        ..., min_length=1, description="문서 이름.", examples=["학사규정_2024"]
    )
    category: str = Field(
        ..., min_length=1, description="문서 카테고리.", examples=["학사"]
    )
    enforcement_date: date = Field(
        ..., description="해당 규정 시행일 (ISO-8601).", examples=["2024-03-01"]
    )

    @field_validator("doc_name", "category")
    @classmethod
    def fields_not_blank(cls, v: str, info) -> str:
        return _check_not_blank(v, info.field_name)


class DocumentChunk(BaseModel):
    """문서 청크."""

    chunk_id: int = Field(
        ...,
        ge=1,
        description="청크 고유 번호 (BE가 전역 유일하게 부여, `documents` 테이블 PK).",
        examples=[1001],
    )
    content: str = Field(
        ...,
        min_length=1,
        description="청크 본문 텍스트.",
        examples=["제1조(목적) 이 규정은 학사운영에 관한 사항을 정함을 목적으로 한다."],
    )
    page: int = Field(
        ..., ge=1, description="원본 문서 내 페이지 번호.", examples=[1]
    )

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        return _check_not_blank(v, "content")


class EmbedRequest(BaseModel):
    """POST /api/documents/embed 요청 모델."""

    doc_id: int = Field(
        ...,
        ge=1,
        description="문서 식별자 (BE의 Document PK). 동일 값 재적재 시 기존 청크는 삭제 후 재삽입됨.",
        examples=[20240001],
    )
    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(
        ...,
        min_length=1,
        description="적재할 청크 목록. 최소 1개 이상, chunk_id 는 유일해야 함.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "doc_id": 20240001,
                    "metadata": {
                        "doc_name": "학사규정_2024",
                        "category": "학사",
                        "enforcement_date": "2024-03-01",
                    },
                    "chunks": [
                        {
                            "chunk_id": 1001,
                            "content": "제1조(목적) 이 규정은 학사운영에 관한 사항을 정함을 목적으로 한다.",
                            "page": 1,
                        },
                        {
                            "chunk_id": 1002,
                            "content": "제2조(적용범위) 이 규정은 본교 학부생에게 적용한다.",
                            "page": 1,
                        },
                    ],
                }
            ]
        }
    )

    @field_validator("chunks")
    @classmethod
    def chunk_ids_must_be_unique(
        cls, v: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        seen: set[int] = set()
        dupes: set[int] = set()
        for c in v:
            if c.chunk_id in seen:
                dupes.add(c.chunk_id)
            seen.add(c.chunk_id)
        if dupes:
            raise ValueError(
                f"중복된 chunk_id가 있습니다: {sorted(dupes)}"
            )
        return v


# ── FAQ 분석 관련 ──


class FAQAnalyzeRequest(BaseModel):
    """POST /api/faq/analyze 요청 모델."""

    questions: list[str] = Field(
        ...,
        min_length=1,
        description="분석 대상 질문 목록. 최소 1개 이상, 각 질문은 공백만으로 이뤄질 수 없음.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="반환할 상위 FAQ 후보 개수. 1~50.",
    )
    min_cluster_size: int = Field(
        default=2,
        ge=2,
        description="FAQ 클러스터로 인정할 최소 질문 수. 2 이상.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "questions": [
                        "휴학 신청 방법 알려주세요",
                        "휴학은 어떻게 하나요",
                        "장학금 신청 기한이 언제인가요",
                        "장학금 언제까지 신청해야 하죠",
                    ],
                    "top_k": 5,
                    "min_cluster_size": 2,
                }
            ]
        }
    )

    @field_validator("questions")
    @classmethod
    def validate_questions_not_blank(cls, v: list[str]) -> list[str]:
        """각 질문이 빈 문자열이나 공백만 있는 문자열이 아닌지 검증."""
        for i, q in enumerate(v):
            if not q.strip():
                raise ValueError(f"questions[{i}]이(가) 비어있거나 공백만 포함합니다.")
        return v


# ── Health 관련 ──


class DependencyStatus(BaseModel):
    """외부 의존성 상태."""

    vector_db: Literal["healthy", "unhealthy"]
    bedrock_llm: Literal["healthy", "unhealthy"]
    bedrock_embedding: Literal["healthy", "unhealthy"]


class HealthResponse(BaseModel):
    """GET /api/health 응답 모델."""

    status: Literal["healthy", "unhealthy"]
    dependencies: DependencyStatus


# ── Error 관련 ──


class ErrorResponse(BaseModel):
    """통일된 에러 응답 모델. OpenAPI 스키마 문서화용."""

    status: str = Field(default="error", description="항상 'error'")
    error_code: str = Field(..., description="에러 코드 (BAD_REQUEST, VALIDATION_ERROR 등)")
    message: str = Field(..., description="사용자 친화적 에러 메시지")
