"""요청/응답 Pydantic 모델 정의."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _check_not_blank(v: str, field_name: str) -> str:
    """문자열이 공백만으로 이루어져 있지 않은지 검증."""
    if not v.strip():
        raise ValueError(f"{field_name}이(가) 비어있거나 공백만 포함합니다.")
    return v


# ── Chat 관련 ──


class HistoryMessage(BaseModel):
    """대화 내역 메시지."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        return _check_not_blank(v, "content")


class ChatRequest(BaseModel):
    """POST /api/chat 요청 모델."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    user_context: str = Field(..., min_length=1, max_length=500)
    is_first_message: bool
    history: list[HistoryMessage] = Field(default_factory=list)

    @field_validator("message", "session_id", "user_context")
    @classmethod
    def fields_not_blank(cls, v: str, info) -> str:
        return _check_not_blank(v, info.field_name)


# ── Document Embed 관련 ──


class DocumentMetadata(BaseModel):
    """문서 메타데이터."""

    doc_name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    enforcement_date: date

    @field_validator("doc_name", "category")
    @classmethod
    def fields_not_blank(cls, v: str, info) -> str:
        return _check_not_blank(v, info.field_name)


class DocumentChunk(BaseModel):
    """문서 청크."""

    chunk_id: int = Field(..., ge=1)
    content: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        return _check_not_blank(v, "content")


class EmbedRequest(BaseModel):
    """POST /api/documents/embed 요청 모델."""

    doc_id: str = Field(..., min_length=1)
    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(..., min_length=1)

    @field_validator("doc_id")
    @classmethod
    def doc_id_not_blank(cls, v: str) -> str:
        return _check_not_blank(v, "doc_id")

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

    questions: list[str] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    min_cluster_size: int = Field(default=2, ge=2)

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
