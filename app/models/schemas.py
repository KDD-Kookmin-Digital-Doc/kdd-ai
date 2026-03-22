"""요청/응답 Pydantic 모델 정의."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Chat 관련 ──


class HistoryMessage(BaseModel):
    """대화 내역 메시지."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """POST /api/chat 요청 모델."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    user_context: str = Field(..., min_length=1)
    is_first_message: bool
    history: list[HistoryMessage] = Field(default_factory=list)


# ── Document Embed 관련 ──


class DocumentMetadata(BaseModel):
    """문서 메타데이터."""

    doc_name: str
    category: str
    enforcement_date: str


class DocumentChunk(BaseModel):
    """문서 청크."""

    content: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)


class EmbedRequest(BaseModel):
    """POST /api/documents/embed 요청 모델."""

    doc_id: str = Field(..., min_length=1)
    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(..., min_length=1)


# ── FAQ 분석 관련 ──


class FAQAnalyzeRequest(BaseModel):
    """POST /api/faq/analyze 요청 모델."""

    questions: list[str] = Field(..., min_length=1)

    @field_validator("questions")
    @classmethod
    def validate_questions_not_blank(cls, v: list[str]) -> list[str]:
        """각 질문이 빈 문자열이나 공백만 있는 문자열이 아닌지 검증."""
        for i, q in enumerate(v):
            if not q.strip():
                raise ValueError(f"questions[{i}]이(가) 비어있거나 공백만 포함합니다.")
        return v
    top_k: int = Field(default=5, ge=1, le=50)
    min_cluster_size: int = Field(default=2, ge=2)


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
