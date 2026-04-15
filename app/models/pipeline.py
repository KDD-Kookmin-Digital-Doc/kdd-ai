"""파이프라인 내부 데이터 모델 정의 (dataclass 기반)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TokenUsage:
    """토큰 사용량."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class SearchResult:
    """벡터 검색 결과."""

    chunk_id: int
    doc_id: int
    content: str
    metadata: dict
    similarity_score: float


@dataclass
class SourceDoc:
    """출처 정보."""

    doc_id: int
    chunk_id: int
    doc_name: str
    page: int


@dataclass
class AnswerCache:
    """답변 캐시 레코드. 학사규정 질문의 정상 답변 완료 시에만 저장."""

    question: str
    embedding: list[float]
    answer: str
    source_doc_ids: list[int]
    sources: list[dict]


@dataclass
class CacheMatch:
    """시맨틱 캐시 매칭 결과."""

    question: str
    answer: str
    similarity_score: float
    sources: list[dict]


@dataclass
class PipelineContext:
    """RAG 파이프라인 단계 간 전달되는 컨텍스트."""

    original_question: str
    rewritten_question: str | None = None
    intent: Literal["academic", "chitchat"] | None = None
    search_results: list[SearchResult] | None = None
    cache_hit: bool = False
    cached_answer: str | None = None
    cached_sources: list[SourceDoc] = field(default_factory=list)
    user_context: str = ""
    history: list = field(default_factory=list)
    source_docs: list[SourceDoc] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
