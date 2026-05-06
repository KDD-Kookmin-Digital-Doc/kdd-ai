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

    def accumulate(self, other: "TokenUsage") -> None:
        """다른 TokenUsage 의 값을 in-place 로 합산한다 (PR-R2)."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


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

    @classmethod
    def from_cache_dict(cls, d: dict) -> "SourceDoc":
        """answer_cache.sources JSONB 의 dict 한 개를 SourceDoc 으로 변환 (PR-R3)."""
        return cls(
            doc_id=d["doc_id"],
            chunk_id=d["chunk_id"],
            doc_name=d["doc_name"],
            page=d["page"],
        )

    @classmethod
    def from_search_result(cls, r: "SearchResult") -> "SourceDoc":
        """벡터 검색 SearchResult 한 개를 SourceDoc 으로 변환 (PR-R3)."""
        return cls(
            doc_id=r.doc_id,
            chunk_id=r.chunk_id,
            doc_name=r.metadata.get("doc_name", ""),
            page=r.metadata.get("page", 0),
        )


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
    """RAG 파이프라인 단계 간 전달되는 컨텍스트.

    임베딩 재사용 contract (Task 16):
    - ``question_embedding`` 은 ``embedded_question_text`` 를
      ``embedded_question_input_type`` 으로 임베딩한 결과이다.
    - ``semantic_cache`` 가 임베딩 직후 세 필드를 set하고, 하위 단계
      (``vector_search``, ``_save_answer_cache``) 가 **텍스트 + input_type 둘 다
      일치할 때만** 재사용한다. input_type 가드는 미래에 누군가
      ``semantic_cache`` 의 input_type을 변경했을 때 silent quality
      degradation을 막는 안전장치이다.
    - 하위 단계는 세 필드를 **읽기만** 한다(덮어쓰지 않는다).
    """

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
    question_embedding: list[float] | None = None
    embedded_question_text: str | None = None
    embedded_question_input_type: str | None = None
