"""설정 모듈 및 데이터 모델 단위 테스트."""

import os

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.pipeline import (
    PipelineContext,
    QuestionLog,
    QuestionLogMatch,
    SearchResult,
    SourceDoc,
    TokenUsage,
)
from app.models.schemas import (
    ChatRequest,
    DependencyStatus,
    DocumentChunk,
    DocumentMetadata,
    EmbedRequest,
    FAQAnalyzeRequest,
    HealthResponse,
    HistoryMessage,
)


# ── Config Tests ──


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")
        s = Settings(_env_file=None)
        assert s.BEDROCK_LLM_MODEL_ID == "anthropic.claude-3-haiku-20240307-v1:0"
        assert s.BEDROCK_EMBEDDING_MODEL_ID == "cohere.embed-multilingual-v3"
        assert s.EMBEDDING_DIMENSION == 1024
        assert s.LLM_CONTEXT_WINDOW == 200000
        assert s.LLM_MAX_TOKENS == 1024
        assert s.AWS_REGION == "us-east-1"
        assert s.SIMILARITY_THRESHOLD == 0.75
        assert s.CACHE_SIMILARITY_THRESHOLD == 0.95
        assert s.BEDROCK_LLM_TIMEOUT == 30
        assert s.BEDROCK_EMBEDDING_TIMEOUT == 15
        assert s.SUPABASE_TIMEOUT == 10

    def test_required_fields_missing(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://custom.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "custom-key")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "512")
        s = Settings(_env_file=None)
        assert s.SUPABASE_URL == "https://custom.supabase.co"
        assert s.SIMILARITY_THRESHOLD == 0.8
        assert s.EMBEDDING_DIMENSION == 512


# ── Schema Tests ──


class TestHistoryMessage:
    def test_valid(self):
        msg = HistoryMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            HistoryMessage(role="system", content="hello")

    def test_empty_content(self):
        with pytest.raises(ValidationError):
            HistoryMessage(role="user", content="")


class TestChatRequest:
    def test_valid_minimal(self):
        req = ChatRequest(
            message="질문입니다",
            session_id="sess-1",
            user_context="3학년",
            is_first_message=True,
        )
        assert req.history == []

    def test_valid_with_history(self):
        req = ChatRequest(
            message="후속 질문",
            session_id="sess-1",
            user_context="3학년",
            is_first_message=False,
            history=[HistoryMessage(role="user", content="이전 질문")],
        )
        assert len(req.history) == 1

    def test_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                message="",
                session_id="s",
                user_context="ctx",
                is_first_message=True,
            )

    def test_message_too_long(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                message="a" * 2001,
                session_id="s",
                user_context="ctx",
                is_first_message=True,
            )


class TestEmbedRequest:
    def test_valid(self):
        req = EmbedRequest(
            doc_id="doc-1",
            metadata=DocumentMetadata(
                doc_name="test.pdf",
                category="학사",
                enforcement_date="2026-03-01",
            ),
            chunks=[DocumentChunk(content="내용", page=1)],
        )
        assert req.doc_id == "doc-1"
        assert len(req.chunks) == 1

    def test_empty_chunks(self):
        with pytest.raises(ValidationError):
            EmbedRequest(
                doc_id="doc-1",
                metadata=DocumentMetadata(
                    doc_name="test.pdf",
                    category="학사",
                    enforcement_date="2026-03-01",
                ),
                chunks=[],
            )

    def test_invalid_page(self):
        with pytest.raises(ValidationError):
            DocumentChunk(content="내용", page=0)


class TestFAQAnalyzeRequest:
    def test_defaults(self):
        req = FAQAnalyzeRequest(questions=["질문1"])
        assert req.top_k == 5
        assert req.min_cluster_size == 2

    def test_empty_questions(self):
        with pytest.raises(ValidationError):
            FAQAnalyzeRequest(questions=[])

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            FAQAnalyzeRequest(questions=["q"], top_k=0)
        with pytest.raises(ValidationError):
            FAQAnalyzeRequest(questions=["q"], top_k=51)

    def test_blank_question_rejected(self):
        with pytest.raises(ValidationError):
            FAQAnalyzeRequest(questions=["유효한 질문", ""])
        with pytest.raises(ValidationError):
            FAQAnalyzeRequest(questions=["   "])


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(
            status="healthy",
            dependencies=DependencyStatus(
                vector_db="healthy",
                bedrock_llm="healthy",
                bedrock_embedding="healthy",
            ),
        )
        assert resp.status == "healthy"

    def test_unhealthy(self):
        resp = HealthResponse(
            status="unhealthy",
            dependencies=DependencyStatus(
                vector_db="healthy",
                bedrock_llm="unhealthy",
                bedrock_embedding="healthy",
            ),
        )
        assert resp.dependencies.bedrock_llm == "unhealthy"


# ── Pipeline Model Tests ──


class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total_tokens == 0


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(
            id=1,
            doc_id="doc-1",
            content="내용",
            metadata={"doc_name": "test.pdf", "page": 1},
            similarity_score=0.85,
        )
        assert r.similarity_score == 0.85


class TestSourceDoc:
    def test_creation(self):
        s = SourceDoc(doc_name="test.pdf", page=3)
        assert s.page == 3


class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext(original_question="테스트 질문")
        assert ctx.original_question == "테스트 질문"
        assert ctx.rewritten_question is None
        assert ctx.intent is None
        assert ctx.cache_hit is False
        assert ctx.cached_answer is None
        assert ctx.cached_sources == []
        assert ctx.source_docs == []
        assert ctx.token_usage.total_tokens == 0
        assert ctx.history == []
        assert ctx.search_results is None

    def test_mutable_defaults_isolation(self):
        ctx1 = PipelineContext(original_question="q1")
        ctx2 = PipelineContext(original_question="q2")
        ctx1.source_docs.append(SourceDoc(doc_name="a.pdf", page=1))
        assert len(ctx2.source_docs) == 0


class TestQuestionLog:
    def test_academic(self):
        log = QuestionLog(
            question="휴학 기간",
            embedding=[0.1] * 1024,
            answer="최대 4년입니다.",
            intent="academic",
            source_doc_ids=["doc-1"],
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )
        assert log.intent == "academic"
        assert log.embedding is not None

    def test_chitchat(self):
        log = QuestionLog(
            question="안녕",
            embedding=None,
            answer=None,
            intent="chitchat",
            source_doc_ids=[],
            sources=[],
        )
        assert log.embedding is None
        assert log.answer is None


class TestQuestionLogMatch:
    def test_creation(self):
        m = QuestionLogMatch(
            question="휴학 기간",
            answer="최대 4년",
            similarity_score=0.97,
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )
        assert m.similarity_score == 0.97
