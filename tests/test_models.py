"""설정 모듈 및 데이터 모델 단위 테스트."""

import os

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.pipeline import (
    AnswerCache,
    CacheMatch,
    PipelineContext,
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
        assert s.BEDROCK_LIGHT_MODEL_ID == "anthropic.claude-3-haiku-20240307-v1:0"
        assert s.BEDROCK_ANSWER_MODEL_ID == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert s.BEDROCK_EMBEDDING_MODEL_ID == "cohere.embed-multilingual-v3"
        assert s.EMBEDDING_DIMENSION == 1024
        assert s.LLM_CONTEXT_WINDOW == 200000
        assert s.LLM_MAX_TOKENS == 1024
        assert s.AWS_REGION == "us-east-1"
        assert s.SIMILARITY_THRESHOLD == 0.75
        assert s.CACHE_SIMILARITY_THRESHOLD == 0.95
        assert s.BEDROCK_LLM_TIMEOUT == 30
        assert s.BEDROCK_EMBEDDING_TIMEOUT == 30
        assert s.BEDROCK_EMBEDDING_CONNECT_TIMEOUT == 10
        assert s.EMBED_BATCH_SIZE == 48
        assert s.INTENT_HISTORY_TURNS == 6
        assert s.INTENT_HISTORY_CHARS_PER_TURN == 200
        assert s.SUPABASE_TIMEOUT == 10
        assert s.CACHE_TTL_DAYS == 90
        assert s.LOG_LEVEL == "INFO"

    def test_log_level_normalized_to_upper(self, monkeypatch):
        """LOG_LEVEL 은 대소문자 무관하게 받아들여 대문자로 정규화된다."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        s = Settings(_env_file=None)
        assert s.LOG_LEVEL == "DEBUG"

    def test_log_level_invalid_rejected(self, monkeypatch):
        """알 수 없는 LOG_LEVEL 은 ValidationError."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

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

    def test_embed_batch_size_must_be_positive(self, monkeypatch):
        """EMBED_BATCH_SIZE=0 또는 음수면 Settings 생성이 실패한다 (fail-fast)."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")

        for bad_value in ("0", "-1"):
            monkeypatch.setenv("EMBED_BATCH_SIZE", bad_value)
            with pytest.raises(ValidationError):
                Settings(_env_file=None)


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

    def test_blank_content_rejected(self):
        with pytest.raises(ValidationError):
            HistoryMessage(role="user", content="   ")


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

    def test_blank_fields_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                message="   ",
                session_id="s",
                user_context="ctx",
                is_first_message=True,
            )
        with pytest.raises(ValidationError):
            ChatRequest(
                message="질문",
                session_id="  ",
                user_context="ctx",
                is_first_message=True,
            )


class TestEmbedRequest:
    def test_valid(self):
        req = EmbedRequest(
            doc_id=1,
            metadata=DocumentMetadata(
                doc_name="test.pdf",
                category="학사",
                enforcement_date="2026-03-01",
            ),
            chunks=[DocumentChunk(chunk_id=1, content="내용", page=1)],
        )
        assert req.doc_id == 1
        assert len(req.chunks) == 1

    def test_empty_chunks(self):
        with pytest.raises(ValidationError):
            EmbedRequest(
                doc_id=1,
                metadata=DocumentMetadata(
                    doc_name="test.pdf",
                    category="학사",
                    enforcement_date="2026-03-01",
                ),
                chunks=[],
            )

    def test_invalid_page(self):
        with pytest.raises(ValidationError):
            DocumentChunk(chunk_id=1, content="내용", page=0)

    def test_invalid_chunk_id(self):
        with pytest.raises(ValidationError):
            DocumentChunk(chunk_id=0, content="내용", page=1)
        with pytest.raises(ValidationError):
            DocumentChunk(chunk_id=-1, content="내용", page=1)

    def test_duplicate_chunk_ids_rejected(self):
        with pytest.raises(ValidationError):
            EmbedRequest(
                doc_id=1,
                metadata=DocumentMetadata(
                    doc_name="test.pdf",
                    category="학사",
                    enforcement_date="2026-03-01",
                ),
                chunks=[
                    DocumentChunk(chunk_id=1, content="내용1", page=1),
                    DocumentChunk(chunk_id=1, content="내용2", page=2),
                ],
            )

    def test_invalid_doc_id_rejected(self):
        with pytest.raises(ValidationError):
            EmbedRequest(
                doc_id=0,
                metadata=DocumentMetadata(
                    doc_name="test.pdf",
                    category="학사",
                    enforcement_date="2026-03-01",
                ),
                chunks=[DocumentChunk(chunk_id=1, content="내용", page=1)],
            )

    def test_invalid_enforcement_date(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(
                doc_name="test.pdf",
                category="학사",
                enforcement_date="not-a-date",
            )

    def test_enforcement_date_parsed(self):
        from datetime import date
        meta = DocumentMetadata(
            doc_name="test.pdf",
            category="학사",
            enforcement_date="2026-03-01",
        )
        assert meta.enforcement_date == date(2026, 3, 1)

    def test_enforcement_date_optional_none(self):
        meta = DocumentMetadata(
            doc_name="test.pdf",
            category="학사",
            enforcement_date=None,
        )
        assert meta.enforcement_date is None

    def test_enforcement_date_default_none(self):
        meta = DocumentMetadata(
            doc_name="test.pdf",
            category="학사",
        )
        assert meta.enforcement_date is None


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
            chunk_id=1,
            doc_id=1,
            content="내용",
            metadata={"doc_name": "test.pdf", "page": 1},
            similarity_score=0.85,
        )
        assert r.chunk_id == 1
        assert r.similarity_score == 0.85


class TestSourceDoc:
    def test_creation(self):
        s = SourceDoc(doc_id=1, chunk_id=42, doc_name="test.pdf", page=3)
        assert s.chunk_id == 42
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
        ctx1.source_docs.append(SourceDoc(doc_id=10, chunk_id=1, doc_name="a.pdf", page=1))
        assert len(ctx2.source_docs) == 0


class TestAnswerCache:
    def test_creation(self):
        cache = AnswerCache(
            question="휴학 기간",
            embedding=[0.1] * 1024,
            answer="최대 4년입니다.",
            source_doc_ids=[1],
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )
        assert cache.embedding is not None
        assert cache.answer == "최대 4년입니다."


class TestCacheMatch:
    def test_creation(self):
        m = CacheMatch(
            question="휴학 기간",
            answer="최대 4년",
            similarity_score=0.97,
            sources=[{"doc_name": "학사요람.pdf", "page": 45}],
        )
        assert m.similarity_score == 0.97
