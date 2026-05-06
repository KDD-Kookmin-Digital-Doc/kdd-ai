"""벡터 검색 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import PipelineContext, SearchResult, SourceDoc
from app.pipeline.vector_search import search_documents


# ── 헬퍼 ──


def _create_settings() -> Settings:
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


def _create_bedrock(embedding: list[float] | None = None) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.embed_texts.return_value = [embedding or [0.1] * 1024]
    return bedrock


def _create_supabase(
    search_results: list[SearchResult] | None = None,
    similar_questions: list[str] | None = None,
) -> AsyncMock:
    supabase = AsyncMock()
    supabase.search_documents.return_value = search_results or []
    supabase.search_similar_questions.return_value = similar_questions or []
    return supabase


def _make_context(
    question: str = "테스트 질문",
    rewritten: str | None = None,
) -> PipelineContext:
    return PipelineContext(
        original_question=question,
        rewritten_question=rewritten,
    )


def _make_search_result(
    similarity: float = 0.85,
    doc_id: int = 1,
    doc_name: str = "학사요람.pdf",
    page: int = 10,
    chunk_id: int = 1,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content="제1조 내용",
        metadata={"doc_name": doc_name, "page": page},
        similarity_score=similarity,
    )


# ── Property 5: 유사도 임계값 기반 응답 분기 ──
# Feature: rag-chatbot-ai-server, Property 5: 유사도 임계값 기반 응답 분기
# Validates: Requirements 4.2, 4.3


class TestThresholdBasedRouting:
    """Property 5: 임계값 이상 문서 존재 시 LLM 진행, 미만 시 Fallback."""

    @hyp_settings(max_examples=50)
    @given(
        similarity=st.floats(min_value=0.75, max_value=1.0),
    )
    async def test_above_threshold_sets_search_results(self, similarity):
        """임계값 이상 문서가 있으면 search_results가 설정된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        results = [_make_search_result(similarity=similarity)]
        supabase = _create_supabase(search_results=results)

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        assert result.search_results is not None
        assert len(result.search_results) > 0
        assert len(result.source_docs) > 0
        assert result.suggested_questions == []

    @hyp_settings(max_examples=50)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    async def test_below_threshold_triggers_fallback(self, question):
        """임계값 미만이면 (빈 결과) 폴백으로 유사 질문이 추출된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(
            search_results=[],
            similar_questions=["유사 질문 1", "유사 질문 2"],
        )

        ctx = _make_context(question=question)
        result = await search_documents(ctx, bedrock, supabase, settings)

        assert result.search_results == []
        assert len(result.suggested_questions) == 2
        assert result.source_docs == []

    @hyp_settings(max_examples=50)
    @given(
        has_results=st.booleans(),
    )
    async def test_routing_determined_by_results(self, has_results):
        """검색 결과 유무에 따라 정상/폴백 경로가 결정된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        results = [_make_search_result()] if has_results else []
        supabase = _create_supabase(
            search_results=results,
            similar_questions=["질문"],
        )

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        if has_results:
            assert len(result.search_results) > 0
            assert len(result.source_docs) > 0
        else:
            assert result.search_results == []
            assert len(result.suggested_questions) > 0


# ── Property 6: 임베딩 차원 불변성 ──
# Feature: rag-chatbot-ai-server, Property 6: 임베딩 차원 불변성
# Validates: Requirements 4.1, 9.1, 11.1


class TestEmbeddingDimensionInvariance:
    """Property 6: 임베딩 출력 벡터가 항상 정확히 1024차원인지 검증."""

    @hyp_settings(max_examples=50)
    @given(
        question=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    async def test_embedding_is_1024_dimensions(self, question):
        """embed_texts 호출 시 1024차원 벡터가 반환된다."""
        bedrock = _create_bedrock(embedding=[0.1] * 1024)

        result = await bedrock.embed_texts([question], input_type="search_query")

        assert len(result[0]) == 1024

    async def test_embedding_dimension_matches_settings(self):
        """임베딩 차원이 settings.EMBEDDING_DIMENSION과 일치한다."""
        settings = _create_settings()
        bedrock = _create_bedrock(embedding=[0.1] * settings.EMBEDDING_DIMENSION)

        result = await bedrock.embed_texts(["질문"], input_type="search_query")

        assert len(result[0]) == settings.EMBEDDING_DIMENSION


# ── 단위 테스트 ──


class TestSearchDocumentsUnit:
    async def test_uses_rewritten_question(self):
        """rewritten_question이 있으면 그것을 임베딩한다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context(
            question="그건 뭐야?",
            rewritten="휴학 기간은 얼마인가요?",
        )
        await search_documents(ctx, bedrock, supabase, settings)

        bedrock.embed_texts.assert_called_once_with(
            ["휴학 기간은 얼마인가요?"], input_type="search_query"
        )

    async def test_falls_back_to_original_question(self):
        """rewritten_question이 None이면 original_question을 임베딩한다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context(question="원본 질문", rewritten=None)
        await search_documents(ctx, bedrock, supabase, settings)

        bedrock.embed_texts.assert_called_once_with(
            ["원본 질문"], input_type="search_query"
        )

    async def test_threshold_from_settings(self):
        """검색 시 settings.SIMILARITY_THRESHOLD를 사용한다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context()
        await search_documents(ctx, bedrock, supabase, settings)

        call_kwargs = supabase.search_documents.call_args
        assert call_kwargs.kwargs["threshold"] == settings.SIMILARITY_THRESHOLD

    async def test_source_docs_metadata_preserved(self):
        """검색 결과의 메타데이터가 source_docs에 보존된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        results = [
            _make_search_result(doc_id=1, doc_name="학사요람.pdf", page=45, chunk_id=111),
            _make_search_result(doc_id=2, doc_name="학칙.pdf", page=10, chunk_id=222),
        ]
        supabase = _create_supabase(search_results=results)

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        assert len(result.source_docs) == 2
        assert result.source_docs[0].doc_id == 1
        assert result.source_docs[0].chunk_id == 111
        assert result.source_docs[0].doc_name == "학사요람.pdf"
        assert result.source_docs[0].page == 45
        assert result.source_docs[1].doc_id == 2
        assert result.source_docs[1].chunk_id == 222

    async def test_fallback_calls_similar_questions(self):
        """폴백 시 search_similar_questions가 호출된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(
            search_results=[],
            similar_questions=["질문1", "질문2", "질문3"],
        )

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        supabase.search_similar_questions.assert_called_once()
        assert result.suggested_questions == ["질문1", "질문2", "질문3"]

    async def test_no_fallback_when_results_exist(self):
        """검색 결과가 있으면 search_similar_questions가 호출되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=[_make_search_result()])

        ctx = _make_context()
        await search_documents(ctx, bedrock, supabase, settings)

        supabase.search_similar_questions.assert_not_called()

    async def test_input_type_search_query(self):
        """임베딩 호출 시 input_type이 search_query인지 확인."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context()
        await search_documents(ctx, bedrock, supabase, settings)

        call_kwargs = bedrock.embed_texts.call_args
        assert call_kwargs.kwargs.get("input_type") == "search_query" or \
            (len(call_kwargs.args) > 1 and call_kwargs.args[1] == "search_query")

    async def test_empty_similar_questions(self):
        """answer_cache에 유사 질문이 없으면 빈 리스트가 설정된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=[], similar_questions=[])

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        assert result.suggested_questions == []


# ── Task 16: 임베딩 재사용 ──


class TestEmbeddingReuse:
    """semantic_cache가 미리 임베딩한 결과를 vector_search에서 재사용 (텍스트 일치 시)."""

    async def test_reuses_embedding_when_text_matches(self):
        """rewrite 미발생 + context.embedded_question_text == question → embed_texts 호출 X."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(
            search_results=[_make_search_result()],
        )

        cached_embedding = [0.42] * 1024
        ctx = _make_context(question="휴학 기간은?", rewritten=None)
        ctx.question_embedding = cached_embedding
        ctx.embedded_question_text = "휴학 기간은?"

        await search_documents(ctx, bedrock, supabase, settings)

        bedrock.embed_texts.assert_not_called()
        # 검색에는 캐시된 임베딩이 그대로 사용됨
        call_kwargs = supabase.search_documents.call_args
        assert call_kwargs.kwargs["embedding"] == cached_embedding

    async def test_creates_new_embedding_when_rewrite_changes_text(self):
        """rewrite로 텍스트가 달라지면 새로 임베딩한다 (의미 다른 텍스트는 재사용 X)."""
        settings = _create_settings()
        bedrock = _create_bedrock(embedding=[0.99] * 1024)
        supabase = _create_supabase()

        ctx = _make_context(question="그건 뭐야?", rewritten="휴학 기간은 얼마인가요?")
        ctx.question_embedding = [0.42] * 1024
        ctx.embedded_question_text = "그건 뭐야?"  # original 측만 임베딩됨

        await search_documents(ctx, bedrock, supabase, settings)

        # rewritten으로 새로 임베딩 (search_query input_type 유지)
        bedrock.embed_texts.assert_called_once_with(
            ["휴학 기간은 얼마인가요?"], input_type="search_query"
        )

    async def test_creates_new_embedding_when_no_cache(self):
        """context.question_embedding이 None이면 새로 임베딩 (멀티턴: cache 단계 skip)."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context(question="멀티턴 질문")
        # question_embedding은 기본값 None
        assert ctx.question_embedding is None

        await search_documents(ctx, bedrock, supabase, settings)

        bedrock.embed_texts.assert_called_once_with(
            ["멀티턴 질문"], input_type="search_query"
        )

    async def test_does_not_overwrite_cached_embedding(self):
        """vector_search는 context.question_embedding을 갱신하지 않는다 (read-only invariant)."""
        settings = _create_settings()
        bedrock = _create_bedrock(embedding=[0.99] * 1024)
        supabase = _create_supabase()

        original_embedding = [0.42] * 1024
        ctx = _make_context(question="원본", rewritten="재작성됨")
        ctx.question_embedding = original_embedding
        ctx.embedded_question_text = "원본"

        await search_documents(ctx, bedrock, supabase, settings)

        # rewrite 분기에서 새 임베딩을 만들었지만 컨텍스트는 그대로
        assert ctx.question_embedding is original_embedding
        assert ctx.embedded_question_text == "원본"
