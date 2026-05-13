"""벡터 검색 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import PipelineContext, SearchResult, SourceDoc
from app.pipeline.vector_search import search_documents


# ── 헬퍼 ──


def _create_settings(**overrides: str) -> Settings:
    env = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
        **overrides,
    }
    # clear=True: CI/로컬 셸에 남아있는 환경변수가 테스트로 새는 것을 차단
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)


def _create_bedrock(embedding: list[float] | None = None) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.embed_texts.return_value = [embedding or [0.1] * 1024]
    # D1: RERANK_ENABLED=True 기본이라 rerank 호출이 발생.
    # 기존 테스트 동등성 유지 위해 identity 동작(순서/개수 보존, score=1.0)으로 mock.
    async def _identity_rerank(query, documents, top_n):
        return [(i, 1.0) for i in range(min(len(documents), top_n))]
    bedrock.rerank.side_effect = _identity_rerank
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

    async def test_fallback_passes_threshold_to_supabase(self):
        """이슈 #46: fallback 호출 시 settings.FALLBACK_SIMILARITY_THRESHOLD 가 전달된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=[], similar_questions=["q"])

        ctx = _make_context()
        await search_documents(ctx, bedrock, supabase, settings)

        call_kwargs = supabase.search_similar_questions.call_args.kwargs
        assert call_kwargs["threshold"] == settings.FALLBACK_SIMILARITY_THRESHOLD
        assert call_kwargs["top_k"] == settings.FALLBACK_SUGGESTED_COUNT

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
        ctx.embedded_question_input_type = "search_query"

        await search_documents(ctx, bedrock, supabase, settings)

        bedrock.embed_texts.assert_not_called()
        # 검색에는 캐시된 임베딩이 그대로 사용됨
        call_kwargs = supabase.search_documents.call_args
        assert call_kwargs.kwargs["embedding"] == cached_embedding

    async def test_creates_new_embedding_when_input_type_mismatch(self):
        """input_type 이 search_query 가 아니면 새로 호출 (silent degradation 가드)."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        ctx = _make_context(question="질문", rewritten=None)
        ctx.question_embedding = [0.42] * 1024
        ctx.embedded_question_text = "질문"
        ctx.embedded_question_input_type = "search_document"  # ← 잘못 set된 케이스

        await search_documents(ctx, bedrock, supabase, settings)

        # input_type mismatch → 재사용 거부, 새로 호출
        bedrock.embed_texts.assert_called_once_with(
            ["질문"], input_type="search_query"
        )

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


# ── D1: Rerank (Task 13) ──


def _make_results(n: int) -> list[SearchResult]:
    """Rerank 테스트용 다중 SearchResult — chunk/doc id 분리."""
    return [
        SearchResult(
            chunk_id=100 + i,
            doc_id=10 + i,
            content=f"문서{i}",
            metadata={"doc_name": f"doc{i}.pdf", "page": i + 1},
            similarity_score=0.9 - 0.01 * i,
        )
        for i in range(n)
    ]


class TestRerank:
    """D1 (Task 13): two-stage retrieval (임베딩 stage 1 → 도쿄 rerank stage 2)."""

    async def test_disabled_uses_vector_search_top_k(self):
        """RERANK_ENABLED=False → 검색 top_k=VECTOR_SEARCH_TOP_K (회귀 잠금)."""
        settings = _create_settings(
            RERANK_ENABLED="False", VECTOR_SEARCH_TOP_K="5"
        )
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=_make_results(5))

        await search_documents(_make_context(), bedrock, supabase, settings)

        call_kwargs = supabase.search_documents.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    async def test_disabled_does_not_call_rerank(self):
        """RERANK_ENABLED=False → bedrock.rerank 미호출."""
        settings = _create_settings(RERANK_ENABLED="False")
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=_make_results(3))

        await search_documents(_make_context(), bedrock, supabase, settings)

        bedrock.rerank.assert_not_called()

    async def test_enabled_uses_retrieve_top_k_rerank(self):
        """RERANK_ENABLED=True → 검색 top_k=RETRIEVE_TOP_K_RERANK."""
        settings = _create_settings(
            RERANK_ENABLED="True", RETRIEVE_TOP_K_RERANK="30"
        )
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=_make_results(30))

        await search_documents(_make_context(), bedrock, supabase, settings)

        call_kwargs = supabase.search_documents.call_args.kwargs
        assert call_kwargs["top_k"] == 30

    async def test_enabled_calls_rerank_with_documents(self):
        """rerank 호출 시 documents=[r.content for r in results], top_n=RERANK_TOP_N."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="3",
            RERANK_TOP_N="2",
        )
        bedrock = _create_bedrock()
        supabase = _create_supabase(search_results=_make_results(3))

        await search_documents(
            _make_context(question="휴학 절차"), bedrock, supabase, settings
        )

        bedrock.rerank.assert_called_once()
        call_kwargs = bedrock.rerank.call_args.kwargs
        assert call_kwargs["query"] == "휴학 절차"
        assert call_kwargs["documents"] == ["문서0", "문서1", "문서2"]
        assert call_kwargs["top_n"] == 2

    async def test_reorders_and_slices_results(self):
        """rerank 결과 순서로 재정렬 + RERANK_TOP_N 슬라이스."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="4",
            RERANK_TOP_N="2",
        )
        bedrock = _create_bedrock()
        # rerank 가 index [2, 0] 으로 재정렬 (원본 idx 2 → rank 1, idx 0 → rank 2)
        bedrock.rerank.side_effect = None
        bedrock.rerank.return_value = [(2, 0.95), (0, 0.80)]
        supabase = _create_supabase(search_results=_make_results(4))

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # 2건만 남고 순서는 rerank 가 정한 것
        assert len(result.search_results) == 2
        assert result.search_results[0].chunk_id == 102  # 원본 idx 2
        assert result.search_results[1].chunk_id == 100  # 원본 idx 0

    async def test_rerank_score_populated_similarity_preserved(self):
        """rerank_score 채워지고 similarity_score (코사인) 의미는 보존."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="3",
            RERANK_TOP_N="2",
        )
        bedrock = _create_bedrock()
        bedrock.rerank.side_effect = None
        bedrock.rerank.return_value = [(1, 0.97), (0, 0.45)]
        supabase = _create_supabase(search_results=_make_results(3))

        result = await search_documents(_make_context(), bedrock, supabase, settings)

        # rerank_score 가 새 필드에 채워짐
        assert result.search_results[0].rerank_score == 0.97
        assert result.search_results[1].rerank_score == 0.45
        # similarity_score (코사인) 의미 보존 — 원본 SearchResult 값 그대로
        # 원본 idx 1: similarity=0.89, 원본 idx 0: similarity=0.9
        assert result.search_results[0].similarity_score == 0.89
        assert result.search_results[1].similarity_score == 0.90

    async def test_failure_falls_back_to_embedding_top_n(self):
        """rerank 예외 시 임베딩 상위 RERANK_TOP_N 개로 graceful degradation."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="5",
            RERANK_TOP_N="3",
        )
        bedrock = _create_bedrock()
        bedrock.rerank.side_effect = RuntimeError("도쿄 throttle 시뮬레이션")
        supabase = _create_supabase(search_results=_make_results(5))

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # 임베딩 결과 상위 3개로 fallback — 검색 자체는 실패시키지 않음
        assert len(result.search_results) == 3
        assert result.search_results[0].chunk_id == 100  # 임베딩 순서 유지
        assert result.search_results[1].chunk_id == 101
        assert result.search_results[2].chunk_id == 102
        # rerank_score 는 채워지지 않음 (fallback 경로)
        assert result.search_results[0].rerank_score is None
        # source_docs 도 정상 구성
        assert len(result.source_docs) == 3

    async def test_failure_logs_warning(self, caplog):
        """rerank 실패 시 logger.warning 발생 (CloudWatch 모니터링용)."""
        import logging

        settings = _create_settings(RERANK_ENABLED="True")
        bedrock = _create_bedrock()
        bedrock.rerank.side_effect = RuntimeError("rerank fail")
        supabase = _create_supabase(search_results=_make_results(3))

        with caplog.at_level(logging.WARNING, logger="app.pipeline.vector_search"):
            await search_documents(_make_context(), bedrock, supabase, settings)

        assert any(
            "Rerank 호출 실패" in rec.message for rec in caplog.records
        )

    async def test_skipped_when_no_results(self):
        """results=[] 시 rerank 미호출 (불필요한 도쿄 호출 차단), fallback 분기 진입."""
        settings = _create_settings(RERANK_ENABLED="True")
        bedrock = _create_bedrock()
        supabase = _create_supabase(
            search_results=[], similar_questions=["유사 질문"]
        )

        result = await search_documents(_make_context(), bedrock, supabase, settings)

        bedrock.rerank.assert_not_called()
        assert result.suggested_questions == ["유사 질문"]

    async def test_out_of_range_index_falls_back(self):
        """rerank 응답이 범위 밖 인덱스(>= len) 포함 → 임베딩 fallback (IndexError 차단)."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="5",
            RERANK_TOP_N="3",
        )
        bedrock = _create_bedrock()
        # 첫 항목은 유효, 두 번째에 out-of-range 99 → 전체 fallback
        bedrock.rerank.side_effect = None
        bedrock.rerank.return_value = [(0, 0.9), (99, 0.5)]
        supabase = _create_supabase(search_results=_make_results(5))

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # 임베딩 상위 3건 순서대로 fallback — 챗 요청 자체는 살아있음 (IndexError 차단)
        # 2-pass 검증으로 부분 mutation 차단 — stale rerank_score 외부 노출 방지
        assert [r.chunk_id for r in result.search_results] == [100, 101, 102]
        assert all(r.rerank_score is None for r in result.search_results)

    async def test_negative_index_falls_back(self):
        """rerank 응답이 음수 인덱스 포함 → 임베딩 fallback (negative wrap-around 차단)."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="5",
            RERANK_TOP_N="3",
        )
        bedrock = _create_bedrock()
        bedrock.rerank.side_effect = None
        bedrock.rerank.return_value = [(-1, 0.9)]
        supabase = _create_supabase(search_results=_make_results(5))

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # Python list 음수 인덱스는 wrap-around 되므로 명시적으로 차단
        assert [r.chunk_id for r in result.search_results] == [100, 101, 102]
        # 음수 idx 가 첫 항목이라 mutation 발생 전 차단 — rerank_score 깨끗
        assert all(r.rerank_score is None for r in result.search_results)

    async def test_empty_rerank_response_falls_back(self):
        """Cohere 빈 응답 + stage 1 결과 존재 → fallback path 오라우팅 차단."""
        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="5",
            RERANK_TOP_N="3",
        )
        bedrock = _create_bedrock()
        bedrock.rerank.side_effect = None
        bedrock.rerank.return_value = []  # Cohere 빈 응답
        supabase = _create_supabase(
            search_results=_make_results(5),
            similar_questions=["오라우팅되면 안 됨"],
        )

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # silent quality regression 방지 — 임베딩 결과 살리고 suggested_questions 분기로 가지 않음
        assert len(result.search_results) == 3
        assert result.suggested_questions == []
        supabase.search_similar_questions.assert_not_called()


# ── Citation 마커: 인덱스 매핑 invariant ──
# delightful-greeting-dove.md plan 의 핵심 컨트랙트:
# LLM 프롬프트의 [문서 N] 인덱스 == FE meta.sources[N-1] == ctx.source_docs[N-1]


class TestCitationIndexInvariant:
    """`{{N}}` 마커 도입의 전제 — search_results 순서 == source_docs 순서.

    `vector_search.py:71-72` 에서 동일 `results` 리스트로 1:1 set 되는
    invariant 위에 plan 의 매핑 컨트랙트가 서 있다. 미래에 누군가
    source_docs 생성 정렬을 손대면 마커가 깨지므로 회귀를 명시적으로
    잠근다.
    """

    async def test_source_docs_index_matches_search_results_index(self):
        """search_results[i] 의 doc_name 이 source_docs[i].doc_name 과 항상 일치."""
        settings = _create_settings(RERANK_ENABLED="False")
        bedrock = _create_bedrock()
        results = [
            _make_search_result(doc_id=10, doc_name="A.pdf", page=1, chunk_id=100),
            _make_search_result(doc_id=20, doc_name="B.pdf", page=2, chunk_id=200),
            _make_search_result(doc_id=30, doc_name="C.pdf", page=3, chunk_id=300),
        ]
        supabase = _create_supabase(search_results=results)

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # search_results.metadata["doc_name"] 과 source_docs.doc_name 이 1:1 매핑
        for i, (sr, sd) in enumerate(
            zip(result.search_results, result.source_docs, strict=True)
        ):
            assert sr.metadata["doc_name"] == sd.doc_name, (
                f"인덱스 {i}: search_results['doc_name']={sr.metadata['doc_name']} "
                f"!= source_docs.doc_name={sd.doc_name} — {{N}} 마커 invariant 깨짐"
            )
            assert sr.doc_id == sd.doc_id
            assert sr.chunk_id == sd.chunk_id

    async def test_invariant_preserved_after_rerank_reorder(self):
        """rerank 가 순서를 바꿔도 search_results 와 source_docs 는 같은 새 순서 공유."""
        from app.pipeline.llm_generator import _build_doc_context

        settings = _create_settings(
            RERANK_ENABLED="True",
            RETRIEVE_TOP_K_RERANK="3",
            RERANK_TOP_N="3",
        )
        bedrock = _create_bedrock()
        # rerank 가 [2, 0, 1] 순서로 재정렬
        bedrock.rerank.side_effect = None
        bedrock.rerank.return_value = [(2, 0.95), (0, 0.85), (1, 0.75)]
        supabase = _create_supabase(search_results=_make_results(3))

        ctx = _make_context()
        result = await search_documents(ctx, bedrock, supabase, settings)

        # 재정렬 후에도 1:1 invariant 유지
        for i, (sr, sd) in enumerate(
            zip(result.search_results, result.source_docs, strict=True)
        ):
            assert sr.metadata["doc_name"] == sd.doc_name
            assert sr.doc_id == sd.doc_id
            assert sr.chunk_id == sd.chunk_id

        # 그리고 LLM 이 보는 [문서 N] 라벨 doc_name 이 source_docs[N-1].doc_name 과 일치
        # (이게 곧 {{N}} 마커가 FE meta.sources[N-1] 을 가리킨다는 plan 의 컨트랙트)
        doc_context = _build_doc_context(result)
        for n, sd in enumerate(result.source_docs, 1):
            label = f"[문서 {n}] {sd.doc_name}"
            assert label in doc_context, (
                f"[문서 {n}] 라벨 doc_name 이 source_docs[{n-1}].doc_name 과 불일치 "
                f"— {{N}} 마커 매핑 깨짐. 기대: {label!r}"
            )
