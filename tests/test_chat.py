"""POST /api/chat 엔드포인트 테스트. 파이프라인 오케스트레이션 + 캐시 저장 + 에러 전파."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import (
    AnswerCache,
    CacheMatch,
    PipelineContext,
    SearchResult,
    SourceDoc,
    TokenUsage,
)
from app.models.schemas import ChatRequest
from app.api.chat import _run_pipeline, _save_answer_cache, _streaming_wrapper


# ── 헬퍼 ──


def _create_settings() -> Settings:
    with patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
    }):
        return Settings(_env_file=None)


def _create_bedrock() -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.embed_texts.return_value = [[0.1] * 1024]
    bedrock.invoke_llm.return_value = (
        "academic",
        TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )
    bedrock.last_stream_usage = TokenUsage()
    return bedrock


def _create_supabase() -> AsyncMock:
    supabase = AsyncMock()
    supabase.search_answer_cache.return_value = None
    supabase.search_documents.return_value = []
    supabase.search_similar_questions.return_value = []
    supabase.insert_answer_cache.return_value = None
    return supabase


def _make_chat_request(
    message: str = "휴학 기간은?",
    is_first_message: bool = True,
    user_context: str = "3학년",
    history: list[dict] | None = None,
) -> ChatRequest:
    return ChatRequest(
        message=message,
        session_id="sess-1",
        user_context=user_context,
        is_first_message=is_first_message,
        history=history or [],
    )


def _make_search_result(
    doc_id: int = 1,
    similarity: float = 0.85,
    chunk_id: int = 1,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content="제1조 내용",
        metadata={"doc_name": "학사요람.pdf", "page": 10},
        similarity_score=similarity,
    )


def _make_cache_match() -> CacheMatch:
    return CacheMatch(
        question="캐시된 질문",
        answer="캐시된 답변",
        similarity_score=0.97,
        sources=[{"doc_id": 1, "chunk_id": 1, "doc_name": "학사요람.pdf", "page": 45}],
    )


def _make_http_request(disconnected: bool = False) -> AsyncMock:
    req = AsyncMock()
    req.is_disconnected.return_value = disconnected
    return req


# ── 파이프라인 오케스트레이션 테스트 ──


class TestRunPipeline:
    async def test_cache_hit_skips_remaining_steps(self):
        """캐시 히트 시 재작성/의도분류/벡터검색을 건너뛴다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.search_answer_cache.return_value = _make_cache_match()

        request = _make_chat_request(is_first_message=True)

        with patch("app.api.chat.rewrite_query") as mock_rewrite, \
             patch("app.api.chat.classify_intent") as mock_intent, \
             patch("app.api.chat.search_documents") as mock_search:

            context = await _run_pipeline(request, bedrock, supabase, settings)

            assert context.cache_hit is True
            mock_rewrite.assert_not_called()
            mock_intent.assert_not_called()
            mock_search.assert_not_called()

    async def test_chitchat_skips_rewrite_and_vector_search(self):
        """잡담 분류 시 재작성과 벡터 검색을 모두 건너뛴다 (L안 핵심).

        rewrite를 chitchat 경로에서도 호출하면 history 톤이 잡담을 학사 톤으로
        비트는 회귀가 발생할 수 있어, classify가 chitchat을 내면 rewrite/search
        둘 다 skip한다.
        """
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        async def _mock_classify(ctx, br, st):
            ctx.intent = "chitchat"
            return ctx

        request = _make_chat_request(is_first_message=False)

        with patch("app.api.chat.rewrite_query") as mock_rewrite, \
             patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
             patch("app.api.chat.search_documents") as mock_search:

            context = await _run_pipeline(request, bedrock, supabase, settings)

            assert context.intent == "chitchat"
            mock_rewrite.assert_not_called()
            mock_search.assert_not_called()

    async def test_academic_pipeline_call_order(self):
        """학사 경로 호출 순서: classify → rewrite → search (L안 핵심).

        classify_intent가 rewrite보다 먼저 실행돼야 history 톤이 잡담을
        학사 톤으로 비트는 회귀를 차단할 수 있다.
        """
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        call_order: list[str] = []

        async def _mock_classify(ctx, br, st):
            call_order.append("classify")
            ctx.intent = "academic"
            return ctx

        async def _mock_rewrite(ctx, br):
            call_order.append("rewrite")
            ctx.rewritten_question = ctx.original_question
            return ctx

        async def _mock_search(ctx, br, sb, st):
            call_order.append("search")
            ctx.search_results = [_make_search_result()]
            return ctx

        request = _make_chat_request(is_first_message=False)

        with patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
             patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
             patch("app.api.chat.search_documents", side_effect=_mock_search):

            await _run_pipeline(request, bedrock, supabase, settings)

        assert call_order == ["classify", "rewrite", "search"]

    async def test_full_academic_pipeline(self):
        """학사규정 경로: 캐시미스 → 재작성 → 의도분류 → 벡터검색 전체 실행."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        async def _mock_rewrite(ctx, br):
            ctx.rewritten_question = ctx.original_question
            return ctx

        async def _mock_classify(ctx, br, st):
            ctx.intent = "academic"
            return ctx

        async def _mock_search(ctx, br, sb, st):
            ctx.search_results = [_make_search_result()]
            return ctx

        request = _make_chat_request(is_first_message=False)

        with patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
             patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
             patch("app.api.chat.search_documents", side_effect=_mock_search):

            context = await _run_pipeline(request, bedrock, supabase, settings)

            assert context.intent == "academic"
            assert len(context.search_results) == 1

    async def test_history_converted_to_dicts(self):
        """history가 HistoryMessage → dict로 변환된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        request = _make_chat_request(
            is_first_message=False,
            history=[
                {"role": "user", "content": "이전 질문"},
                {"role": "assistant", "content": "이전 답변"},
            ],
        )

        contexts_captured = []

        async def _mock_rewrite(ctx, br):
            contexts_captured.append(ctx)
            ctx.rewritten_question = ctx.original_question
            return ctx

        async def _mock_classify(ctx, br, st):
            ctx.intent = "academic"
            return ctx

        async def _mock_search(ctx, br, sb, st):
            ctx.search_results = []
            return ctx

        with patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
             patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
             patch("app.api.chat.search_documents", side_effect=_mock_search):

            await _run_pipeline(request, bedrock, supabase, settings)

            ctx = contexts_captured[0]
            assert len(ctx.history) == 2
            assert ctx.history[0]["role"] == "user"
            assert ctx.history[0]["content"] == "이전 질문"


# ── Property 15: 답변 캐시 저장 완전성 ──
# Validates: Requirements 12.1


class TestAnswerCacheCompleteness:
    """Property 15: 학사규정 답변 후 answer_cache에 모든 필드 포함 검증."""

    async def test_cache_saved_with_all_fields(self):
        """학사규정 정상 답변 시 question, embedding, answer, source_doc_ids, sources 모두 저장."""
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        context = PipelineContext(
            original_question="휴학 기간은?",
            intent="academic",
            search_results=[_make_search_result(doc_id=1)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="학사요람.pdf", page=10)],
        )

        await _save_answer_cache(context, ["최대 ", "4년입니다."], bedrock, supabase)

        supabase.insert_answer_cache.assert_called_once()
        cache: AnswerCache = supabase.insert_answer_cache.call_args[0][0]
        assert cache.question == "휴학 기간은?"
        assert len(cache.embedding) == 1024
        assert cache.answer == "최대 4년입니다."
        assert 1 in cache.source_doc_ids
        assert cache.sources[0]["doc_name"] == "학사요람.pdf"
        assert cache.sources[0]["chunk_id"] == 1

    @hyp_settings(max_examples=30)
    @given(
        question=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        answer=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    async def test_cache_fields_never_empty(self, question, answer):
        """캐시 저장 시 필수 필드가 비어있지 않다."""
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        context = PipelineContext(
            original_question=question,
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        await _save_answer_cache(context, [answer], bedrock, supabase)

        cache: AnswerCache = supabase.insert_answer_cache.call_args[0][0]
        assert cache.question
        assert cache.embedding
        assert cache.answer
        assert cache.source_doc_ids
        assert cache.sources


# ── Property 16: 잡담/Fallback 캐시 미저장 ──
# Validates: Requirements 12.2, 12.4


class TestNoCacheForChitchatAndFallback:
    """Property 16: 잡담/Fallback 시 answer_cache에 저장하지 않음."""

    async def test_chitchat_no_cache(self):
        """잡담 경로에서는 캐시가 저장되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        http_request = _make_http_request()

        context = PipelineContext(
            original_question="안녕하세요",
            intent="chitchat",
        )

        with patch("app.api.chat.stream_sse_response") as mock_sse:
            async def _mock_stream(*args, **kwargs):
                yield 'data: {"type": "meta", "subtype": "chitchat"}\n\n'
                yield 'data: {"type": "done", "usage": {}}\n\n'
            mock_sse.return_value = _mock_stream()

            chunks = []
            async for chunk in _streaming_wrapper(
                context, bedrock, supabase, settings, http_request, is_first_message=True
            ):
                chunks.append(chunk)

        supabase.insert_answer_cache.assert_not_called()

    async def test_fallback_no_cache(self):
        """폴백 경로에서는 캐시가 저장되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        http_request = _make_http_request()

        context = PipelineContext(
            original_question="알 수 없는 질문",
            intent="academic",
            search_results=[],
            suggested_questions=["유사 질문"],
        )

        with patch("app.api.chat.stream_sse_response") as mock_sse:
            async def _mock_stream(*args, **kwargs):
                yield 'data: {"type": "fallback", "message": "없음"}\n\n'
                yield 'data: {"type": "done", "usage": {}}\n\n'
            mock_sse.return_value = _mock_stream()

            async for _ in _streaming_wrapper(
                context, bedrock, supabase, settings, http_request, is_first_message=True
            ):
                pass

        supabase.insert_answer_cache.assert_not_called()

    async def test_cache_hit_no_cache(self):
        """캐시 히트 경로에서는 새로운 캐시를 저장하지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        http_request = _make_http_request()

        context = PipelineContext(
            original_question="질문",
            cache_hit=True,
            cached_answer="캐시 답변",
            cached_sources=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        with patch("app.api.chat.stream_sse_response") as mock_sse:
            async def _mock_stream(*args, **kwargs):
                yield 'data: {"type": "meta", "subtype": "cache"}\n\n'
                yield 'data: {"type": "text", "content": "캐시 답변"}\n\n'
                yield 'data: {"type": "done", "usage": {}}\n\n'
            mock_sse.return_value = _mock_stream()

            async for _ in _streaming_wrapper(
                context, bedrock, supabase, settings, http_request, is_first_message=True
            ):
                pass

        supabase.insert_answer_cache.assert_not_called()


# ── 에러 전파 테스트 ──
# Validates: Requirements 7.3, 7.4, 7.5, 12.3


class TestErrorPropagation:
    async def test_intent_classification_failure_raises(self):
        """의도 분류 단계 장애 시 예외가 전파된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        request = _make_chat_request(is_first_message=False)

        async def _mock_rewrite(ctx, br):
            ctx.rewritten_question = ctx.original_question
            return ctx

        with patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
             patch("app.api.chat.classify_intent") as mock_ci:
            mock_ci.side_effect = RuntimeError("Bedrock 장애")

            with pytest.raises(RuntimeError, match="Bedrock 장애"):
                await _run_pipeline(request, bedrock, supabase, settings)

    async def test_cache_search_graceful_degradation(self):
        """시맨틱 캐시 탐색 실패 시 파이프라인이 계속 진행된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()

        # check_cache 내부에서 예외를 잡고 graceful degradation
        bedrock.embed_texts.side_effect = RuntimeError("Supabase 장애")

        request = _make_chat_request(is_first_message=True)

        async def _mock_rewrite(ctx, br):
            ctx.rewritten_question = ctx.original_question
            return ctx

        async def _mock_classify(ctx, br, st):
            ctx.intent = "academic"
            return ctx

        async def _mock_search(ctx, br, sb, st):
            ctx.search_results = []
            return ctx

        with patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
             patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
             patch("app.api.chat.search_documents", side_effect=_mock_search):

            # check_cache는 graceful degradation하므로 예외 없이 진행
            context = await _run_pipeline(request, bedrock, supabase, settings)
            assert context.cache_hit is False

    async def test_cache_save_failure_does_not_affect_response(self):
        """캐시 저장 실패 시 사용자 응답에 영향 없음."""
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        supabase.insert_answer_cache.side_effect = RuntimeError("DB 장애")

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        # 예외가 발생하지 않아야 함
        await _save_answer_cache(context, ["답변"], bedrock, supabase)
        # insert_answer_cache가 호출되었지만 예외는 잡힘
        supabase.insert_answer_cache.assert_called_once()


# ── Property 20: 파이프라인 경로 결정론 ──
# Validates: Requirements 6.3


class TestPipelineDeterminism:
    """Property 20: 동일 요청 파라미터 → 동일 파이프라인 경로."""

    @hyp_settings(max_examples=20)
    @given(
        is_first=st.booleans(),
        intent=st.sampled_from(["academic", "chitchat"]),
        has_results=st.booleans(),
    )
    async def test_same_params_same_path(self, is_first, intent, has_results):
        """동일 파라미터로 두 번 실행하면 동일 경로를 따른다."""
        settings = _create_settings()

        paths = []
        for _ in range(2):
            bedrock = _create_bedrock()
            supabase = _create_supabase()

            async def _mock_rewrite(ctx, br):
                ctx.rewritten_question = ctx.original_question
                return ctx

            async def _mock_classify(ctx, br, st, _intent=intent):
                ctx.intent = _intent
                return ctx

            async def _mock_search(ctx, br, sb, st, _has=has_results):
                ctx.search_results = [_make_search_result()] if _has else []
                return ctx

            request = _make_chat_request(is_first_message=is_first)

            with patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
                 patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
                 patch("app.api.chat.search_documents", side_effect=_mock_search):

                ctx = await _run_pipeline(request, bedrock, supabase, settings)

                path = (ctx.cache_hit, ctx.intent, bool(ctx.search_results))
                paths.append(path)

        assert paths[0] == paths[1]


# ── disconnect 테스트 ──


class TestClientDisconnect:
    async def test_disconnect_stops_streaming(self):
        """클라이언트 disconnect 시 스트리밍이 중단된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        supabase = _create_supabase()
        http_request = _make_http_request()
        http_request.is_disconnected.side_effect = [False, True]  # 두 번째에서 disconnect

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[],
        )

        with patch("app.api.chat.stream_sse_response") as mock_sse:
            async def _mock_stream(*args, **kwargs):
                yield 'data: {"type": "fallback"}\n\n'
                yield 'data: {"type": "done"}\n\n'
                yield 'data: {"type": "extra"}\n\n'  # disconnect 후이므로 도달 안 됨
            mock_sse.return_value = _mock_stream()

            chunks = []
            async for chunk in _streaming_wrapper(
                context, bedrock, supabase, settings, http_request, is_first_message=False
            ):
                chunks.append(chunk)

        # disconnect 전 첫 번째 청크만 수신되어야 함
        assert len(chunks) == 1
        assert "fallback" in chunks[0]
