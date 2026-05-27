"""POST /api/chat 엔드포인트 테스트. 파이프라인 오케스트레이션 + 캐시 저장 + 에러 전파."""

from __future__ import annotations

import asyncio
import json
import logging
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
from app.streaming.sse import stream_sse_response


# ── 헬퍼 ──


def _create_settings() -> Settings:
    with patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
    }, clear=True):
        return Settings(_env_file=None)


def _create_bedrock() -> AsyncMock:
    bedrock = AsyncMock()

    # 입력 길이에 맞춰 임베딩 반환 — semantic_cache batch [cache_key, q] 와
    # _save_answer_cache 단건 호출 양쪽 호환. off-by-one buggy mock 회피.
    def _embed(texts, **kwargs):
        return [[0.1 + 0.01 * i] * 1024 for i in range(len(texts))]

    bedrock.embed_texts.side_effect = _embed
    bedrock.invoke_llm.return_value = (
        "academic",
        TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )
    bedrock.last_stream_usage = TokenUsage()
    return bedrock


def _create_postgres() -> AsyncMock:
    postgres = AsyncMock()
    postgres.search_answer_cache.return_value = None
    postgres.search_documents.return_value = []
    postgres.search_similar_questions.return_value = []
    postgres.upsert_answer_cache.return_value = None
    return postgres


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
        confidence="high",
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
        postgres = _create_postgres()
        postgres.search_answer_cache.return_value = _make_cache_match()

        request = _make_chat_request(is_first_message=True)

        with patch("app.api.chat.rewrite_query") as mock_rewrite, \
             patch("app.api.chat.classify_intent") as mock_intent, \
             patch("app.api.chat.search_documents") as mock_search:

            context = await _run_pipeline(request, bedrock, postgres, settings)

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
        postgres = _create_postgres()

        async def _mock_classify(ctx, br, st):
            ctx.intent = "chitchat"
            return ctx

        request = _make_chat_request(is_first_message=False)

        with patch("app.api.chat.rewrite_query") as mock_rewrite, \
             patch("app.api.chat.classify_intent", side_effect=_mock_classify), \
             patch("app.api.chat.search_documents") as mock_search:

            context = await _run_pipeline(request, bedrock, postgres, settings)

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
        postgres = _create_postgres()

        call_order: list[str] = []

        async def _mock_classify(ctx, br, st):
            call_order.append("classify")
            ctx.intent = "academic"
            return ctx

        async def _mock_rewrite(ctx, br, st):
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

            await _run_pipeline(request, bedrock, postgres, settings)

        assert call_order == ["classify", "rewrite", "search"]

    async def test_full_academic_pipeline(self):
        """학사규정 경로: 캐시미스 → 재작성 → 의도분류 → 벡터검색 전체 실행."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        async def _mock_rewrite(ctx, br, st):
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

            context = await _run_pipeline(request, bedrock, postgres, settings)

            assert context.intent == "academic"
            assert len(context.search_results) == 1

    async def test_session_id_propagated_to_context(self):
        """PR-50: ChatRequest.session_id 가 PipelineContext.session_id 에 복사된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        request = _make_chat_request(is_first_message=True)
        # _make_chat_request 의 session_id 값은 헬퍼 default 따름
        expected_session = request.session_id

        context = await _run_pipeline(request, bedrock, postgres, settings)

        assert context.session_id == expected_session

    async def test_chat_endpoint_sets_session_contextvar(self):
        """PR-50 핵심 회귀: chat() endpoint 진입 시 SESSION_ID_VAR 가 설정된다.

        외부 리뷰 Minor 2 — `_run_pipeline` 직접 호출 테스트만으로는
        chat 함수 진입부의 `set_session_id(request.session_id)` 줄이 회귀
        테스트 밖에 있다. 본 테스트는 chat() 호출 시 _run_pipeline 안에서
        SESSION_ID_VAR.get() 이 request.session_id 와 같은지 spy 로 확인.
        """
        from app.api.chat import chat
        from app.logging_context import SESSION_ID_VAR

        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        http_request = _make_http_request()
        request = _make_chat_request(is_first_message=True)

        captured: list[str] = []

        async def _spy_pipeline(req, br, sb, st):
            captured.append(SESSION_ID_VAR.get())
            ctx = PipelineContext(
                original_question=req.message,
                user_context=req.user_context,
                session_id=req.session_id,
            )
            ctx.cache_hit = True
            ctx.cached_answer = "test"
            return ctx

        async def _mock_stream_gen():
            yield 'data: {"type": "done", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}\n\n'

        with patch("app.api.chat._run_pipeline", side_effect=_spy_pipeline), \
             patch("app.api.chat.stream_sse_response", side_effect=lambda *a, **k: _mock_stream_gen()):
            await chat(request, http_request, settings, bedrock, postgres)

        assert captured == [request.session_id]

    async def test_history_converted_to_dicts(self):
        """history가 HistoryMessage → dict로 변환된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        request = _make_chat_request(
            is_first_message=False,
            history=[
                {"role": "user", "content": "이전 질문"},
                {"role": "assistant", "content": "이전 답변"},
            ],
        )

        contexts_captured = []

        async def _mock_rewrite(ctx, br, st):
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

            await _run_pipeline(request, bedrock, postgres, settings)

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
        postgres = _create_postgres()

        context = PipelineContext(
            original_question="휴학 기간은?",
            intent="academic",
            search_results=[_make_search_result(doc_id=1)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="학사요람.pdf", page=10)],
        )

        await _save_answer_cache(context, ["최대 ", "4년입니다."], bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_called_once()
        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
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
        postgres = _create_postgres()

        context = PipelineContext(
            original_question=question,
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        await _save_answer_cache(context, [answer], bedrock, postgres, _create_settings())

        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert cache.question
        assert cache.embedding
        assert cache.answer
        assert cache.source_doc_ids
        assert cache.sources

    async def test_save_reuses_cache_key_embedding(self):
        """semantic_cache 가 보관한 cache_key_embedding 을 재사용 — embed_texts 호출 X.

        Task 16 후속 — 재사용 키가 검색용 question_embedding 에서 캐시 키
        cache_key_embedding 으로 변경됨 (cross-user_context 누설 차단 도입).
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        cached_embedding = [0.42] * 1024
        context = PipelineContext(
            original_question="휴학 기간은?",
            intent="academic",
            search_results=[_make_search_result(doc_id=1)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="학사요람.pdf", page=10)],
            cache_key_embedding=cached_embedding,
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        bedrock.embed_texts.assert_not_called()
        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert cache.embedding == cached_embedding

    async def test_save_creates_new_cache_key_embedding_when_none(self):
        """context.cache_key_embedding 이 None 이면 _build_cache_key 결과로 새로 임베딩.

        멀티턴 academic 시나리오처럼 semantic_cache 단계가 skip 된 경로에서
        _save_answer_cache 가 cache 키를 직접 구성해야 한다.
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        context = PipelineContext(
            original_question="휴학 기간은?",
            user_context="소프트웨어학부 3학년 재학",
            intent="academic",
            search_results=[_make_search_result(doc_id=1)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="학사요람.pdf", page=10)],
        )
        assert context.cache_key_embedding is None

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        # _build_cache_key 결과 = "[사용자] 소프트웨어학부 3학년 재학\n[질문] 휴학 기간은?"
        bedrock.embed_texts.assert_called_once()
        call = bedrock.embed_texts.call_args
        texts = call.args[0]
        assert len(texts) == 1
        assert "[사용자]" in texts[0]
        assert "소프트웨어학부 3학년 재학" in texts[0]
        assert "휴학 기간은?" in texts[0]
        assert call.kwargs["input_type"] == "search_query"

    async def test_save_uses_cache_key_text_in_question_column(self):
        """answer_cache.question 컬럼에 _build_cache_key 결과가 박혀 dedup 키 분리.

        RPC upsert_answer_cache 의 DELETE WHERE question = ... dedup 키 + 락 키
        둘 다 cohort 별로 갈라져 cross-cohort thrash 차단 + 락 경합 해결
        (외부 리뷰 B1 a/b 처리).
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="휴학 기간은?",
            user_context="소프트웨어학부 3학년 재학",
            intent="academic",
            search_results=[_make_search_result(doc_id=1)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=10)],
            cache_key_embedding=[0.42] * 1024,
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert "[사용자] 소프트웨어학부 3학년 재학" in cache.question
        assert "[질문] 휴학 기간은?" in cache.question

    async def test_save_with_empty_user_context_uses_original_question(self):
        """user_context 가 비어있으면 cache.question 에 원문 그대로 (BE 폴백 호환).

        _build_cache_key contract — user_context 빈 경우 question 만 반환.
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="휴학 기간은?",
            user_context="",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=10)],
            cache_key_embedding=[0.42] * 1024,
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert cache.question == "휴학 기간은?"
        assert "[사용자]" not in cache.question

    # ── 오염 답변 WRITE 게이트 (Layer 2 / 외부 리뷰 A3) ──

    async def test_save_rejects_polluted_answer_id10_case(self):
        """id=10 사고 원문 그대로 → 캐시 저장 거부 (upsert RPC 호출 X).

        LLM Layer 1 자발 준수 실패해도 환각 답변이 캐시에 박히는 경로 차단.
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="2025년도 졸업요건에 자격증 따야해?",
            user_context="소프트웨어학부 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        polluted = [
            "2025학번 이후 입학생부터는 K*코딩역량인증이 필요합니다. ",
            "참고로 당신은 2025학번 이전 학생이므로 해당되지 않습니다.",
        ]

        await _save_answer_cache(context, polluted, bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_not_called()

    async def test_save_accepts_clean_answer(self):
        """정상 답변 → 정상 캐시 저장 (false positive 방지)."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="휴학 신청 방법은?",
            user_context="소프트웨어학부 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        clean = [
            "휴학은 신청서를 제출하면 됩니다. 본인의 학번에 해당하는 규정을 확인하세요.",
        ]

        await _save_answer_cache(context, clean, bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_called_once()

    async def test_save_accepts_objective_year_mention(self):
        """객관 진술 ("2025학번 이후 입학생부터는") → 캐시 저장 허용 (false positive 방지).

        가드 정규식은 (호칭 + 학번) 또는 (학번 + 이전/이후 + 단정 종결어미) 만 매치 —
        객관 진술 키워드 ("이후 입학생", "이전에 입학") 와 분리.
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="K*코딩역량 누가 따야해?",
            user_context="컴공 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        objective = [
            "K*코딩역량인증은 2025학번 이후 입학생에게 적용됩니다. ",
            "본인의 학번 기준 적용 여부를 확인하세요.",
        ]

        await _save_answer_cache(context, objective, bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_called_once()

    async def test_save_logs_warning_on_polluted_answer(self, caplog):
        """게이트 발동 시 WARNING 로그 박힘 (CloudWatch 모니터링 호환).

        log prefix "캐시 저장 skip — 오염 답변" 으로 Layer 1 자발 준수 실패율
        측정 가능 (m1 자동 누적).
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="질문",
            user_context="컴공 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        polluted = ["당신은 2025학번 이전 학생이므로 면제됩니다."]

        with caplog.at_level(logging.WARNING):
            await _save_answer_cache(
                context, polluted, bedrock, postgres, _create_settings()
            )

        assert "캐시 저장 skip — 오염 답변 패턴 감지" in caplog.text
        postgres.upsert_answer_cache.assert_not_called()

    # ── Layer 4 — freeform 캐시 skip (BE PR #96 후속, 외부 리뷰 M1) ──

    async def test_save_skips_cache_when_user_context_has_additional_info(self):
        """학생 자유 입력 (additionalInfo) 부착 시 캐시 저장 skip — cohort 누설 차단."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="휴학 신청 방법은?",
            user_context="소프트웨어학부 2024학번 3학년 재학. 추가 정보: 부전공 통계학 신청 예정",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_not_called()

    async def test_save_skips_cache_when_user_context_has_job_description(self):
        """직원 자유 입력 (jobDescription) 부착 시 캐시 저장 skip."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="직원 휴가 신청 절차는?",
            user_context="학사지원과 직원. 담당 업무: 졸업 사정 + 학적 변동 처리",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_not_called()

    async def test_save_logs_info_on_freeform_skip(self, caplog):
        """Layer 4 발동 시 INFO 로그 박힘 (CloudWatch 모니터링 — freeform 사용자 비율 추적)."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="질문",
            user_context="컴공 3학년 재학. 추가 정보: 부전공 통계학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )

        with caplog.at_level(logging.INFO):
            await _save_answer_cache(
                context, ["답변"], bedrock, postgres, _create_settings()
            )

        assert "캐시 저장 skip — 자유 입력 개인화 답변" in caplog.text
        postgres.upsert_answer_cache.assert_not_called()

    async def test_save_accepts_when_no_freeform_in_user_context(self):
        """cohort-only user_context (freeform 없음) → 정상 캐시 저장 (기존 행동 보존)."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="휴학 신청 방법은?",
            user_context="소프트웨어학부 2024학번 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_called_once()

    # ── Layer 2 정규식 정정 (M2 처방, 외부 리뷰) ──
    # L1 의 rule 2 완화 (명시 학번 인용 허용) 와 L2 정합 — bare 패턴 2개 제거
    # 후 명시 인용 통과, 환각 추론 (이전/이후 + 단정 종결어미) 차단 유지.

    async def test_save_accepts_explicit_admission_year_citation(self):
        """L1 허용 예시 ("본인은 2024학번이시므로") 답변이 L2 통과해 정상 캐시 저장.

        BE PR #96 후 명시 학번 인용 정상 개인화 답변의 캐시 경로 보장 (M2 처방).
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="K*코딩역량 적용 대상인가요?",
            user_context="소프트웨어학부 2024학번 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        # L1 의 허용 예시 그대로 답변에 박힌 케이스
        explicit_citation = [
            "본인은 2024학번이시므로 K*코딩역량인증 면제 대상입니다.",
        ]

        await _save_answer_cache(
            context, explicit_citation, bedrock, postgres, _create_settings()
        )

        postgres.upsert_answer_cache.assert_called_once()

    async def test_save_rejects_id10_with_before_after_assertion(self):
        """id=10 사고 패턴 ("X학번 이전/이후 + 단정 종결어미") 은 변경 후에도 차단 유지."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="K*코딩역량 면제 가능한가요?",
            user_context="컴공 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        polluted = [
            "당신은 2025학번 이전 학생이므로 K*코딩역량 면제됩니다.",
        ]

        await _save_answer_cache(context, polluted, bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_not_called()

    async def test_save_accepts_bonin_citation_without_before_after(self):
        """비교어 없는 명시 인용 ("본인은 24학번 학생이라") 도 통과 (M2 정합)."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="졸업 요건은?",
            user_context="컴공 2024학번 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        citation = [
            "본인은 24학번 학생이라 적용 규정은 다음과 같습니다.",
        ]

        await _save_answer_cache(context, citation, bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_called_once()

    async def test_save_rejects_other_before_after_patterns(self):
        """일반화된 "X학번 이후 + 단정 종결어미" 도 차단 유지 (3rd 패턴)."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        context = PipelineContext(
            original_question="질문",
            user_context="컴공 3학년 재학",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
            cache_key_embedding=[0.42] * 1024,
        )
        polluted = [
            "2024학번 이후이며 적용되지 않는 규정이 있습니다.",
        ]

        await _save_answer_cache(context, polluted, bedrock, postgres, _create_settings())

        postgres.upsert_answer_cache.assert_not_called()

    async def test_save_includes_high_confidence(self):
        """search_results 최고 유사도가 HIGH 임계값(0.9) 이상이면 'high' 박제.

        SSE 시나리오 C(cache hit) 가 동일 박제값을 노출해 시나리오 A(cache miss)
        와 UX 정합을 유지하기 위한 회귀 잠금.
        """
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result(similarity=0.95)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert cache.confidence == "high"

    async def test_save_includes_medium_confidence(self):
        """MEDIUM(0.8) ≤ max < HIGH(0.9) 이면 'medium' 박제."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result(similarity=0.85)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert cache.confidence == "medium"

    async def test_save_includes_low_confidence(self):
        """max < MEDIUM(0.8) 이면 'low' 박제."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result(similarity=0.5)],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())

        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        assert cache.confidence == "low"

    async def test_persisted_confidence_equals_scenario_a_meta_confidence(self):
        """박제 invariant — SSE 시나리오 A 가 노출한 meta.confidence ==
        동일 답변의 _save_answer_cache 박제 confidence.

        박제 방식의 핵심 약속(같은 답변은 cache miss/hit 무관하게 동일 신뢰도를
        화면에 표시) 의 회귀 잠금. 미래에 ``context.search_results`` 변형 도입 /
        두 호출 중 한쪽 인자 변경 시 이 단언이 즉시 fail.
        """
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result(similarity=0.85)],  # MEDIUM 범위
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        async def _mock_token_stream(*args, **kwargs):
            yield "답변"

        # 1) SSE 시나리오 A meta.confidence 캡처
        chunks: list[str] = []
        with patch("app.streaming.sse.generate_response", side_effect=_mock_token_stream):
            async for chunk in stream_sse_response(context, bedrock, settings):
                chunks.append(chunk)
        meta_event = json.loads(chunks[0].removeprefix("data: ").strip())
        sse_confidence = meta_event["confidence"]

        # 2) 동일 context 로 _save_answer_cache 박제 confidence 캡처
        await _save_answer_cache(context, ["답변"], bedrock, postgres, settings)
        cache: AnswerCache = postgres.upsert_answer_cache.call_args[0][0]
        persisted_confidence = cache.confidence

        # 3) 동일성 단언 — 두 값 일치 + 의도된 값(medium) 추가 가드
        assert sse_confidence == persisted_confidence == "medium"


# ── Property 16: 잡담/Fallback 캐시 미저장 ──
# Validates: Requirements 12.2, 12.4


class TestNoCacheForChitchatAndFallback:
    """Property 16: 잡담/Fallback 시 answer_cache에 저장하지 않음."""

    async def test_chitchat_no_cache(self):
        """잡담 경로에서는 캐시가 저장되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()
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
                context, bedrock, postgres, settings, http_request, is_first_message=True
            ):
                chunks.append(chunk)

        postgres.upsert_answer_cache.assert_not_called()

    async def test_fallback_no_cache(self):
        """폴백 경로에서는 캐시가 저장되지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()
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
                context, bedrock, postgres, settings, http_request, is_first_message=True
            ):
                pass

        postgres.upsert_answer_cache.assert_not_called()

    async def test_cache_hit_no_cache(self):
        """캐시 히트 경로에서는 새로운 캐시를 저장하지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()
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
                context, bedrock, postgres, settings, http_request, is_first_message=True
            ):
                pass

        postgres.upsert_answer_cache.assert_not_called()


# ── 에러 전파 테스트 ──
# Validates: Requirements 7.3, 7.4, 7.5, 12.3


class TestErrorPropagation:
    async def test_intent_classification_failure_raises(self):
        """의도 분류 단계 장애 시 예외가 전파된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        request = _make_chat_request(is_first_message=False)

        async def _mock_rewrite(ctx, br, st):
            ctx.rewritten_question = ctx.original_question
            return ctx

        with patch("app.api.chat.rewrite_query", side_effect=_mock_rewrite), \
             patch("app.api.chat.classify_intent") as mock_ci:
            mock_ci.side_effect = RuntimeError("Bedrock 장애")

            with pytest.raises(RuntimeError, match="Bedrock 장애"):
                await _run_pipeline(request, bedrock, postgres, settings)

    async def test_cache_search_graceful_degradation(self):
        """시맨틱 캐시 탐색 실패 시 파이프라인이 계속 진행된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()

        # check_cache 내부에서 예외를 잡고 graceful degradation
        bedrock.embed_texts.side_effect = RuntimeError("Postgres 장애")

        request = _make_chat_request(is_first_message=True)

        async def _mock_rewrite(ctx, br, st):
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
            context = await _run_pipeline(request, bedrock, postgres, settings)
            assert context.cache_hit is False

    async def test_cache_save_failure_does_not_affect_response(self):
        """캐시 저장 실패 시 사용자 응답에 영향 없음."""
        bedrock = _create_bedrock()
        postgres = _create_postgres()
        postgres.upsert_answer_cache.side_effect = RuntimeError("DB 장애")

        context = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[SourceDoc(doc_id=1, chunk_id=1, doc_name="a.pdf", page=1)],
        )

        # 예외가 발생하지 않아야 함
        await _save_answer_cache(context, ["답변"], bedrock, postgres, _create_settings())
        # upsert_answer_cache가 호출되었지만 예외는 잡힘
        postgres.upsert_answer_cache.assert_called_once()


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
            postgres = _create_postgres()

            async def _mock_rewrite(ctx, br, st):
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

                ctx = await _run_pipeline(request, bedrock, postgres, settings)

                path = (ctx.cache_hit, ctx.intent, bool(ctx.search_results))
                paths.append(path)

        assert paths[0] == paths[1]


# ── disconnect 테스트 ──


class TestClientDisconnect:
    async def test_disconnect_stops_streaming(self):
        """클라이언트 disconnect 시 스트리밍이 중단된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        postgres = _create_postgres()
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
                context, bedrock, postgres, settings, http_request, is_first_message=False
            ):
                chunks.append(chunk)

        # disconnect 전 첫 번째 청크만 수신되어야 함
        assert len(chunks) == 1
        assert "fallback" in chunks[0]
