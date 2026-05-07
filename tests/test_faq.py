"""FAQ 분석 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import SearchResult, TokenUsage
from app.services.faq_analyzer import InsufficientDataError, analyze_faq
from app.api.faq import faq_analyze
from app.models.schemas import FAQAnalyzeRequest


# ── 헬퍼 ──


def _create_settings() -> Settings:
    with patch.dict(os.environ, {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
    }, clear=True):
        return Settings(_env_file=None)


def _create_bedrock(n_questions: int = 5) -> AsyncMock:
    """임베딩 + LLM 모킹. 클러스터가 형성되도록 유사 벡터 생성."""
    bedrock = AsyncMock()

    # 클러스터 형성을 위해 2그룹의 유사 벡터 생성
    rng = np.random.RandomState(42)
    embeddings = []
    for i in range(n_questions):
        if i < n_questions // 2:
            base = np.ones(1024) * 0.8
        else:
            base = np.ones(1024) * -0.5
        noise = rng.normal(0, 0.01, 1024)
        embeddings.append((base + noise).tolist())

    bedrock.embed_texts.return_value = embeddings
    bedrock.invoke_llm.return_value = (
        "FAQ 답변 초안입니다.",
        TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    )
    return bedrock


def _create_supabase() -> AsyncMock:
    supabase = AsyncMock()
    supabase.search_documents.return_value = [
        SearchResult(
            chunk_id=1,
            doc_id=1,
            content="제1조 내용",
            metadata={"doc_name": "학사요람.pdf", "page": 10},
            similarity_score=0.85,
        ),
    ]
    return supabase


# ── Property 18: FAQ 분석 응답 구조 ──
# Validates: Requirements 8.2, 8.5


class TestFAQResponseStructure:
    """Property 18: 각 후보에 question, draft_answer, frequency 포함, 후보 수 ≤ top_k."""

    async def test_candidates_have_required_fields(self):
        """각 후보에 question, draft_answer, frequency가 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        supabase = _create_supabase()

        questions = [f"질문 {i}" for i in range(6)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        for c in candidates:
            assert "question" in c
            assert "draft_answer" in c
            assert "frequency" in c
            assert len(c["question"]) > 0
            assert len(c["draft_answer"]) > 0
            assert c["frequency"] >= 1

    async def test_candidates_count_le_top_k(self):
        """후보 수가 top_k 이하이다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=10)
        supabase = _create_supabase()

        questions = [f"질문 {i}" for i in range(10)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=3,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        assert len(candidates) <= 3

    @hyp_settings(max_examples=10)
    @given(
        top_k=st.integers(min_value=1, max_value=10),
    )
    async def test_never_exceeds_top_k(self, top_k):
        """어떤 top_k 값이든 후보 수를 초과하지 않는다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=8)
        supabase = _create_supabase()

        questions = [f"질문 {i}" for i in range(8)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=top_k,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        assert len(candidates) <= top_k


# ── INSUFFICIENT_DATA 테스트 ──


class TestInsufficientData:
    async def test_raises_when_too_few_questions(self):
        """질문 수가 min_cluster_size 미만이면 InsufficientDataError."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=1)
        supabase = _create_supabase()

        with pytest.raises(InsufficientDataError) as exc_info:
            await analyze_faq(
                questions=["질문 하나"],
                top_k=5,
                min_cluster_size=3,
                bedrock=bedrock,
                supabase=supabase,
                settings=settings,
            )

        assert exc_info.value.min_required == 3

    async def test_endpoint_returns_400(self):
        """엔드포인트에서 INSUFFICIENT_DATA 시 HTTP 400 반환."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=1)
        supabase = _create_supabase()

        request = FAQAnalyzeRequest(
            questions=["질문 하나"],
            top_k=5,
            min_cluster_size=3,
        )

        result = await faq_analyze(request, settings, bedrock, supabase)

        assert result.status_code == 400
        import json
        body = json.loads(result.body)
        assert body["error_code"] == "INSUFFICIENT_DATA"

    async def test_exact_min_cluster_size_proceeds(self):
        """질문 수가 정확히 min_cluster_size이면 클러스터링을 시도한다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=2)
        supabase = _create_supabase()

        # 예외가 발생하지 않아야 함
        candidates = await analyze_faq(
            questions=["질문 A", "질문 B"],
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        assert isinstance(candidates, list)


# ── 단위 테스트 ──


class TestAnalyzeFAQUnit:
    async def test_sorted_by_frequency_desc(self):
        """후보가 빈도순 내림차순으로 정렬된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        supabase = _create_supabase()

        questions = [f"질문 {i}" for i in range(6)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=10,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                assert candidates[i]["frequency"] >= candidates[i + 1]["frequency"]

    async def test_no_clusters_returns_empty(self):
        """클러스터가 형성되지 않으면 빈 리스트를 반환한다."""
        settings = _create_settings()
        bedrock = AsyncMock()

        # 모든 벡터를 완전히 랜덤하게 → 클러스터 미형성
        rng = np.random.RandomState(99)
        embeddings = [rng.normal(0, 1, 1024).tolist() for _ in range(5)]
        bedrock.embed_texts.return_value = embeddings

        supabase = _create_supabase()

        candidates = await analyze_faq(
            questions=[f"랜덤 질문 {i}" for i in range(5)],
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        assert isinstance(candidates, list)
        # 클러스터 미형성 시 빈 리스트
        assert len(candidates) == 0

    async def test_no_search_results_fallback_answer(self):
        """벡터 검색 결과가 없으면 폴백 답변 초안이 생성된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=4)
        supabase = _create_supabase()
        supabase.search_documents.return_value = []  # 검색 결과 없음

        questions = [f"질문 {i}" for i in range(4)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        for c in candidates:
            assert "관련 문서를 찾을 수 없어" in c["draft_answer"]

    async def test_uses_answer_model(self):
        """답변 초안 생성 시 answer 모델을 사용한다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=4)
        supabase = _create_supabase()

        questions = [f"질문 {i}" for i in range(4)]
        await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        # invoke_llm이 호출되었으면 model="answer" 확인
        if bedrock.invoke_llm.called:
            call_kwargs = bedrock.invoke_llm.call_args
            assert call_kwargs.kwargs.get("model") == "answer"

    async def test_endpoint_success_response(self):
        """엔드포인트 성공 시 status=success, candidates 배열 반환."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        supabase = _create_supabase()

        request = FAQAnalyzeRequest(
            questions=[f"질문 {i}" for i in range(6)],
            top_k=5,
            min_cluster_size=2,
        )

        result = await faq_analyze(request, settings, bedrock, supabase)

        assert result["status"] == "success"
        assert "candidates" in result
        assert isinstance(result["candidates"], list)


# ── 병렬화 회귀 테스트 (R8) ──


class TestParallelization:
    """답변 초안 생성이 병렬화 + 동시성 제한 + 부분 실패 처리되는지 검증."""

    async def test_invoke_llm_called_per_cluster_in_order(self):
        """클러스터 수만큼 invoke_llm 호출 + 빈도순 정렬 보존."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=8)
        supabase = _create_supabase()

        questions = [f"질문 {i}" for i in range(8)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=10,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        assert bedrock.invoke_llm.call_count == len(candidates)
        # 정렬 검증이 trivially pass 하지 않도록 cluster 2개 이상 보장
        assert len(candidates) >= 2, "정렬 검증 의미 있으려면 cluster 2개 이상 필요"
        # 빈도 내림차순 보존 (gather 결과 매핑이 순서를 깨뜨리지 않음)
        for i in range(len(candidates) - 1):
            assert candidates[i]["frequency"] >= candidates[i + 1]["frequency"]

    async def test_partial_failure_placeholder(self, caplog):
        """일부 cluster 실패 시 나머지는 정상 + 실패는 placeholder + warning."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        supabase = _create_supabase()

        # 두 번째 호출만 실패 (호출 순서는 비결정적이지만 1개는 반드시 실패)
        bedrock.invoke_llm.side_effect = [
            ("정상 답변 1", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
            RuntimeError("Bedrock throttle"),
            ("정상 답변 2", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
            ("정상 답변 3", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
            ("정상 답변 4", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
        ]

        questions = [f"질문 {i}" for i in range(6)]
        with caplog.at_level("WARNING", logger="app.services.faq_analyzer"):
            candidates = await analyze_faq(
                questions=questions,
                top_k=10,
                min_cluster_size=2,
                bedrock=bedrock,
                supabase=supabase,
                settings=settings,
            )

        failed = [c for c in candidates if c["draft_answer"].startswith("답변 초안 생성 실패")]
        succeeded = [c for c in candidates if not c["draft_answer"].startswith("답변 초안 생성 실패")]

        assert len(failed) == 1
        assert "RuntimeError" in failed[0]["draft_answer"]
        assert len(succeeded) >= 1
        assert any("FAQ 답변 초안 생성 실패" in rec.message for rec in caplog.records)

    async def test_concurrency_capped_by_semaphore(self):
        """FAQ_CONCURRENCY=1 시 직렬 강제 — Semaphore가 실제로 동시성 제한 적용."""
        settings = _create_settings()
        # FAQ_CONCURRENCY=1 → invoke_llm 동시 진입 정확히 1개로 강제
        object.__setattr__(settings, "FAQ_CONCURRENCY", 1)

        bedrock = _create_bedrock(n_questions=8)
        supabase = _create_supabase()

        in_flight = 0
        max_in_flight = 0

        async def _slow_invoke(*args, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                # 다른 task 가 Semaphore 를 못 받아 대기 중인지 확인할 시간
                await asyncio.sleep(0.01)
                return (
                    "답변",
                    TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
            finally:
                in_flight -= 1

        bedrock.invoke_llm.side_effect = _slow_invoke

        questions = [f"질문 {i}" for i in range(8)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=10,
            min_cluster_size=2,
            bedrock=bedrock,
            supabase=supabase,
            settings=settings,
        )

        # 클러스터 2개 이상 형성돼야 동시성 제한이 의미 있음
        assert len(candidates) >= 2
        # FAQ_CONCURRENCY=1 → 직렬 강제 → 동시 진입 정확히 1
        assert max_in_flight == 1

    async def test_inner_cancelled_error_propagates(self):
        """inner task의 CancelledError 는 caller 로 re-raise 됨 (응답 깨짐 방지).

        gather(return_exceptions=True) 가 CancelledError 도 결과로 wrap 하므로,
        BaseException 가드가 약화되면 draft_answer 자리에 CancelledError 객체가
        들어가 JSON 직렬화 시 응답 전체가 깨질 수 있음. 본 테스트가 회귀 잠금.
        """
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        supabase = _create_supabase()

        # _create_bedrock(n=6) 은 cluster 2개 → invoke_llm 2회 호출 (각 cluster 1회).
        # 어느 호출이든 CancelledError 가 caller 로 propagate 되는지가 핵심.
        bedrock.invoke_llm.side_effect = [
            ("정상 답변", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
            asyncio.CancelledError(),
            ("여분", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
        ]

        questions = [f"질문 {i}" for i in range(6)]
        with pytest.raises(asyncio.CancelledError):
            await analyze_faq(
                questions=questions,
                top_k=10,
                min_cluster_size=2,
                bedrock=bedrock,
                supabase=supabase,
                settings=settings,
            )
