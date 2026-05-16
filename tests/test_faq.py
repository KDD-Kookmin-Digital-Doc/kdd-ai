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
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
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


def _create_postgres() -> AsyncMock:
    postgres = AsyncMock()
    postgres.search_documents.return_value = [
        SearchResult(
            chunk_id=1,
            doc_id=1,
            content="제1조 내용",
            metadata={"doc_name": "학사요람.pdf", "page": 10},
            similarity_score=0.85,
        ),
    ]
    return postgres


# ── Property 18: FAQ 분석 응답 구조 ──
# Validates: Requirements 8.2, 8.5


class TestFAQResponseStructure:
    """Property 18: 각 후보에 question, draft_answer, frequency 포함, 후보 수 ≤ top_k."""

    async def test_candidates_have_required_fields(self):
        """각 후보에 question, draft_answer, frequency가 포함된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        questions = [f"질문 {i}" for i in range(6)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
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
        postgres = _create_postgres()

        questions = [f"질문 {i}" for i in range(10)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=3,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
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
        postgres = _create_postgres()

        questions = [f"질문 {i}" for i in range(8)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=top_k,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        assert len(candidates) <= top_k


# ── INSUFFICIENT_DATA 테스트 ──


class TestInsufficientData:
    async def test_raises_when_too_few_questions(self):
        """질문 수가 min_cluster_size 미만이면 InsufficientDataError."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=1)
        postgres = _create_postgres()

        with pytest.raises(InsufficientDataError) as exc_info:
            await analyze_faq(
                questions=["질문 하나"],
                top_k=5,
                min_cluster_size=3,
                bedrock=bedrock,
                postgres=postgres,
                settings=settings,
            )

        assert exc_info.value.min_required == 3

    async def test_endpoint_returns_400(self):
        """엔드포인트에서 INSUFFICIENT_DATA 시 HTTP 400 반환."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=1)
        postgres = _create_postgres()

        request = FAQAnalyzeRequest(
            questions=["질문 하나"],
            top_k=5,
            min_cluster_size=3,
        )

        result = await faq_analyze(request, settings, bedrock, postgres)

        assert result.status_code == 400
        import json
        body = json.loads(result.body)
        assert body["error_code"] == "INSUFFICIENT_DATA"

    async def test_exact_min_cluster_size_proceeds(self):
        """질문 수가 정확히 min_cluster_size이면 클러스터링을 시도한다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=2)
        postgres = _create_postgres()

        # 예외가 발생하지 않아야 함
        candidates = await analyze_faq(
            questions=["질문 A", "질문 B"],
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        assert isinstance(candidates, list)


# ── 단위 테스트 ──


class TestAnalyzeFAQUnit:
    async def test_sorted_by_frequency_desc(self):
        """후보가 빈도순 내림차순으로 정렬된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        questions = [f"질문 {i}" for i in range(6)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=10,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
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

        postgres = _create_postgres()

        candidates = await analyze_faq(
            questions=[f"랜덤 질문 {i}" for i in range(5)],
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        assert isinstance(candidates, list)
        # 클러스터 미형성 시 빈 리스트
        assert len(candidates) == 0

    async def test_no_search_results_fallback_answer(self):
        """벡터 검색 결과가 없으면 폴백 답변 초안이 생성된다."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=4)
        postgres = _create_postgres()
        postgres.search_documents.return_value = []  # 검색 결과 없음

        questions = [f"질문 {i}" for i in range(4)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        for c in candidates:
            assert "관련 문서를 찾을 수 없어" in c["draft_answer"]

    async def test_uses_answer_and_light_models(self):
        """답변 초안은 answer 모델, 카테고리 분류는 light 모델 사용 (B1)."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=4)
        postgres = _create_postgres()

        questions = [f"질문 {i}" for i in range(4)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        # 답변(answer) + 카테고리(light) 각각 cluster 수만큼 호출.
        # call_args(마지막 호출) 가 아닌 call_args_list(전체) 순회 — 인터리브 순서 무관.
        call_list = bedrock.invoke_llm.call_args_list
        answer_calls = [c for c in call_list if c.kwargs.get("model") == "answer"]
        light_calls = [c for c in call_list if c.kwargs.get("model") == "light"]
        assert len(answer_calls) == len(candidates)
        assert len(light_calls) == len(candidates)

    async def test_endpoint_success_response(self):
        """엔드포인트 성공 시 status=success, candidates 배열 반환."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        request = FAQAnalyzeRequest(
            questions=[f"질문 {i}" for i in range(6)],
            top_k=5,
            min_cluster_size=2,
        )

        result = await faq_analyze(request, settings, bedrock, postgres)

        assert result["status"] == "success"
        assert "candidates" in result
        assert isinstance(result["candidates"], list)


# ── 병렬화 회귀 테스트 (R8) ──


class TestParallelization:
    """답변 초안 생성이 병렬화 + 동시성 제한 + 부분 실패 처리되는지 검증."""

    async def test_invoke_llm_called_per_cluster_in_order(self):
        """클러스터당 답변+카테고리 2회 invoke_llm 호출 + 빈도순 정렬 보존 (B1)."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=8)
        postgres = _create_postgres()

        questions = [f"질문 {i}" for i in range(8)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=10,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        # 답변(answer) + 카테고리(light) = cluster 당 2회.
        assert bedrock.invoke_llm.call_count == len(candidates) * 2
        # 모델별 분리 단언으로 의미 명확
        call_list = bedrock.invoke_llm.call_args_list
        answer_count = sum(1 for c in call_list if c.kwargs.get("model") == "answer")
        light_count = sum(1 for c in call_list if c.kwargs.get("model") == "light")
        assert answer_count == len(candidates)
        assert light_count == len(candidates)

        # 정렬 검증이 trivially pass 하지 않도록 cluster 2개 이상 보장
        assert len(candidates) >= 2, "정렬 검증 의미 있으려면 cluster 2개 이상 필요"
        # 빈도 내림차순 보존 (gather 결과 매핑이 순서를 깨뜨리지 않음)
        for i in range(len(candidates) - 1):
            assert candidates[i]["frequency"] >= candidates[i + 1]["frequency"]

    async def test_partial_failure_placeholder(self, caplog):
        """답변 1건 실패 시 placeholder + 나머지/카테고리 정상 (격리 검증, B1).

        B1 후 answer+light 둘 다 호출되므로 list side_effect 대신 함수로 model 분기.
        답변 첫 호출만 fail, 나머지 답변 + 모든 카테고리 정상.
        """
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        answer_calls_seen: list[int] = []

        async def _selective_invoke(*, system_prompt, messages, max_tokens, model="light"):
            if model == "answer":
                answer_calls_seen.append(1)
                if len(answer_calls_seen) == 1:
                    raise RuntimeError("Bedrock throttle")
                return ("정상 답변", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
            # model="light" — 카테고리 분류는 항상 정상 enum 반환
            return ("enrollment_status", TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6))

        bedrock.invoke_llm.side_effect = _selective_invoke

        questions = [f"질문 {i}" for i in range(6)]
        with caplog.at_level("WARNING", logger="app.services.faq_analyzer"):
            candidates = await analyze_faq(
                questions=questions,
                top_k=10,
                min_cluster_size=2,
                bedrock=bedrock,
                postgres=postgres,
                settings=settings,
            )

        failed = [c for c in candidates if c["draft_answer"].startswith("답변 초안 생성 실패")]
        succeeded = [c for c in candidates if not c["draft_answer"].startswith("답변 초안 생성 실패")]

        assert len(failed) == 1
        assert "RuntimeError" in failed[0]["draft_answer"]
        assert len(succeeded) >= 1
        # 격리 검증 — 답변 fail 이 카테고리 결과 오염 X (모든 candidate 의 category 는 정상)
        assert all(c["category"] == "enrollment_status" for c in candidates)
        assert any("FAQ 답변 초안 생성 실패" in rec.message for rec in caplog.records)

    async def test_concurrency_capped_by_semaphore(self):
        """FAQ_CONCURRENCY=1 시 직렬 강제 — Semaphore가 실제로 동시성 제한 적용."""
        settings = _create_settings()
        # FAQ_CONCURRENCY=1 → invoke_llm 동시 진입 정확히 1개로 강제
        object.__setattr__(settings, "FAQ_CONCURRENCY", 1)

        bedrock = _create_bedrock(n_questions=8)
        postgres = _create_postgres()

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
            postgres=postgres,
            settings=settings,
        )

        # 클러스터 2개 이상 형성돼야 동시성 제한이 의미 있음
        assert len(candidates) >= 2
        # FAQ_CONCURRENCY=1 → 직렬 강제 → 동시 진입 정확히 1
        assert max_in_flight == 1

    async def test_inner_cancelled_error_propagates(self):
        """inner task 의 CancelledError 는 caller 로 re-raise 됨 (응답 깨짐 방지).

        gather(return_exceptions=True) 가 CancelledError 도 결과로 wrap 하므로,
        BaseException 가드가 약화되면 draft_answer 자리에 CancelledError 객체가
        들어가 JSON 직렬화 시 응답 전체가 깨질 수 있음. 본 테스트가 회귀 잠금.

        B1 후 답변(answer) + 카테고리(light) 둘 다 호출되므로 model 인자로 분기.
        답변 첫 호출에서 cancel → caller 로 propagate 되어야 함.
        """
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        answer_calls_seen: list[int] = []

        async def _selective_invoke(*, system_prompt, messages, max_tokens, model="light"):
            if model == "answer":
                answer_calls_seen.append(1)
                if len(answer_calls_seen) == 1:
                    raise asyncio.CancelledError()
                return ("정상", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
            return ("enrollment_status", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        bedrock.invoke_llm.side_effect = _selective_invoke

        questions = [f"질문 {i}" for i in range(6)]
        with pytest.raises(asyncio.CancelledError):
            await analyze_faq(
                questions=questions,
                top_k=10,
                min_cluster_size=2,
                bedrock=bedrock,
                postgres=postgres,
                settings=settings,
            )


# ── B1 FAQ 카테고리 분류 ──


class TestCategoryClassification:
    """B1: FAQ 카테고리 분류 (`_classify_category` + `_parse_category` + 격리/회귀)."""

    async def test_candidates_have_category_field(self):
        """각 candidate 응답 dict 에 category 필드 + 9 enum 중 하나 (B1)."""
        from app.services.faq_analyzer import FAQ_CATEGORIES

        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        # _create_bedrock 의 default invoke_llm.return_value 가 enum 값이 아니라
        # _parse_category 가 "other" fallback. 9 enum 안 포함 자체만 검증.
        questions = [f"질문 {i}" for i in range(6)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=5,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        assert len(candidates) >= 1
        for c in candidates:
            assert "category" in c
            assert c["category"] in FAQ_CATEGORIES

    @pytest.mark.parametrize("category", [
        "academic", "graduation", "enrollment_status", "scholarship",
        "registration", "curriculum", "career", "event", "other",
    ])
    def test_parse_category_valid_enum(self, category):
        """9 enum 각각 정상 정규화 (parametrize) — 대소문자/공백 무관."""
        from app.services.faq_analyzer import _parse_category

        assert _parse_category(category) == category
        assert _parse_category(f"  {category.upper()}  ") == category

    def test_parse_category_invalid_returns_other(self, caplog):
        """비표준 응답 → 'other' fallback + warning 발동."""
        from app.services.faq_analyzer import _parse_category

        invalid_cases = [
            "Some Other",          # 대문자 + 공백 (정규화 후에도 매칭 X)
            "category: academic",  # prefix 포함
            "unknown",             # 9 enum 외
            "",                    # 빈 문자열
        ]
        with caplog.at_level("WARNING", logger="app.services.faq_analyzer"):
            for response in invalid_cases:
                assert _parse_category(response) == "other"

        # 매 invalid case 마다 warning 1건
        warning_count = sum(
            1 for r in caplog.records
            if "FAQ 카테고리 분류 응답이 유효하지 않음" in r.message
        )
        assert warning_count == len(invalid_cases)

    async def test_classify_failure_isolated_from_draft(self, caplog):
        """카테고리 분류 실패 시 'other' placeholder + 답변은 정상 (옵션 A 격리 검증)."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        # 카테고리(light) 모든 호출 fail, 답변(answer) 정상.
        async def _selective_invoke(*, system_prompt, messages, max_tokens, model="light"):
            if model == "light":
                raise RuntimeError("Bedrock throttle (카테고리)")
            return ("정상 답변", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        bedrock.invoke_llm.side_effect = _selective_invoke

        questions = [f"질문 {i}" for i in range(6)]
        with caplog.at_level("WARNING", logger="app.services.faq_analyzer"):
            candidates = await analyze_faq(
                questions=questions,
                top_k=10,
                min_cluster_size=2,
                bedrock=bedrock,
                postgres=postgres,
                settings=settings,
            )

        assert len(candidates) >= 1
        # 카테고리 fail → 모든 candidate 의 category 가 "other" 폴백
        assert all(c["category"] == "other" for c in candidates)
        # 답변은 격리되어 정상
        assert all(c["draft_answer"] == "정상 답변" for c in candidates)
        # 카테고리 분류 실패 warning 발동
        assert any("FAQ 카테고리 분류 실패" in r.message for r in caplog.records)

    async def test_classify_cancelled_error_propagates(self):
        """카테고리 분류 task 의 CancelledError 도 caller 로 re-raise (회귀 잠금)."""
        settings = _create_settings()
        bedrock = _create_bedrock(n_questions=6)
        postgres = _create_postgres()

        light_calls_seen: list[int] = []

        async def _selective_invoke(*, system_prompt, messages, max_tokens, model="light"):
            if model == "light":
                light_calls_seen.append(1)
                if len(light_calls_seen) == 1:
                    raise asyncio.CancelledError()
                return ("enrollment_status", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
            return ("정상", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        bedrock.invoke_llm.side_effect = _selective_invoke

        questions = [f"질문 {i}" for i in range(6)]
        with pytest.raises(asyncio.CancelledError):
            await analyze_faq(
                questions=questions,
                top_k=10,
                min_cluster_size=2,
                bedrock=bedrock,
                postgres=postgres,
                settings=settings,
            )

    async def test_semaphore_shared_between_draft_and_classify(self):
        """답변 + 카테고리가 같은 semaphore 공유 — 동시 호출 ≤ FAQ_CONCURRENCY.

        feedback-asyncmock-race-test-pattern 룰: AsyncMock 즉시 반환은 sequential
        실행이라 side_effect 안에 await asyncio.sleep 으로 yield point 강제.
        """
        settings = _create_settings()
        object.__setattr__(settings, "FAQ_CONCURRENCY", 2)

        bedrock = _create_bedrock(n_questions=8)
        postgres = _create_postgres()

        in_flight = 0
        max_in_flight = 0

        async def _slow_invoke(*, model, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                # yield point — 다른 coroutine 이 semaphore 받을 기회 확보
                await asyncio.sleep(0.01)
                if model == "light":
                    return ("enrollment_status", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
                return ("답변", TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
            finally:
                in_flight -= 1

        bedrock.invoke_llm.side_effect = _slow_invoke

        questions = [f"질문 {i}" for i in range(8)]
        candidates = await analyze_faq(
            questions=questions,
            top_k=10,
            min_cluster_size=2,
            bedrock=bedrock,
            postgres=postgres,
            settings=settings,
        )

        # 클러스터 2개 이상 형성 → 답변 + 카테고리 합쳐 동시 호출 4건 이상.
        # FAQ_CONCURRENCY=2 → 답변/카테고리 무관 동시 호출 ≤ 2.
        assert len(candidates) >= 2
        assert max_in_flight == 2, f"동시 호출이 {max_in_flight} — semaphore 공유 깨짐"
