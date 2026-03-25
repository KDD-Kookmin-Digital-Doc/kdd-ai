"""SSE 스트리밍 모듈 테스트. 속성 기반 테스트 + 단위 테스트."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.config import Settings
from app.models.pipeline import (
    PipelineContext,
    SearchResult,
    SourceDoc,
    TokenUsage,
)
from app.streaming.sse import (
    _determine_confidence,
    _format_sse_event,
    stream_sse_response,
)


# ── 헬퍼 ──


def _create_settings() -> Settings:
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


def _create_bedrock(tokens: list[str] | None = None) -> AsyncMock:
    bedrock = AsyncMock()
    bedrock.last_stream_usage = TokenUsage()
    return bedrock


def _make_search_result(
    similarity: float = 0.85,
    doc_id: str = "doc-1",
    doc_name: str = "학사요람.pdf",
    page: int = 10,
) -> SearchResult:
    return SearchResult(
        id=1,
        doc_id=doc_id,
        content="제1조 내용",
        metadata={"doc_name": doc_name, "page": page},
        similarity_score=similarity,
    )


def _make_source_doc(
    doc_id: str = "doc-1",
    doc_name: str = "학사요람.pdf",
    page: int = 10,
) -> SourceDoc:
    return SourceDoc(doc_id=doc_id, doc_name=doc_name, page=page)


async def _collect_chunks(async_gen) -> list[dict]:
    """SSE 스트림에서 청크를 수집하여 파싱된 dict 리스트로 반환."""
    chunks = []
    async for event in async_gen:
        data_str = event.removeprefix("data: ").strip()
        if data_str:
            chunks.append(json.loads(data_str))
    return chunks


def _mock_generate_response(tokens: list[str]):
    """generate_response를 모킹하는 async generator를 반환."""
    async def _mock(context, bedrock, settings):
        for t in tokens:
            yield t
        context.token_usage.prompt_tokens += 100
        context.token_usage.completion_tokens += 20
        context.token_usage.total_tokens += 120
    return _mock


# ── Property 7: SSE 스트림 구조 정합성 ──
# Feature: rag-chatbot-ai-server, Property 7: SSE 스트림 구조 정합성
# Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.7, 5.8


class TestSSEStreamStructure:
    """Property 7: 시나리오별 올바른 청크 시퀀스 검증."""

    @patch("app.streaming.sse.generate_response")
    async def test_scenario_a_normal(self, mock_gen):
        """시나리오 A: meta(document) → text* → done."""
        mock_gen.return_value = _mock_generate_response(["답변", "입니다"])("", "", "")
        mock_gen.side_effect = _mock_generate_response(["답변", "입니다"])

        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[_make_source_doc()],
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        assert chunks[0]["type"] == "meta"
        assert chunks[0]["subtype"] == "document"
        assert "confidence" in chunks[0]
        assert "sources" in chunks[0]
        assert all(c["type"] == "text" for c in chunks[1:-1])
        assert chunks[-1]["type"] == "done"
        assert "usage" in chunks[-1]

    async def test_scenario_b_fallback(self):
        """시나리오 B: fallback → done."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[],
            suggested_questions=["유사 질문 1", "유사 질문 2"],
        )
        ctx.token_usage.prompt_tokens = 50
        ctx.token_usage.total_tokens = 50

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        assert len(chunks) == 2
        assert chunks[0]["type"] == "fallback"
        assert "suggested_questions" in chunks[0]
        assert chunks[1]["type"] == "done"

    async def test_scenario_c_cache_hit(self):
        """시나리오 C: meta(cache) → text → done."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="질문",
            cache_hit=True,
            cached_answer="캐시된 답변입니다.",
            cached_sources=[_make_source_doc()],
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        assert len(chunks) == 3
        assert chunks[0]["type"] == "meta"
        assert chunks[0]["subtype"] == "cache"
        assert chunks[0]["cache_hit"] is True
        assert chunks[1]["type"] == "text"
        assert chunks[1]["content"] == "캐시된 답변입니다."
        assert chunks[2]["type"] == "done"
        assert chunks[2]["usage"]["total_tokens"] == 0

    @patch("app.streaming.sse.generate_response")
    async def test_scenario_d_chitchat(self, mock_gen):
        """시나리오 D: meta(chitchat) → text* → done."""
        mock_gen.side_effect = _mock_generate_response(["안녕", "하세요"])

        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="안녕",
            intent="chitchat",
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        assert chunks[0]["type"] == "meta"
        assert chunks[0]["subtype"] == "chitchat"
        assert chunks[0]["intent"] == "chitchat"
        assert all(c["type"] == "text" for c in chunks[1:-1])
        assert chunks[-1]["type"] == "done"

    @patch("app.streaming.sse.generate_response")
    async def test_error_scenario(self, mock_gen):
        """에러 시나리오: error 청크 전송 후 종료 (done 없음)."""
        async def _failing_gen(*args, **kwargs):
            raise RuntimeError("Bedrock 장애")
            yield  # noqa: unreachable — make it a generator

        mock_gen.side_effect = _failing_gen

        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="질문",
            intent="academic",
            search_results=[_make_search_result()],
            source_docs=[_make_source_doc()],
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        # meta 청크 후 error
        error_chunks = [c for c in chunks if c["type"] == "error"]
        assert len(error_chunks) == 1
        assert "message" in error_chunks[0]
        # done 청크가 없어야 함
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 0

    async def test_all_scenarios_end_with_done_except_error(self):
        """모든 정상 시나리오에서 마지막 청크는 done이다."""
        settings = _create_settings()
        bedrock = _create_bedrock()

        # 캐시 히트
        ctx_cache = PipelineContext(
            original_question="q",
            cache_hit=True,
            cached_answer="a",
            cached_sources=[_make_source_doc()],
        )
        chunks = await _collect_chunks(
            stream_sse_response(ctx_cache, bedrock, settings)
        )
        assert chunks[-1]["type"] == "done"

        # 폴백
        ctx_fallback = PipelineContext(
            original_question="q",
            intent="academic",
            search_results=[],
        )
        chunks = await _collect_chunks(
            stream_sse_response(ctx_fallback, bedrock, settings)
        )
        assert chunks[-1]["type"] == "done"


# ── Property 8: 출처 정보 보존 ──
# Feature: rag-chatbot-ai-server, Property 8: 출처 정보 보존
# Validates: Requirements 4.4


class TestSourceInfoPreservation:
    """Property 8: sources 배열의 doc_name, page 필드 존재 및 메타데이터 일치."""

    @hyp_settings(max_examples=30)
    @given(
        doc_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        page=st.integers(min_value=1, max_value=1000),
        doc_id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    async def test_cache_hit_sources_preserved(self, doc_name, page, doc_id):
        """캐시 히트 시 sources에 doc_id, doc_name, page가 보존된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="q",
            cache_hit=True,
            cached_answer="a",
            cached_sources=[SourceDoc(doc_id=doc_id, doc_name=doc_name, page=page)],
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        meta = chunks[0]
        assert len(meta["sources"]) == 1
        src = meta["sources"][0]
        assert src["doc_id"] == doc_id
        assert src["doc_name"] == doc_name
        assert src["page"] == page

    @patch("app.streaming.sse.generate_response")
    async def test_normal_sources_preserved(self, mock_gen):
        """정상 응답 시 sources에 doc_id, doc_name, page가 보존된다."""
        mock_gen.side_effect = _mock_generate_response(["답변"])

        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="q",
            intent="academic",
            search_results=[_make_search_result(
                doc_id="doc-99", doc_name="학칙.pdf", page=42,
            )],
            source_docs=[_make_source_doc(
                doc_id="doc-99", doc_name="학칙.pdf", page=42,
            )],
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        meta = chunks[0]
        src = meta["sources"][0]
        assert src["doc_id"] == "doc-99"
        assert src["doc_name"] == "학칙.pdf"
        assert src["page"] == 42

    async def test_multiple_sources(self):
        """여러 출처가 모두 보존된다."""
        settings = _create_settings()
        bedrock = _create_bedrock()
        ctx = PipelineContext(
            original_question="q",
            cache_hit=True,
            cached_answer="a",
            cached_sources=[
                SourceDoc(doc_id="d1", doc_name="a.pdf", page=1),
                SourceDoc(doc_id="d2", doc_name="b.pdf", page=2),
                SourceDoc(doc_id="d3", doc_name="c.pdf", page=3),
            ],
        )

        chunks = await _collect_chunks(
            stream_sse_response(ctx, bedrock, settings)
        )

        assert len(chunks[0]["sources"]) == 3


# ── confidence 단위 테스트 ──


class TestDetermineConfidence:
    def test_high_confidence(self):
        settings = _create_settings()
        results = [_make_search_result(similarity=0.95)]
        assert _determine_confidence(results, settings) == "high"

    def test_medium_confidence(self):
        settings = _create_settings()
        results = [_make_search_result(similarity=0.85)]
        assert _determine_confidence(results, settings) == "medium"

    def test_low_confidence(self):
        settings = _create_settings()
        results = [_make_search_result(similarity=0.76)]
        assert _determine_confidence(results, settings) == "low"

    def test_empty_results(self):
        settings = _create_settings()
        assert _determine_confidence([], settings) == "low"

    def test_uses_max_score(self):
        """여러 결과 중 최고 유사도로 판단한다."""
        settings = _create_settings()
        results = [
            _make_search_result(similarity=0.76),
            _make_search_result(similarity=0.92),
            _make_search_result(similarity=0.80),
        ]
        assert _determine_confidence(results, settings) == "high"

    def test_exact_boundary_high(self):
        """정확히 high 임계값이면 high이다."""
        settings = _create_settings()
        results = [_make_search_result(similarity=0.9)]
        assert _determine_confidence(results, settings) == "high"

    def test_exact_boundary_medium(self):
        """정확히 medium 임계값이면 medium이다."""
        settings = _create_settings()
        results = [_make_search_result(similarity=0.8)]
        assert _determine_confidence(results, settings) == "medium"

    def test_custom_thresholds(self):
        """settings에서 커스텀 임계값을 사용한다."""
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "CONFIDENCE_HIGH_THRESHOLD": "0.95",
            "CONFIDENCE_MEDIUM_THRESHOLD": "0.85",
        }):
            settings = Settings(_env_file=None)
            results = [_make_search_result(similarity=0.9)]
            assert _determine_confidence(results, settings) == "medium"


# ── SSE 포맷 테스트 ──


class TestFormatSSEEvent:
    def test_format(self):
        result = _format_sse_event({"type": "text", "content": "안녕"})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        parsed = json.loads(result.replace("data: ", "").strip())
        assert parsed["content"] == "안녕"

    def test_korean_not_escaped(self):
        """한국어가 유니코드 이스케이프 없이 출력된다."""
        result = _format_sse_event({"content": "한국어"})
        assert "한국어" in result
        assert "\\u" not in result
