"""Bedrock 클라이언트 단위 테스트. 모킹 기반."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from app.clients.bedrock import BedrockClient
from app.config import Settings
from app.models.pipeline import TokenUsage


def _create_settings(**overrides: str) -> Settings:
    """PR #62 헬퍼 패턴 — clear=True 로 셸/CI env leak 차단."""
    env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        **overrides,
    }
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)


@pytest.fixture
def settings():
    return _create_settings()


@pytest.fixture
def bedrock_client(settings):
    with patch("app.clients.bedrock.boto3.client"):
        client = BedrockClient(settings)
    client._llm_client = MagicMock()
    client._embedding_client = MagicMock()
    client._rerank_client = MagicMock()
    return client


def _make_client_error(code: str = "ThrottlingException", message: str = "error"):
    """재시도 가능한 ClientError 생성 (기본: ThrottlingException)."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}}, "TestOperation"
    )


def _make_non_retryable_error(code: str = "ValidationException", message: str = "bad input"):
    """재시도 불가능한 ClientError 생성."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}}, "TestOperation"
    )


# ── invoke_llm 테스트 ──


class TestInvokeLLM:
    async def test_success(self, bedrock_client):
        bedrock_client._llm_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "답변입니다"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        text, usage = await bedrock_client.invoke_llm(
            "시스템 프롬프트",
            [{"role": "user", "content": [{"text": "질문"}]}],
        )

        assert text == "답변입니다"
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    async def test_retry_on_failure_then_success(self, bedrock_client):
        bedrock_client._llm_client.converse.side_effect = [
            _make_client_error(),
            {
                "output": {"message": {"content": [{"text": "성공"}]}},
                "usage": {"inputTokens": 5, "outputTokens": 3},
            },
        ]

        with patch("asyncio.sleep"):
            text, _ = await bedrock_client.invoke_llm(
                "prompt",
                [{"role": "user", "content": [{"text": "q"}]}],
            )

        assert text == "성공"
        assert bedrock_client._llm_client.converse.call_count == 2

    async def test_retry_exhausted_raises(self, bedrock_client):
        bedrock_client._llm_client.converse.side_effect = _make_client_error()

        with patch("asyncio.sleep"):
            with pytest.raises(ClientError):
                await bedrock_client.invoke_llm(
                    "prompt",
                    [{"role": "user", "content": [{"text": "q"}]}],
                )

        assert bedrock_client._llm_client.converse.call_count == 3  # 1 + 2 retries

    async def test_http_5xx_triggers_retry(self, bedrock_client):
        """HTTP 5xx 상태 코드는 에러 코드명과 무관하게 재시도."""
        error = ClientError(
            {
                "Error": {"Code": "UnknownServerError", "Message": "unknown"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
            "TestOperation",
        )
        bedrock_client._llm_client.converse.side_effect = [
            error,
            {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
        ]

        with patch("asyncio.sleep"):
            text, _ = await bedrock_client.invoke_llm(
                "p", [{"role": "user", "content": [{"text": "q"}]}]
            )

        assert text == "ok"
        assert bedrock_client._llm_client.converse.call_count == 2

    async def test_non_retryable_error_raises_immediately(self, bedrock_client):
        """ValidationException 등 비재시도 에러는 즉시 raise."""
        bedrock_client._llm_client.converse.side_effect = _make_non_retryable_error()

        with pytest.raises(ClientError):
            await bedrock_client.invoke_llm(
                "prompt",
                [{"role": "user", "content": [{"text": "q"}]}],
            )

        assert bedrock_client._llm_client.converse.call_count == 1  # 재시도 없음

    async def test_timeout_triggers_retry(self, bedrock_client):
        bedrock_client._llm_client.converse.side_effect = [
            ReadTimeoutError(endpoint_url="https://bedrock.amazonaws.com"),
            {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
        ]

        with patch("asyncio.sleep"):
            text, _ = await bedrock_client.invoke_llm(
                "p", [{"role": "user", "content": [{"text": "q"}]}]
            )

        assert text == "ok"


# ── invoke_llm_stream 테스트 ──


class TestInvokeLLMStream:
    async def test_stream_tokens_and_usage(self, bedrock_client):
        stream_events = [
            {"contentBlockDelta": {"delta": {"text": "안녕"}}},
            {"contentBlockDelta": {"delta": {"text": "하세요"}}},
            {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 8}}},
        ]
        bedrock_client._llm_client.converse_stream.return_value = {
            "stream": iter(stream_events)
        }

        tokens = []
        async for token in bedrock_client.invoke_llm_stream(
            "system", [{"role": "user", "content": [{"text": "hi"}]}]
        ):
            tokens.append(token)

        assert tokens == ["안녕", "하세요"]
        assert bedrock_client.last_stream_usage.prompt_tokens == 20
        assert bedrock_client.last_stream_usage.completion_tokens == 8
        assert bedrock_client.last_stream_usage.total_tokens == 28

    async def test_stream_empty(self, bedrock_client):
        bedrock_client._llm_client.converse_stream.return_value = {
            "stream": iter([
                {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 0}}}
            ])
        }

        tokens = []
        async for token in bedrock_client.invoke_llm_stream(
            "system", [{"role": "user", "content": [{"text": "q"}]}]
        ):
            tokens.append(token)

        assert tokens == []
        assert bedrock_client.last_stream_usage.prompt_tokens == 5

    async def test_stream_resets_usage(self, bedrock_client):
        """각 스트리밍 호출 시 last_stream_usage가 초기화되는지 확인."""
        bedrock_client.last_stream_usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        bedrock_client._llm_client.converse_stream.return_value = {
            "stream": iter([
                {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}
            ])
        }

        async for _ in bedrock_client.invoke_llm_stream(
            "s", [{"role": "user", "content": [{"text": "q"}]}]
        ):
            pass

        assert bedrock_client.last_stream_usage.prompt_tokens == 1

    async def test_stream_retry_on_initial_call(self, bedrock_client):
        bedrock_client._llm_client.converse_stream.side_effect = [
            _make_client_error(),
            {"stream": iter([
                {"contentBlockDelta": {"delta": {"text": "ok"}}},
                {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
            ])},
        ]

        with patch("asyncio.sleep"):
            tokens = []
            async for token in bedrock_client.invoke_llm_stream(
                "s", [{"role": "user", "content": [{"text": "q"}]}]
            ):
                tokens.append(token)

        assert tokens == ["ok"]
        assert bedrock_client._llm_client.converse_stream.call_count == 2


# ── embed_texts 테스트 ──


class TestEmbedTexts:
    async def test_success(self, bedrock_client):
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {"embeddings": embeddings}
        ).encode()
        bedrock_client._embedding_client.invoke_model.return_value = {
            "body": mock_body
        }

        result = await bedrock_client.embed_texts(["텍스트1", "텍스트2"])

        assert len(result) == 2
        assert len(result[0]) == 1024
        assert result[0][0] == 0.1

    async def test_input_type_search_query(self, bedrock_client):
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {"embeddings": [[0.5] * 1024]}
        ).encode()
        bedrock_client._embedding_client.invoke_model.return_value = {
            "body": mock_body
        }

        await bedrock_client.embed_texts(["질문"], input_type="search_query")

        call_kwargs = bedrock_client._embedding_client.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs.get("body", call_kwargs[1].get("body", "")))
        assert body["input_type"] == "search_query"

    async def test_retry_on_failure(self, bedrock_client):
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {"embeddings": [[0.1] * 1024]}
        ).encode()
        bedrock_client._embedding_client.invoke_model.side_effect = [
            _make_client_error(),
            {"body": mock_body},
        ]

        with patch("asyncio.sleep"):
            result = await bedrock_client.embed_texts(["텍스트"])

        assert len(result) == 1


# ── 헬스체크 테스트 ──


class TestHealthCheck:
    async def test_llm_healthy(self, bedrock_client):
        bedrock_client._llm_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }

        assert await bedrock_client.health_check_llm() is True

    async def test_llm_unhealthy(self, bedrock_client):
        bedrock_client._llm_client.converse.side_effect = _make_client_error()

        assert await bedrock_client.health_check_llm() is False

    async def test_embedding_healthy(self, bedrock_client):
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {"embeddings": [[0.1] * 1024]}
        ).encode()
        bedrock_client._embedding_client.invoke_model.return_value = {
            "body": mock_body
        }

        assert await bedrock_client.health_check_embedding() is True

    async def test_embedding_unhealthy(self, bedrock_client):
        bedrock_client._embedding_client.invoke_model.side_effect = (
            _make_client_error()
        )

        assert await bedrock_client.health_check_embedding() is False


# ── 클라이언트 초기화 (타임아웃 분리) ──


class TestClientInitialization:
    """임베딩 boto Config의 read/connect 타임아웃이 분리 적용되는지."""

    def test_embedding_read_and_connect_timeout_separated(self):
        settings = _create_settings(
            BEDROCK_EMBEDDING_TIMEOUT="30",
            BEDROCK_EMBEDDING_CONNECT_TIMEOUT="10",
        )

        with patch("app.clients.bedrock.boto3.client") as mock_client:
            BedrockClient(settings)

        # D1 이후 boto3.client 는 LLM/임베딩/rerank 3회 호출
        assert mock_client.call_count == 3
        embedding_call = mock_client.call_args_list[1]
        embedding_config = embedding_call.kwargs["config"]

        assert embedding_config.read_timeout == 30
        assert embedding_config.connect_timeout == 10

    def test_llm_timeout_unchanged_by_embedding_split(self):
        """LLM 클라이언트는 read=connect로 통합 유지."""
        settings = _create_settings(BEDROCK_LLM_TIMEOUT="45")

        with patch("app.clients.bedrock.boto3.client") as mock_client:
            BedrockClient(settings)

        llm_call = mock_client.call_args_list[0]
        llm_config = llm_call.kwargs["config"]

        assert llm_config.read_timeout == 45
        assert llm_config.connect_timeout == 45

    def test_rerank_client_uses_tokyo_region(self):
        """D1: Cohere Rerank 3.5 는 Single-region — 도쿄 클라이언트 분리 검증."""
        settings = _create_settings(
            AWS_REGION="ap-northeast-2",
            BEDROCK_RERANK_TIMEOUT="15",
        )

        with patch("app.clients.bedrock.boto3.client") as mock_client:
            BedrockClient(settings)

        # 3번째 호출이 rerank client (LLM → Embedding → Rerank 순)
        rerank_call = mock_client.call_args_list[2]
        rerank_config = rerank_call.kwargs["config"]

        assert rerank_config.region_name == "ap-northeast-1"
        assert rerank_config.read_timeout == 15
        assert rerank_config.connect_timeout == 15


# ── rerank 테스트 (D1, Task 13) ──


class TestRerank:
    async def test_success_returns_index_score_tuples(self, bedrock_client):
        """정상 응답을 [(index, relevance_score), ...] 로 파싱."""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.97},
                    {"index": 0, "relevance_score": 0.85},
                    {"index": 1, "relevance_score": 0.12},
                ]
            }
        ).encode()
        bedrock_client._rerank_client.invoke_model.return_value = {"body": mock_body}

        result = await bedrock_client.rerank(
            query="휴학 절차",
            documents=["문서A", "문서B", "문서C"],
            top_n=3,
        )

        assert result == [(2, 0.97), (0, 0.85), (1, 0.12)]

    async def test_request_body_shape(self, bedrock_client):
        """body 에 query/documents/top_n/api_version=2 가 포함된다."""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"results": []}).encode()
        bedrock_client._rerank_client.invoke_model.return_value = {"body": mock_body}

        await bedrock_client.rerank(
            query="질문", documents=["A", "B"], top_n=2
        )

        call_kwargs = bedrock_client._rerank_client.invoke_model.call_args.kwargs
        body = json.loads(call_kwargs["body"])
        assert body["query"] == "질문"
        assert body["documents"] == ["A", "B"]
        assert body["top_n"] == 2
        assert body["api_version"] == 2
        assert call_kwargs["modelId"] == "cohere.rerank-v3-5:0"

    async def test_uses_rerank_client_not_llm_or_embedding(self, bedrock_client):
        """도쿄 분리 — _llm_client/_embedding_client 는 호출되지 않음."""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"results": []}).encode()
        bedrock_client._rerank_client.invoke_model.return_value = {"body": mock_body}

        await bedrock_client.rerank(query="q", documents=["d"], top_n=1)

        bedrock_client._rerank_client.invoke_model.assert_called_once()
        bedrock_client._llm_client.invoke_model.assert_not_called()
        bedrock_client._embedding_client.invoke_model.assert_not_called()

    async def test_retry_on_throttle_then_success(self, bedrock_client):
        """ThrottlingException 1회 후 성공 — _retry_async 공유 검증."""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {"results": [{"index": 0, "relevance_score": 0.9}]}
        ).encode()
        bedrock_client._rerank_client.invoke_model.side_effect = [
            _make_client_error(),
            {"body": mock_body},
        ]

        with patch("asyncio.sleep"):
            result = await bedrock_client.rerank(
                query="q", documents=["d"], top_n=1
            )

        assert result == [(0, 0.9)]
        assert bedrock_client._rerank_client.invoke_model.call_count == 2

    async def test_non_retryable_raises_immediately(self, bedrock_client):
        """ValidationException 등 비재시도 에러는 즉시 raise."""
        bedrock_client._rerank_client.invoke_model.side_effect = (
            _make_non_retryable_error()
        )

        with pytest.raises(ClientError):
            await bedrock_client.rerank(query="q", documents=["d"], top_n=1)

        assert bedrock_client._rerank_client.invoke_model.call_count == 1

    async def test_retry_exhausted_raises(self, bedrock_client):
        """재시도 끝까지 throttle → 최종 raise (vector_search 가 잡고 fallback)."""
        bedrock_client._rerank_client.invoke_model.side_effect = _make_client_error()

        with patch("asyncio.sleep"):
            with pytest.raises(ClientError):
                await bedrock_client.rerank(
                    query="q", documents=["d"], top_n=1
                )

        assert bedrock_client._rerank_client.invoke_model.call_count == 3  # 1 + 2 retries

    async def test_empty_results_returns_empty_list(self, bedrock_client):
        """results=[] 응답 — Cohere 가 빈 결과 반환 시 [] 반환."""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"results": []}).encode()
        bedrock_client._rerank_client.invoke_model.return_value = {"body": mock_body}

        result = await bedrock_client.rerank(
            query="q", documents=["A", "B"], top_n=2
        )

        assert result == []
