"""Bedrock 클라이언트 단위 테스트. 모킹 기반."""

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from app.clients.bedrock import BedrockClient
from app.config import Settings
from app.models.pipeline import TokenUsage


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    return Settings(_env_file=None)


@pytest.fixture
def bedrock_client(settings):
    with patch("app.clients.bedrock.boto3.client"):
        client = BedrockClient(settings)
    client._llm_client = MagicMock()
    client._embedding_client = MagicMock()
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
            text, usage = await bedrock_client.invoke_llm(
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
