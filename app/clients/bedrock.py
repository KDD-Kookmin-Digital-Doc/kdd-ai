"""AWS Bedrock 클라이언트 모듈. boto3 bedrock-runtime 사용."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import AsyncGenerator

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from app.config import Settings
from app.models.pipeline import TokenUsage

logger = logging.getLogger(__name__)

# ClientError 중 재시도 가능한 에러 코드 (throttling, 5xx 계열)
_RETRYABLE_ERROR_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
    "InternalServerException",
    "InternalFailure",
    "ServiceUnavailableException",
    "ModelNotReadyException",
})

# ClientError 외 무조건 재시도 대상인 예외
_RETRYABLE_EXCEPTIONS = (
    ReadTimeoutError,
    EndpointConnectionError,
    ConnectionError,
)


def _is_retryable(exc: Exception) -> bool:
    """재시도 가능한 예외인지 판별한다."""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return code in _RETRYABLE_ERROR_CODES or status >= 500
    return False


class BedrockClient:
    """AWS Bedrock API 클라이언트. boto3 bedrock-runtime 사용."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._light_model_id = settings.BEDROCK_LIGHT_MODEL_ID
        self._answer_model_id = settings.BEDROCK_ANSWER_MODEL_ID
        self._embedding_model_id = settings.BEDROCK_EMBEDDING_MODEL_ID
        self._max_retries = settings.BEDROCK_MAX_RETRIES

        llm_config = Config(
            region_name=settings.AWS_REGION,
            read_timeout=settings.BEDROCK_LLM_TIMEOUT,
            connect_timeout=settings.BEDROCK_LLM_TIMEOUT,
            retries={"max_attempts": 0},
        )
        embedding_config = Config(
            region_name=settings.AWS_REGION,
            read_timeout=settings.BEDROCK_EMBEDDING_TIMEOUT,
            connect_timeout=settings.BEDROCK_EMBEDDING_CONNECT_TIMEOUT,
            retries={"max_attempts": 0},
        )
        # D1: Cohere Rerank 3.5 는 Single-region only — 도쿄 클라이언트 분리.
        rerank_config = Config(
            region_name=settings.RERANK_REGION,
            read_timeout=settings.BEDROCK_RERANK_TIMEOUT,
            connect_timeout=settings.BEDROCK_RERANK_TIMEOUT,
            retries={"max_attempts": 0},
        )

        self._llm_client = boto3.client("bedrock-runtime", config=llm_config)
        self._embedding_client = boto3.client(
            "bedrock-runtime", config=embedding_config
        )
        self._rerank_client = boto3.client("bedrock-runtime", config=rerank_config)
        self.last_stream_usage: TokenUsage = TokenUsage()

    # ── 내부 헬퍼 ──

    async def _retry_async(self, sync_callable, *args, **kwargs):
        """동기 함수를 비동기로 실행하며 exponential backoff 재시도 (최대 2회)."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.to_thread(sync_callable, *args, **kwargs)
            except (ClientError, *_RETRYABLE_EXCEPTIONS) as exc:
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                if attempt < self._max_retries:
                    delay = 2**attempt  # 1초, 2초
                    logger.warning(
                        "Bedrock 호출 실패 (시도 %d/%d), %d초 후 재시도: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Bedrock 호출 최종 실패: %s", exc)
        raise last_exc  # type: ignore[misc]

    # ── 공개 메서드 ──

    async def invoke_llm_stream(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 1024,
        model: str = "light",
        usage_out: TokenUsage | None = None,
    ) -> AsyncGenerator[str, None]:
        """LLM 스트리밍 호출. 토큰 단위로 yield.

        Args:
            model: "light" (재작성/의도분류) 또는 "answer" (답변 생성).
            usage_out: 제공 시 토큰 사용량을 이 객체에 기록 (동시 요청 안전).
                       미제공 시 ``self.last_stream_usage``에 기록 (하위 호환).

        스트리밍 완료 후 토큰 사용량은 Bedrock 스트림의 마지막 metadata
        이벤트에서 수집된다.
        """
        self.last_stream_usage = TokenUsage()
        model_id = self._answer_model_id if model == "answer" else self._light_model_id

        response = await self._retry_async(
            self._llm_client.converse_stream,
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens},
        )

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()
        _sentinel = object()
        stop_event = threading.Event()

        def _read_stream() -> None:
            try:
                for event in response["stream"]:
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(event_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, _sentinel)

        read_future = loop.run_in_executor(None, _read_stream)

        try:
            while True:
                event = await event_queue.get()
                if event is _sentinel:
                    break
                if isinstance(event, Exception):
                    raise event
                if "contentBlockDelta" in event:
                    text = (
                        event["contentBlockDelta"].get("delta", {}).get("text")
                    )
                    if text:
                        yield text
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    inp = usage.get("inputTokens", 0)
                    out = usage.get("outputTokens", 0)
                    recorded = TokenUsage(
                        prompt_tokens=inp,
                        completion_tokens=out,
                        total_tokens=inp + out,
                    )
                    self.last_stream_usage = recorded
                    if usage_out is not None:
                        usage_out.prompt_tokens = recorded.prompt_tokens
                        usage_out.completion_tokens = recorded.completion_tokens
                        usage_out.total_tokens = recorded.total_tokens
        finally:
            stop_event.set()
            close_fn = getattr(response.get("stream"), "close", None)
            if callable(close_fn):
                await asyncio.to_thread(close_fn)
            await read_future

    async def invoke_llm(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 512,
        model: str = "light",
    ) -> tuple[str, TokenUsage]:
        """LLM 비스트리밍 호출. 질문 재작성, 의도 분류에 사용.

        Args:
            model: "light" (재작성/의도분류) 또는 "answer" (답변 생성).
        """
        model_id = self._answer_model_id if model == "answer" else self._light_model_id
        response = await self._retry_async(
            self._llm_client.converse,
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens},
        )

        text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        inp = usage.get("inputTokens", 0)
        out = usage.get("outputTokens", 0)

        return text, TokenUsage(
            prompt_tokens=inp,
            completion_tokens=out,
            total_tokens=inp + out,
        )

    async def embed_texts(
        self,
        texts: list[str],
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """Cohere Embed로 텍스트 배열을 벡터로 변환."""
        body = json.dumps(
            {
                "texts": texts,
                "input_type": input_type,
                "truncate": "NONE",
                "output_dimension": self._settings.EMBEDDING_DIMENSION,
            }
        )

        response = await self._retry_async(
            self._embedding_client.invoke_model,
            modelId=self._embedding_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        stream = response["body"]
        try:
            raw = await asyncio.to_thread(stream.read)
            response_body = json.loads(raw)
            embeddings = response_body["embeddings"]
            if isinstance(embeddings, dict):
                return embeddings["float"]
            return embeddings
        finally:
            await asyncio.to_thread(stream.close)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[tuple[int, float]]:
        """Cohere Rerank 3.5 호출 (도쿄 리전, D1, Task 13).

        Args:
            query: 검색 질의.
            documents: 임베딩 단계에서 추출된 후보 문서 텍스트 리스트.
            top_n: 상위 N개만 반환받기. 일반적으로 RERANK_TOP_N.

        Returns:
            [(index, relevance_score), ...]  — relevance_score 내림차순.
            index 는 ``documents`` 의 원본 인덱스라 호출자가 재정렬에 사용.
        """
        body = json.dumps(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_version": 2,
            }
        )

        response = await self._retry_async(
            self._rerank_client.invoke_model,
            modelId=self._settings.RERANK_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        stream = response["body"]
        try:
            raw = await asyncio.to_thread(stream.read)
            response_body = json.loads(raw)
            results = response_body.get("results", [])
            return [(r["index"], r["relevance_score"]) for r in results]
        finally:
            await asyncio.to_thread(stream.close)

    async def health_check_llm(self) -> bool:
        """LLM 서비스 연결 상태 확인 (light + answer 모델 모두 확인)."""
        try:
            for model_id in (self._light_model_id, self._answer_model_id):
                await asyncio.to_thread(
                    self._llm_client.converse,
                    modelId=model_id,
                    messages=[
                        {"role": "user", "content": [{"text": "ping"}]}
                    ],
                    inferenceConfig={"maxTokens": 1},
                )
            return True
        except Exception as exc:
            logger.warning("Bedrock LLM 헬스체크 실패: %s", exc)
            return False

    async def health_check_embedding(self) -> bool:
        """임베딩 서비스 연결 상태 확인."""
        try:
            body = json.dumps(
                {
                    "texts": ["ping"],
                    "input_type": "search_query",
                    "truncate": "NONE",
                }
            )
            response = await asyncio.to_thread(
                self._embedding_client.invoke_model,
                modelId=self._embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            await asyncio.to_thread(response["body"].read)
            await asyncio.to_thread(response["body"].close)
            return True
        except Exception as exc:
            logger.warning("Bedrock Embedding 헬스체크 실패: %s", exc)
            return False
