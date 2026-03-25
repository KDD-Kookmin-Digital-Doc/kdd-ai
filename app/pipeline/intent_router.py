"""의도 분류 모듈. 재작성된 질문을 학사규정 질문 또는 인사/잡담으로 분류한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.models.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_INTENT_SYSTEM_PROMPT = """\
당신은 대학교 학사규정 챗봇의 의도 분류기입니다.

## 규칙
1. 사용자 질문이 학사규정·학교 관련 질문이면 "academic"을 출력하세요.
2. 사용자 질문이 인사·잡담·학사규정과 무관한 질문이면 "chitchat"을 출력하세요.
3. "academic" 또는 "chitchat" 중 하나만 출력하세요. 다른 텍스트를 포함하지 마세요.
"""

_VALID_INTENTS = frozenset({"academic", "chitchat"})


async def classify_intent(
    context: PipelineContext,
    bedrock: BedrockClient,
) -> PipelineContext:
    """재작성된 질문(또는 원본)의 의도를 분류하여 PipelineContext에 설정한다.

    - Claude 3 Haiku를 호출하여 ``"academic"`` 또는 ``"chitchat"``으로 분류.
    - 분류 결과를 ``PipelineContext.intent``에 저장.
    - 토큰 사용량을 ``PipelineContext.token_usage``에 합산.
    - LLM 응답이 유효하지 않으면 ``"academic"``으로 기본 분류 (안전 측 폴백).
    """
    question = context.rewritten_question or context.original_question

    messages = [{"role": "user", "content": [{"text": question}]}]

    response, usage = await bedrock.invoke_llm(
        system_prompt=_INTENT_SYSTEM_PROMPT,
        messages=messages,
        max_tokens=16,
    )

    cleaned = response.strip()
    raw_intent = cleaned.lower()
    if raw_intent in _VALID_INTENTS:
        context.intent = raw_intent
    else:
        context.intent = "academic"
        logger.warning(
            "의도 분류 결과가 유효하지 않음 (%r) — academic으로 폴백",
            cleaned,
        )

    context.token_usage.prompt_tokens += usage.prompt_tokens
    context.token_usage.completion_tokens += usage.completion_tokens
    context.token_usage.total_tokens += usage.total_tokens

    logger.info(
        "의도 분류 완료: %r → %s (토큰: %d)",
        question,
        context.intent,
        usage.total_tokens,
    )

    return context
