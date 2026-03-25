"""질문 재작성 모듈. 대화 히스토리 기반으로 후속 질문을 독립 문장으로 재작성한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.models.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
당신은 대화 맥락을 분석하여 후속 질문을 독립적인 문장으로 재작성하는 전문가입니다.

## 규칙
1. 대화 히스토리와 현재 질문을 분석하여, 대명사·생략된 주어·지시어를 원래 대상으로 복원하세요.
2. 원래 질문의 핵심 의도를 반드시 보존하세요. 의미를 추가하거나 변경하지 마세요.
3. 재작성된 질문만 출력하세요. 부연 설명이나 접두사를 붙이지 마세요.
"""


def _build_rewrite_messages(
    history: list[dict],
    question: str,
) -> list[dict]:
    """LLM에 전달할 메시지 배열을 구성한다."""
    conversation_lines: list[str] = []
    for msg in history:
        role_label = "사용자" if msg.get("role") == "user" else "어시스턴트"
        conversation_lines.append(f"{role_label}: {msg.get('content', '')}")

    user_text = (
        f"[대화 히스토리]\n"
        f"{chr(10).join(conversation_lines)}\n\n"
        f"[현재 질문]\n"
        f"{question}\n\n"
        f"위 대화 맥락을 반영하여 현재 질문을 독립적인 문장으로 재작성하세요."
    )

    return [{"role": "user", "content": [{"text": user_text}]}]


async def rewrite_query(
    context: PipelineContext,
    bedrock: BedrockClient,
) -> PipelineContext:
    """대화 히스토리가 있으면 질문을 재작성하고, 없으면 원본을 그대로 사용한다.

    - history가 비어있으면 원본 질문을 ``rewritten_question``에 그대로 복사.
    - history가 있으면 Claude 3 Haiku를 호출하여 독립 문장으로 재작성.
    - 토큰 사용량을 ``PipelineContext.token_usage``에 합산.
    """
    if not context.history:
        context.rewritten_question = context.original_question
        logger.debug("히스토리 없음 — 원본 질문 유지")
        return context

    messages = _build_rewrite_messages(context.history, context.original_question)

    rewritten, usage = await bedrock.invoke_llm(
        system_prompt=_REWRITE_SYSTEM_PROMPT,
        messages=messages,
        max_tokens=512,
    )

    cleaned = rewritten.strip()
    context.rewritten_question = cleaned or context.original_question
    context.token_usage.prompt_tokens += usage.prompt_tokens
    context.token_usage.completion_tokens += usage.completion_tokens
    context.token_usage.total_tokens += usage.total_tokens

    logger.info(
        "질문 재작성 완료: %r → %r (토큰: %d)",
        context.original_question,
        context.rewritten_question,
        usage.total_tokens,
    )

    return context
