"""의도 분류 모듈. 현재 메시지(원본)를 학사규정 질문 또는 인사/잡담으로 분류한다.

rewrite 이전에 호출되며, history 톤이 잡담을 학사 톤으로 비트는 회귀를 차단한다.
"""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.config import Settings
from app.models.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_INTENT_SYSTEM_PROMPT = """\
당신은 대학교 학사규정 챗봇의 의도 분류기입니다.

## 규칙
1. 분류 대상은 **현재 메시지** 하나입니다. 직전 대화 히스토리는 보조 참고일 뿐이며, 현재 메시지의 의도가 우선합니다.
2. 현재 메시지가 학사·학교 운영(학점·휴학·복학·수강·졸업·전공·학사일정·등록·장학·전과 등)을 명시적으로 묻거나, 직전 학사 답변에 대한 명백한 후속 질문이면 "academic"을 출력하세요.
3. 그 외 인사·감탄·일상 잡담·감사 표현은 "chitchat"을 출력하세요. 학사 대화 히스토리가 있어도 현재 메시지가 잡담이면 chitchat입니다.
4. "academic" 또는 "chitchat" 중 하나만 출력하세요. 다른 텍스트를 포함하지 마세요.

## 예시
- "오늘 날씨 좋네요" → chitchat
- "고마워요" → chitchat
- "ㅋㅋㅋ" → chitchat
- "오늘 점심 뭐 먹지" → chitchat
- "휴학 신청 기간 언제예요?" → academic
- (직전 답변: 휴학은 4년) "그럼 복학은 어떻게 해?" → academic
"""

_VALID_INTENTS = frozenset({"academic", "chitchat"})


def _build_classify_user_message(
    context: PipelineContext, settings: Settings
) -> str:
    """분류 LLM에 전달할 user 메시지를 구성한다.

    직전 ``settings.INTENT_HISTORY_TURNS``개 메시지를 짧게 첨부하되 분류 대상은
    항상 ``context.original_question``.
    """
    parts: list[str] = []

    if context.user_context:
        parts.append(f"[사용자 정보]\n{context.user_context}")

    history = context.history or []
    if history:
        recent = history[-settings.INTENT_HISTORY_TURNS:]
        char_limit = settings.INTENT_HISTORY_CHARS_PER_TURN
        lines: list[str] = []
        for msg in recent:
            role = "사용자" if msg.get("role") == "user" else "어시스턴트"
            content = (msg.get("content") or "")[:char_limit]
            lines.append(f"{role}: {content}")
        parts.append("[직전 대화 (참고용)]\n" + "\n".join(lines))

    parts.append(
        f"[분류 대상 — 현재 메시지]\n{context.original_question}\n\n"
        "위 현재 메시지의 의도를 academic 또는 chitchat 중 하나로 출력하세요."
    )
    return "\n\n".join(parts)


async def classify_intent(
    context: PipelineContext,
    bedrock: BedrockClient,
    settings: Settings,
) -> PipelineContext:
    """현재 메시지(원본)의 의도를 history-aware하게 분류한다.

    - 분류 대상: ``context.original_question`` (rewrite 결과 아님 — history 톤이
      잡담을 학사 톤으로 비트는 회귀를 차단).
    - 직전 ``settings.INTENT_HISTORY_TURNS``개 메시지를 보조 컨텍스트로만 첨부,
      각 메시지는 ``settings.INTENT_HISTORY_CHARS_PER_TURN``자로 절단.
    - 분류 결과를 ``PipelineContext.intent``에 저장.
    - 토큰 사용량을 ``PipelineContext.token_usage``에 합산.
    - LLM 응답이 유효하지 않으면 ``"academic"``으로 기본 분류 (안전 측 폴백).
    """
    user_text = _build_classify_user_message(context, settings)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

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
        "의도 분류 완료: %r → %s (토큰: %d, history=%d)",
        context.original_question,
        context.intent,
        usage.total_tokens,
        len(context.history or []),
    )

    return context
