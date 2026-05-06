"""LLM 메시지 dict 구성 헬퍼 (PR-R4).

Bedrock Converse API 의 ``content`` 는 ``[{"text": "..."}, ...]`` 블록 배열
형태이다. 파이프라인 여러 곳에서 동일한 dict 구조를 수동으로 만들어 반복이
발생해 헬퍼로 통합한다.
"""

from __future__ import annotations


def make_text_content(text: str) -> list[dict]:
    """Bedrock content 블록 한 개를 만든다."""
    return [{"text": text}]


def make_user_message(text: str) -> dict:
    """``role=user`` 메시지 한 개를 만든다."""
    return {"role": "user", "content": make_text_content(text)}


def make_assistant_message(text: str) -> dict:
    """``role=assistant`` 메시지 한 개를 만든다 (history 재구성용)."""
    return {"role": "assistant", "content": make_text_content(text)}
