from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.request import ChatRequest
from app.services.cache import check_cache, update_cache
from app.services.vector_db import search_documents
from app.services.llm import rag_chain, multiturn_rag_chain
from app.services.memory import (
    add_user_message, add_ai_message,
    should_summarize, summarize_and_compress,
    build_chat_context,
)

router = APIRouter()


@router.post(
    "/chat",
    summary="학사규정 질의응답 (Streaming)",
    description="사용자의 질문을 받아 RAG 파이프라인을 거친 후 SSE 스트리밍으로 반환합니다. "
                "session_id를 포함하면 멀티턴 대화가 가능합니다.",
    responses={
        200: {
            "description": "SSE 스트리밍 응답 (완료 시 `[DONE]` 전송)",
            "content": {
                "text/event-stream": {
                    "example": "data: 국민대학교 학사규정에 따르면...\n\ndata: [DONE]\n\n"
                }
            }
        }
    }
)
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    session_id = req.session_id

    # 1. 시맨틱 캐시 확인 (싱글턴 대화에서만 적용)
    if not session_id:
        cached_answer = check_cache(user_msg)
        if cached_answer:
            async def cache_streamer():
                yield "data: [캐시된 답변입니다 ⚡]\n\n"
                yield f"data: {cached_answer}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cache_streamer(), media_type="text/event-stream")

    # 2. 경량 라우터 (짧은 인사말은 LLM 호출 생략)
    greetings = ["안녕", "반가워", "누구야", "고마워", "하이"]
    if len(user_msg) < 4 and any(g in user_msg for g in greetings):
        async def greeting_streamer():
            yield "data: 안녕하세요! 국민대학교 학사규정 챗봇입니다. 무엇을 도와드릴까요?\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(greeting_streamer(), media_type="text/event-stream")

    # 3. Vector DB 검색
    print("[ROUTER] 규정 검색 진행 중...")
    context_text = search_documents(user_msg)

    # 4. 멀티턴 vs 싱글턴 분기
    if session_id:
        return await _handle_multiturn(session_id, user_msg, context_text)
    else:
        return await _handle_single(user_msg, context_text)


async def _handle_single(user_msg: str, context_text: str):
    """싱글턴 대화: 기존 방식 그대로"""
    async def streamer():
        full_answer = ""
        for chunk in rag_chain.stream({"context": context_text, "question": user_msg}):
            text_chunk = chunk.content
            full_answer += text_chunk
            yield f"data: {text_chunk}\n\n"
        update_cache(user_msg, full_answer)
        yield "data: [DONE]\n\n"
    return StreamingResponse(streamer(), media_type="text/event-stream")


async def _handle_multiturn(session_id: str, user_msg: str, context_text: str):
    """멀티턴 대화: 히스토리 관리 + 요약 압축"""
    # 히스토리에 사용자 메시지 추가
    add_user_message(session_id, user_msg)

    # 요약이 필요하면 실행
    if should_summarize(session_id):
        summarize_and_compress(session_id)

    # 대화 컨텍스트 구성
    chat_history = build_chat_context(session_id)

    async def streamer():
        full_answer = ""
        for chunk in multiturn_rag_chain.stream({
            "context": context_text,
            "chat_history": chat_history,
            "question": user_msg,
        }):
            text_chunk = chunk.content
            full_answer += text_chunk
            yield f"data: {text_chunk}\n\n"

        # AI 응답을 히스토리에 저장
        add_ai_message(session_id, full_answer)
        yield "data: [DONE]\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")
