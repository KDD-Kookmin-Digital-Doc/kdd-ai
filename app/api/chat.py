from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.request import ChatRequest
from app.services.cache import check_cache, update_cache
from app.services.vector_db import search_documents
from app.services.llm import rag_chain
#Spring Boot 서버의 요청을 받아 캐시 -> DB -> LLM 순서로 제어
router = APIRouter()

@router.post(
    "/chat",
    summary="학사규정 질의응답 (Streaming)",
    description="사용자의 질문을 받아 RAG 파이프라인(Vector DB 검색 + LLM)을 거친 후, 답변을 실시간 SSE(Server-Sent Events) 스트리밍 방식으로 반환합니다.",
    responses={
        200: {
            "description": "정상적인 SSE 스트리밍 응답 (응답 완료 시 `[DONE]` 전송)",
            "content": {
                "text/event-stream": {
                    "example": "data: 국민\n\ndata: 대학교 학사규정\n\ndata: 제22조에 따르면,\n\ndata: 휴학은 통산 4년(8학기)을 초과할 수 없습니다.\n\ndata: [DONE]\n\n"
                }
            }
        }
    }
)
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()

    # 1. 로컬 캐시 확인
    cached_answer = check_cache(user_msg)
    if cached_answer:
        async def cache_streamer():
            yield f"data: [캐시된 답변입니다 ⚡]\n\n"
            yield f"data: {cached_answer}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cache_streamer(), media_type="text/event-stream")

    # 2. 경량 라우터 (일상 대화시 LLM 호출x)
    greetings = ["안녕", "반가워", "누구야", "고마워", "하이"]
    if len(user_msg) < 4 and any(greet in user_msg for greet in greetings):
        async def greeting_streamer():
            yield "data: 안녕하세요! 국민대학교 학사규정 챗봇입니다. 무엇을 도와드릴까요?\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(greeting_streamer(), media_type="text/event-stream")

    # 3. Vector DB 검색
    print("[ROUTER] 규정 검색 진행 중...")
    context_text = search_documents(user_msg)

    # 4. LLM 스트리밍 생성 & 캐시 업데이트
    async def llm_streamer():
        full_answer = ""
        for chunk in rag_chain.stream({"context": context_text, "question": user_msg}):
            text_chunk = chunk.content
            full_answer += text_chunk
            yield f"data: {text_chunk}\n\n"
        
        # 답변 생성이 끝나면 백그라운드에서 캐시에 저장
        update_cache(user_msg, full_answer)
        yield "data: [DONE]\n\n"

    return StreamingResponse(llm_streamer(), media_type="text/event-stream")