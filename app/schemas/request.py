from pydantic import BaseModel, Field
from typing import Optional

# Spring Boot 백엔드에서 AI 서버로 보내줄 데이터의 형식을 정의

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="사용자가 챗봇에게 던지는 질문 텍스트",
        json_schema_extra={
            "example": "휴학은 최대 몇 년까지 가능한가요?"
        }
    )
    session_id: Optional[str] = Field(
        default=None,
        description="멀티턴 대화를 위한 세션 ID. 미입력 시 싱글턴 대화로 처리됩니다.",
        json_schema_extra={
            "example": "user-abc-123"
        }
    )
