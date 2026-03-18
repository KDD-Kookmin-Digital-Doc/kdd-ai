from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="사용자가 챗봇에게 던지는 질문 텍스트",
        json_schema_extra={
            "example": "휴학은 최대 몇 년까지 가능한가요?"
        }
    )
    session_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="멀티턴 대화를 위한 세션 ID. 미입력 시 싱글턴 대화로 처리됩니다.",
        json_schema_extra={
            "example": "user-abc-123"
        }
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message는 공백만으로 구성될 수 없습니다.")
        return v

    @field_validator("session_id")
    @classmethod
    def session_id_not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("session_id는 공백만으로 구성될 수 없습니다.")
        return v
