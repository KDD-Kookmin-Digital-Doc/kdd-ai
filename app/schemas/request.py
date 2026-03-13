from pydantic import BaseModel
#Spring Boot 백엔드에서 AI 서버로 보내줄 데이터의 형식을 정의
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    # Field를 사용해서 Swagger UI에 띄울 설명과 예시 데이터를 직접 지정합니다.
    message: str = Field(
        ..., 
        description="사용자가 챗봇에게 던지는 질문 텍스트", 
        json_schema_extra={
            "example": "휴학은 최대 몇 년까지 가능한가요?"
        }
    )