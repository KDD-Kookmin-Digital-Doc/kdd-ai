from fastapi import FastAPI
from app.api.chat import router as chat_router

app = FastAPI(title="Kookmin RAG API Server (MSA)")

# /api 접두사를 붙여서 chat 라우터를 연결합니다.
app.include_router(chat_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # 실행 명령어: python app/main.py 또는 uvicorn app.main:app --reload
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)