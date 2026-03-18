import os
from fastapi import FastAPI
from app.api.chat import router as chat_router

app = FastAPI(title="Kookmin RAG API Server (MSA)")

app.include_router(chat_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    is_prod = os.getenv("APP_ENV", "development").lower() == "production"
    host = os.getenv("UVICORN_HOST", "127.0.0.1" if is_prod else "0.0.0.0")
    reload = os.getenv("UVICORN_RELOAD", "false" if is_prod else "true").lower() == "true"

    uvicorn.run("app.main:app", host=host, port=8000, reload=reload)
