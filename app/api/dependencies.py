"""FastAPI 의존성 주입 모듈. 클라이언트 싱글턴 관리."""

from __future__ import annotations

from functools import lru_cache

from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings, get_settings


@lru_cache
def get_bedrock_client() -> BedrockClient:
    """BedrockClient 싱글턴 인스턴스를 반환한다."""
    return BedrockClient(get_settings())


@lru_cache
def get_postgres_client() -> PostgresVectorClient:
    """PostgresVectorClient 싱글턴 인스턴스를 반환한다."""
    return PostgresVectorClient(get_settings())
