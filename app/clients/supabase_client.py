"""Supabase Vector DB 클라이언트 모듈. supabase-py 사용."""

from __future__ import annotations

import asyncio
import logging

from supabase import Client, create_client

from app.config import Settings
from app.models.pipeline import AnswerCache, CacheMatch, SearchResult

logger = logging.getLogger(__name__)


class SupabaseVectorClient:
    """Supabase Vector DB 클라이언트."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_KEY
        )
        self._timeout = settings.SUPABASE_TIMEOUT

    # ── 내부 헬퍼 ──

    async def _run_with_timeout(self, fn):
        """동기 함수를 비동기로 실행하며 타임아웃을 적용한다."""
        return await asyncio.wait_for(
            asyncio.to_thread(fn),
            timeout=self._timeout,
        )

    # ── 벡터 검색 ──

    async def search_documents(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[SearchResult]:
        """documents 테이블에서 코사인 유사도 기반 벡터 검색."""
        result = await self._run_with_timeout(
            lambda: self._client.rpc(
                "match_documents",
                {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "match_count": top_k,
                },
            ).execute()
        )

        return [
            SearchResult(
                id=row["id"],
                doc_id=row["doc_id"],
                content=row["content"],
                metadata=row["metadata"],
                similarity_score=row["similarity"],
            )
            for row in (result.data or [])
        ]

    async def search_answer_cache(
        self,
        embedding: list[float],
        threshold: float = 0.95,
        ttl_days: int | None = None,
    ) -> CacheMatch | None:
        """answer_cache 테이블에서 시맨틱 캐시 탐색.

        유사도 최상위 1건만 반환. TTL 만료 레코드는 제외.
        반환 시 sources JSONB 컬럼도 함께 조회하여 캐시 히트 시
        출처 정보를 포함한다.
        """
        effective_ttl = (
            self._settings.CACHE_TTL_DAYS if ttl_days is None else ttl_days
        )

        result = await self._run_with_timeout(
            lambda: self._client.rpc(
                "match_answer_cache",
                {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "ttl_days": effective_ttl,
                },
            ).execute()
        )

        rows = result.data or []
        if not rows:
            return None

        row = rows[0]
        return CacheMatch(
            question=row["question"],
            answer=row["answer"],
            similarity_score=row["similarity"],
            sources=row.get("sources") or [],
        )

    async def search_similar_questions(
        self,
        embedding: list[float],
        top_k: int = 3,
    ) -> list[str]:
        """answer_cache에서 유사 질문 추출 (Fallback용).

        임계값 없이 상위 top_k개 반환.
        """
        result = await self._run_with_timeout(
            lambda: self._client.rpc(
                "match_similar_questions",
                {
                    "query_embedding": embedding,
                    "match_count": top_k,
                },
            ).execute()
        )

        return [row["question"] for row in (result.data or [])]

    # ── 문서 삽입/삭제 ──

    async def insert_document_chunks(
        self, doc_id: str, chunks: list[dict]
    ) -> int:
        """문서 청크를 documents 테이블에 일괄 삽입. 삽입된 행 수 반환."""
        payload = [{**chunk, "doc_id": doc_id} for chunk in chunks]
        result = await self._run_with_timeout(
            lambda: self._client.table("documents").insert(payload).execute()
        )
        return len(result.data or [])

    async def delete_document_chunks(self, doc_id: str) -> int:
        """doc_id에 해당하는 모든 청크 삭제. 삭제된 행 수 반환."""
        result = await self._run_with_timeout(
            lambda: self._client.table("documents")
            .delete()
            .eq("doc_id", doc_id)
            .execute()
        )
        return len(result.data or [])

    # ── 캐시 관리 ──

    async def invalidate_cache_by_doc_id(self, doc_id: str) -> int:
        """source_doc_ids에 해당 doc_id가 포함된 answer_cache 삭제."""
        result = await self._run_with_timeout(
            lambda: self._client.table("answer_cache")
            .delete()
            .contains("source_doc_ids", [doc_id])
            .execute()
        )
        return len(result.data or [])

    async def insert_answer_cache(self, cache: AnswerCache) -> None:
        """답변 캐시 저장. 학사규정 질문의 정상 답변 완료 시에만 호출."""
        await self._run_with_timeout(
            lambda: self._client.table("answer_cache")
            .insert(
                {
                    "question": cache.question,
                    "embedding": cache.embedding,
                    "answer": cache.answer,
                    "source_doc_ids": cache.source_doc_ids,
                    "sources": cache.sources,
                }
            )
            .execute()
        )

    # ── 헬스체크 ──

    async def health_check(self) -> bool:
        """DB 연결 상태 확인."""
        try:
            await self._run_with_timeout(
                lambda: self._client.table("documents")
                .select("id")
                .limit(1)
                .execute()
            )
            return True
        except Exception as exc:
            logger.warning("Supabase 헬스체크 실패: %s", exc)
            return False
