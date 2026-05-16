"""PostgreSQL Vector DB 클라이언트 모듈. asyncpg + pgvector 사용.

D3 (2026-05-13): supabase-py → asyncpg 마이그레이션. 호출자 인터페이스 (메서드명
+ 시그니처 + 반환 타입) 는 기존 ``SupabaseVectorClient`` 와 100% 동일하게 유지.
"""

from __future__ import annotations

import asyncio
import json
import logging

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings
from app.models.pipeline import AnswerCache, CacheMatch, SearchResult

logger = logging.getLogger(__name__)


class PostgresVectorClient:
    """RDS PostgreSQL + pgvector 클라이언트 (asyncpg pool 기반).

    Pool 은 lazy 초기화 — 첫 호출 시점에 ``create_pool``. ``lifespan`` 종료 시
    ``close()`` 호출로 graceful drain. ``command_timeout`` 으로 쿼리 단위 타임아웃
    강제, 초과 시 ``asyncio.TimeoutError`` 발생.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None
        # double-checked locking — 동시 코루틴이 _init_pool 진입 시 create_pool
        # 가 두 번 호출돼 pool 1개 누수되는 race 차단 (CodeRabbit CR1).
        self._pool_lock = asyncio.Lock()

    async def _init_pool(self) -> None:
        """Pool 을 lazy 초기화한다. 이미 만들어졌으면 no-op.

        ``asyncio.wait_for`` 로 pool 생성 자체에 timeout 강제 — RDS endpoint
        무응답 시 startup/health_check 가 무한정 hang 하는 사고 차단.
        ``command_timeout`` (쿼리 단위) 과 별개로 connect/handshake 가드.

        ``asyncio.Lock`` + double-check 로 동시 진입 race 차단. 첫 가드는
        lock 미획득 fast-path, lock 안의 두 번째 가드가 실제 atomic 보장.
        """
        if self._pool is not None:
            return
        async with self._pool_lock:
            if self._pool is not None:
                return
            self._pool = await asyncio.wait_for(
                asyncpg.create_pool(
                    dsn=self._settings.DATABASE_URL,
                    min_size=self._settings.POSTGRES_POOL_MIN_SIZE,
                    max_size=self._settings.POSTGRES_POOL_MAX_SIZE,
                    command_timeout=self._settings.POSTGRES_TIMEOUT,
                    init=self._init_connection,
                ),
                timeout=self._settings.POSTGRES_TIMEOUT,
            )

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """새 connection 마다 호출 — vector + jsonb codec 등록.

        - ``pgvector.asyncpg.register_vector`` → ``vector(1024) ↔ list[float]``
        - jsonb codec → ``jsonb ↔ dict/list`` (양방향 자동 직렬화)
        BIGINT[] 은 asyncpg 가 native 처리하므로 codec 등록 불필요.
        """
        await register_vector(conn)
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

    async def close(self) -> None:
        """Pool drain. ``lifespan`` 종료 시 호출."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ── 벡터 검색 ──

    async def search_documents(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[SearchResult]:
        """documents 테이블에서 코사인 유사도 기반 벡터 검색."""
        await self._init_pool()
        rows = await self._pool.fetch(
            "SELECT chunk_id, doc_id, content, metadata, similarity "
            "FROM match_documents($1, $2, $3)",
            embedding,
            threshold,
            top_k,
        )
        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                content=row["content"],
                metadata=row["metadata"],
                similarity_score=row["similarity"],
            )
            for row in rows
        ]

    async def search_answer_cache(
        self,
        embedding: list[float],
        threshold: float = 0.95,
        ttl_days: int | None = None,
    ) -> CacheMatch | None:
        """answer_cache 에서 시맨틱 캐시 탐색 (유사도 최상위 1건).

        TTL 만료 레코드는 RPC 내부에서 제외. sources JSONB 컬럼도 함께 조회.
        """
        effective_ttl = (
            self._settings.CACHE_TTL_DAYS if ttl_days is None else ttl_days
        )
        await self._init_pool()
        row = await self._pool.fetchrow(
            "SELECT question, answer, similarity, sources, confidence "
            "FROM match_answer_cache($1, $2, $3)",
            embedding,
            threshold,
            effective_ttl,
        )
        if row is None:
            return None
        return CacheMatch(
            question=row["question"],
            answer=row["answer"],
            similarity_score=row["similarity"],
            sources=row["sources"] or [],
            confidence=row["confidence"],
        )

    async def search_similar_questions(
        self,
        embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> list[str]:
        """answer_cache 에서 유사 질문 추출 (Fallback 용).

        ``threshold`` 이상 유사도를 가진 질문만 상위 ``top_k`` 개 반환 (이슈 #46).
        """
        await self._init_pool()
        rows = await self._pool.fetch(
            "SELECT question FROM match_similar_questions($1, $2, $3)",
            embedding,
            top_k,
            threshold,
        )
        return [row["question"] for row in rows]

    # ── 문서 삽입/삭제 ──

    async def replace_document_chunks(
        self, doc_id: int, chunks: list[dict]
    ) -> int:
        """단일 트랜잭션 내 기존 청크/관련 캐시 삭제 후 새 청크 삽입.

        ``replace_document_chunks`` RPC (이슈 #43) 호출 — INSERT 실패 시 DELETE 도
        자동 롤백. chunks 가 비어 있으면 RPC 미호출 + 0 반환 (no-op, 기존 데이터
        보존). 문서를 비우려는 의도라면 ``delete_document_chunks`` 사용.
        """
        if not chunks:
            return 0
        await self._init_pool()
        # chunks 안의 embedding 은 list[float] — jsonb 직렬화 가능 (RPC 내부에서
        # `(chunk->>'embedding')::VECTOR(1024)` 캐스트).
        result = await self._pool.fetchval(
            "SELECT replace_document_chunks($1, $2)",
            doc_id,
            chunks,
        )
        return int(result or 0)

    async def delete_document_chunks(self, doc_id: int) -> int:
        """doc_id 에 해당하는 모든 청크 삭제. 삭제된 행 수 반환."""
        await self._init_pool()
        rows = await self._pool.fetch(
            "DELETE FROM documents WHERE doc_id = $1 RETURNING chunk_id",
            doc_id,
        )
        return len(rows)

    # ── 캐시 관리 ──

    async def invalidate_cache_by_doc_id(self, doc_id: int) -> int:
        """source_doc_ids 에 해당 doc_id 가 포함된 answer_cache 삭제.

        asyncpg 는 BIGINT[] 컬럼을 native 처리 — supabase-py 가 강제했던 stringify
        우회 불필요.
        """
        await self._init_pool()
        rows = await self._pool.fetch(
            "DELETE FROM answer_cache WHERE $1 = ANY(source_doc_ids) RETURNING id",
            doc_id,
        )
        return len(rows)

    async def upsert_answer_cache(self, cache: AnswerCache) -> None:
        """답변 캐시 저장 (UPSERT, 이슈 #49). 학사규정 정상 답변 완료 시에만 호출.

        ``upsert_answer_cache`` RPC 가 단일 트랜잭션 + ``pg_advisory_xact_lock``
        으로 동일 ``question`` 의 기존 row 를 DELETE 후 INSERT. 동시 호출 시에도
        중복 row 누적 차단.
        """
        await self._init_pool()
        await self._pool.execute(
            "SELECT upsert_answer_cache($1, $2, $3, $4, $5, $6)",
            cache.question,
            cache.embedding,
            cache.answer,
            cache.source_doc_ids,
            cache.sources,
            cache.confidence,
        )

    # ── 헬스체크 ──

    async def health_check(self) -> bool:
        """DB 연결 상태 확인."""
        try:
            await self._init_pool()
            await self._pool.fetchval("SELECT 1 FROM documents LIMIT 1")
            return True
        except Exception as exc:
            logger.warning("Postgres 헬스체크 실패: %s", exc)
            return False
