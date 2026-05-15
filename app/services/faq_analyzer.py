"""FAQ 분석 서비스. 질문 클러스터링 → 대표 질문 선정 → 답변 초안 생성."""

from __future__ import annotations

import asyncio
import logging

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from app.clients.bedrock import BedrockClient
from app.clients.postgres_client import PostgresVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """클러스터링을 수행하기에 질문 데이터가 부족할 때 발생."""

    def __init__(self, min_required: int):
        self.min_required = min_required
        super().__init__(
            f"클러스터링을 수행하기에 질문 데이터가 부족합니다. "
            f"최소 {min_required}개 이상의 질문이 필요합니다."
        )


async def analyze_faq(
    questions: list[str],
    top_k: int,
    min_cluster_size: int,
    bedrock: BedrockClient,
    postgres: PostgresVectorClient,
    settings: Settings,
) -> list[dict]:
    """질문 배열을 클러스터링하여 FAQ 후보를 생성한다.

    1. 질문 수가 min_cluster_size 미만이면 InsufficientDataError.
    2. Cohere Embed로 질문 벡터화.
    3. HDBSCAN으로 클러스터링 (노이즈 자동 제외).
    4. 각 클러스터에서 중심에 가장 가까운 질문을 대표 질문으로 선정.
    5. 대표 질문에 대해 벡터 검색 → LLM 답변 초안 생성.
    6. 빈도순 정렬, top_k개 반환.
    """
    if len(questions) < min_cluster_size:
        raise InsufficientDataError(min_cluster_size)

    # 1. 질문 벡터화
    embeddings = await bedrock.embed_texts(questions, input_type="search_query")
    embedding_matrix = np.array(embeddings)

    # 2. HDBSCAN 클러스터링 (L2 정규화 후 euclidean = cosine 거리 동치)
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 영벡터 방어
    normalized = embedding_matrix / norms

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(normalized)

    # 3. 클러스터별 대표 질문 선정
    cluster_ids = set(labels)
    cluster_ids.discard(-1)  # 노이즈 제외

    if not cluster_ids:
        return []

    clusters: list[dict] = []
    for cid in cluster_ids:
        member_indices = [i for i, l in enumerate(labels) if l == cid]
        member_embeddings = embedding_matrix[member_indices]

        # 클러스터 중심 계산
        centroid = member_embeddings.mean(axis=0).reshape(1, -1)
        distances = cosine_distances(centroid, member_embeddings)[0]
        closest_idx = member_indices[int(np.argmin(distances))]

        clusters.append({
            "representative_idx": closest_idx,
            "representative_question": questions[closest_idx],
            "representative_embedding": embeddings[closest_idx],
            "frequency": len(member_indices),
        })

    # 4. 빈도순 정렬 후 top_k개
    clusters.sort(key=lambda c: c["frequency"], reverse=True)
    top_clusters = clusters[:top_k]

    # 5. 대표 질문별 답변 초안 생성 (병렬 + Semaphore burst 가드)
    semaphore = asyncio.Semaphore(settings.FAQ_CONCURRENCY)

    async def _bounded_draft(cluster: dict) -> str:
        async with semaphore:
            return await _generate_draft_answer(
                question=cluster["representative_question"],
                embedding=cluster["representative_embedding"],
                bedrock=bedrock,
                postgres=postgres,
                settings=settings,
            )

    results = await asyncio.gather(
        *(_bounded_draft(c) for c in top_clusters),
        return_exceptions=True,
    )

    candidates: list[dict] = []
    for cluster, result in zip(top_clusters, results, strict=True):
        if isinstance(result, BaseException):
            # gather(return_exceptions=True) 는 inner CancelledError 도 결과로 wrap.
            # caller 일관성을 위해 cancel 은 재전파 (응답 자체가 폐기됨).
            if isinstance(result, asyncio.CancelledError):
                raise result
            logger.warning(
                "FAQ 답변 초안 생성 실패 (질문: %r): %s",
                cluster["representative_question"],
                result,
            )
            draft_answer = f"답변 초안 생성 실패: {type(result).__name__}"
        else:
            draft_answer = result
        candidates.append({
            "question": cluster["representative_question"],
            "draft_answer": draft_answer,
            "frequency": cluster["frequency"],
        })

    return candidates


async def _generate_draft_answer(
    question: str,
    embedding: list[float],
    bedrock: BedrockClient,
    postgres: PostgresVectorClient,
    settings: Settings,
) -> str:
    """대표 질문에 대해 벡터 검색 → LLM 답변 초안을 생성한다."""
    # 벡터 검색
    results = await postgres.search_documents(
        embedding=embedding,
        top_k=settings.VECTOR_SEARCH_TOP_K,
        threshold=settings.SIMILARITY_THRESHOLD,
    )

    if not results:
        return "관련 문서를 찾을 수 없어 답변 초안을 생성할 수 없습니다."

    # 문서 컨텍스트 구성
    doc_lines: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r.metadata
        doc_name = meta.get("doc_name", "unknown")
        page = meta.get("page", "?")
        enforcement_date = meta.get("enforcement_date", "미상")
        doc_lines.append(
            f"[문서 {i}] {doc_name} (시행일: {enforcement_date}, 페이지: {page})"
        )
        doc_lines.append(r.content)
        doc_lines.append("")

    doc_context = "\n".join(doc_lines).strip()

    system_prompt = (
        "당신은 대학교 학사규정 FAQ 답변 초안을 작성하는 전문가입니다.\n"
        "아래 문서 컨텍스트를 근거로 질문에 대한 간결한 답변 초안을 작성하세요.\n"
        "문서에 없는 내용은 포함하지 마세요.\n\n"
        f"## 문서 컨텍스트\n{doc_context}"
    )
    messages = [{"role": "user", "content": [{"text": question}]}]

    answer, _ = await bedrock.invoke_llm(
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=settings.FAQ_LLM_MAX_TOKENS,
        model="answer",
    )

    return answer.strip()
