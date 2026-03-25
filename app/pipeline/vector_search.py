"""벡터 검색 모듈. 질문 임베딩 기반으로 관련 문서를 검색하고, 임계값 미만 시 폴백 처리한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext, SourceDoc

logger = logging.getLogger(__name__)


async def search_documents(
    context: PipelineContext,
    bedrock: BedrockClient,
    supabase: SupabaseVectorClient,
    settings: Settings,
) -> PipelineContext:
    """질문을 임베딩하여 벡터 검색을 수행하고 PipelineContext를 갱신한다.

    - 재작성된 질문(또는 원본)을 Cohere Embed v3로 1024차원 벡터로 변환.
    - documents 테이블에서 코사인 유사도 기반 상위 5개 검색.
    - 유사도 임계값 이상 문서가 1개 이상 → search_results, source_docs 설정.
    - 모든 문서가 임계값 미만 → Fallback: answer_cache에서 유사 질문 3개 추출.
    """
    question = context.rewritten_question or context.original_question

    embeddings = await bedrock.embed_texts(
        [question], input_type="search_query"
    )
    question_embedding = embeddings[0]

    results = await supabase.search_documents(
        embedding=question_embedding,
        top_k=5,
        threshold=settings.SIMILARITY_THRESHOLD,
    )

    if results:
        context.search_results = results
        context.source_docs = [
            SourceDoc(
                doc_id=r.doc_id,
                doc_name=r.metadata.get("doc_name", ""),
                page=r.metadata.get("page", 0),
            )
            for r in results
        ]
        logger.info(
            "벡터 검색 성공: %d건 (최고 유사도=%.4f)",
            len(results),
            results[0].similarity_score,
        )
    else:
        context.search_results = []
        suggested = await supabase.search_similar_questions(
            embedding=question_embedding,
            top_k=3,
        )
        context.suggested_questions = suggested
        logger.info(
            "벡터 검색 폴백: 임계값(%.2f) 이상 문서 없음, 유사 질문 %d건 추출",
            settings.SIMILARITY_THRESHOLD,
            len(suggested),
        )

    return context
