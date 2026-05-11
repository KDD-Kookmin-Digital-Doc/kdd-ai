"""벡터 검색 모듈. 질문 임베딩 기반으로 관련 문서를 검색하고, 임계값 미만 시 폴백 처리한다."""

from __future__ import annotations

import logging

from app.clients.bedrock import BedrockClient
from app.clients.supabase_client import SupabaseVectorClient
from app.config import Settings
from app.models.pipeline import PipelineContext, SearchResult, SourceDoc

logger = logging.getLogger(__name__)


async def search_documents(
    context: PipelineContext,
    bedrock: BedrockClient,
    supabase: SupabaseVectorClient,
    settings: Settings,
) -> PipelineContext:
    """질문을 임베딩하여 벡터 검색을 수행하고 PipelineContext를 갱신한다.

    - 재작성된 질문(또는 원본)을 Cohere Embed로 1024차원 벡터로 변환.
    - documents 테이블에서 코사인 유사도 기반 후보 검색.
    - D1: ``RERANK_ENABLED=True`` 시 stage 1 후보 ``RETRIEVE_TOP_K_RERANK``개 →
      도쿄 리전 Cohere Rerank 3.5 → stage 2 ``RERANK_TOP_N``개. False 시
      ``VECTOR_SEARCH_TOP_K``개 단일 단계.
    - 유사도 임계값 이상 문서가 1개 이상 → search_results, source_docs 설정.
    - 모든 문서가 임계값 미만 → Fallback: answer_cache에서 유사 질문 3개 추출.
    """
    question = context.rewritten_question or context.original_question

    # Task 16: semantic_cache가 미리 임베딩한 결과를 텍스트 + input_type 일치 시 재사용.
    # rewrite로 텍스트가 달라졌거나 cache 단계 자체를 skip(멀티턴)했을 때만 새로 호출.
    # input_type 까지 비교해 미래에 cache 측 input_type 이 바뀌어도 silent quality
    # degradation 이 발생하지 않도록 가드.
    if (
        context.question_embedding is not None
        and context.embedded_question_text == question
        and context.embedded_question_input_type == "search_query"
    ):
        question_embedding = context.question_embedding
        logger.debug("임베딩 재사용 (텍스트 일치): %r", question)
    else:
        embeddings = await bedrock.embed_texts(
            [question], input_type="search_query"
        )
        question_embedding = embeddings[0]

    # D1: rerank 활성 시 후보를 넓혀야 cross-encoder 가 임베딩이 놓친 문서를 끌어올릴 수
    # 있음. 비활성 시 기존 동작 (top_k=VECTOR_SEARCH_TOP_K) 그대로.
    retrieve_top_k = (
        settings.RETRIEVE_TOP_K_RERANK
        if settings.RERANK_ENABLED
        else settings.VECTOR_SEARCH_TOP_K
    )
    results = await supabase.search_documents(
        embedding=question_embedding,
        top_k=retrieve_top_k,
        threshold=settings.SIMILARITY_THRESHOLD,
    )

    if settings.RERANK_ENABLED and results:
        results = await _apply_rerank(bedrock, question, results, settings)

    # PR-R7: 두 분기 모두에서 search_results/source_docs/suggested_questions 를
    # 명시적으로 set 한다. 이전엔 함수 진입 직후 빈 list 로 사전 할당 후 분기
    # 내에서 일부만 갱신했는데, "분기마다 끝났을 때의 컨텍스트 상태"가 한 곳에
    # 모이도록 정리.
    if results:
        context.search_results = results
        context.source_docs = [SourceDoc.from_search_result(r) for r in results]
        context.suggested_questions = []
        logger.info(
            "벡터 검색 성공: %d건 (최고 유사도=%.4f)",
            len(results),
            results[0].similarity_score,
        )
    else:
        suggested = await supabase.search_similar_questions(
            embedding=question_embedding,
            top_k=settings.FALLBACK_SUGGESTED_COUNT,
            threshold=settings.FALLBACK_SIMILARITY_THRESHOLD,
        )
        context.search_results = []
        context.source_docs = []
        context.suggested_questions = suggested
        logger.info(
            "벡터 검색 폴백: 임계값(%.2f) 이상 문서 없음, 유사 질문 %d건 추출",
            settings.SIMILARITY_THRESHOLD,
            len(suggested),
        )

    return context


async def _apply_rerank(
    bedrock: BedrockClient,
    question: str,
    results: list[SearchResult],
    settings: Settings,
) -> list[SearchResult]:
    """Stage 2: Cohere Rerank 3.5 호출 + 재정렬 + RERANK_TOP_N 슬라이스.

    실패 시 graceful degradation — 임베딩 결과 상위 ``RERANK_TOP_N`` 개로 fallback.
    검색 자체는 실패시키지 않는다 (rerank 는 품질 향상 layer 일 뿐).
    """
    try:
        reranked = await bedrock.rerank(
            query=question,
            documents=[r.content for r in results],
            top_n=settings.RERANK_TOP_N,
        )
    except Exception as exc:
        logger.warning("Rerank 호출 실패, 임베딩 결과 fallback: %s", exc)
        return results[: settings.RERANK_TOP_N]

    reordered: list[SearchResult] = []
    for idx, score in reranked:
        r = results[idx]
        r.rerank_score = score
        reordered.append(r)
    logger.info(
        "Rerank 성공: %d → %d건 (최고 rerank_score=%.4f)",
        len(results),
        len(reordered),
        reordered[0].rerank_score if reordered else 0.0,
    )
    return reordered
