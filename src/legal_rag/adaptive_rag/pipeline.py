"""One-shot adaptive RAG pipeline with direct and multi-hop retrieval routes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from loguru import logger

from legal_rag.adaptive_rag.models import (
    AdaptiveQueryPlan,
    AdaptiveRAGResult,
    AdaptiveRetrievalResult,
    AdaptiveRoute,
)
from legal_rag.adaptive_rag.planner import AdaptivePlanner, BamlAdaptivePlanner
from legal_rag.datastores.milvus import MilvusCorpusFingerprint, build_milvus_client
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.naive_rag.generator import BamlGenerationClient, GenerationClient, generate_answer
from legal_rag.naive_rag.models import RetrievedChunk, SearchMode
from legal_rag.naive_rag.retriever import Retriever
from legal_rag.query_rewriting import QueryRewriter, SyncQueryRewriter
from legal_rag.retrieval.rrf import reciprocal_rank_fusion
from legal_rag.settings import AdaptiveRagSettings, MilvusSettings, NaiveRagSettings, Settings


class AdaptiveRAGPipeline:
    """Plan one retrieval route, gather evidence, then generate one answer."""

    def __init__(
        self,
        settings: Settings,
        milvus_settings: MilvusSettings,
        naive_rag_settings: NaiveRagSettings,
        adaptive_rag_settings: AdaptiveRagSettings,
        *,
        retriever: Retriever | None = None,
        generation_client: GenerationClient | None = None,
        query_rewriter: QueryRewriter | None = None,
        planner: AdaptivePlanner | None = None,
    ) -> None:
        """Initialize adaptive orchestration from existing Naive RAG components."""
        self.settings = adaptive_rag_settings
        self._default_top_k = naive_rag_settings.top_k
        if retriever is None:
            profile = load_configured_embedding_profile(naive_rag_settings.embedding_profile)
            encoder = EmbeddingEncoder.from_profile(profile)
            retriever = Retriever(encoder, build_milvus_client(milvus_settings), naive_rag_settings)
        self._retriever = retriever
        self._generation_client = generation_client or BamlGenerationClient(settings, naive_rag_settings)
        self._query_rewriter = query_rewriter or SyncQueryRewriter(settings)
        self._planner = planner or BamlAdaptivePlanner(settings, adaptive_rag_settings)
        logger.info("AdaptiveRAGPipeline initialized")

    def run(
        self,
        query: str,
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
        rewrite_query: bool = True,
    ) -> AdaptiveRAGResult:
        """Plan retrieval, fuse multi-hop evidence when needed, and generate an answer."""
        retrieval = self.retrieve(query, top_k=top_k, history=history, rewrite_query=rewrite_query)
        answer = generate_answer(retrieval.rewritten_query or query, retrieval.sources, self._generation_client)
        logger.info(
            "Adaptive pipeline completed: route={}, subqueries={}, sources={}",
            retrieval.route.value,
            len(retrieval.subqueries),
            len(retrieval.sources),
        )
        return AdaptiveRAGResult(
            query=query,
            rewritten_query=retrieval.rewritten_query,
            answer=answer,
            sources=retrieval.sources,
            route=retrieval.route,
            subqueries=retrieval.subqueries,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
        rewrite_query: bool = True,
    ) -> AdaptiveRetrievalResult:
        """Plan and execute adaptive retrieval without generating an answer."""
        k = top_k or self._default_top_k
        retrieval_query = query
        if rewrite_query and history:
            rewrite_result = self._query_rewriter.rewrite(query, tuple(history))
            retrieval_query = rewrite_result.rewritten

        plan = self._planner.plan(retrieval_query)
        mode = SearchMode(self.settings.base_search_mode)
        hits = self._retrieve_plan(retrieval_query, plan, k, mode)
        logger.info(
            "Adaptive retrieval completed: route={}, subqueries={}, sources={}",
            plan.route.value,
            len(plan.subqueries),
            len(hits),
        )
        return AdaptiveRetrievalResult(
            query=query,
            rewritten_query=retrieval_query if retrieval_query != query else None,
            sources=tuple(hits),
            route=plan.route,
            subqueries=plan.subqueries,
        )

    def _retrieve_plan(
        self,
        retrieval_query: str,
        plan: AdaptiveQueryPlan,
        top_k: int,
        mode: SearchMode,
    ) -> list[RetrievedChunk]:
        """Execute the selected route and return final ranked evidence."""
        if plan.route == AdaptiveRoute.DIRECT:
            return self._retriever.retrieve(retrieval_query, top_k=top_k, mode=mode)

        leg_limit = max(top_k, self.settings.per_leg_limit)
        rankings = [
            self._retriever.retrieve(leg_query, top_k=leg_limit, mode=mode)
            for leg_query in (retrieval_query, *plan.subqueries)
        ]
        fused = reciprocal_rank_fusion(
            rankings,
            key=lambda hit: hit.effective_unit_id,
            rrf_k=self.settings.rrf_k,
        )
        return [replace(result.item, score=result.score) for result in fused[:top_k]]

    def preflight_corpus(self) -> MilvusCorpusFingerprint:
        """Verify that the active Milvus collection matches the configured local corpus."""
        return self._retriever.preflight_corpus()

    def close(self) -> None:
        """Close the owned Milvus retrieval connection."""
        self._retriever.close()
