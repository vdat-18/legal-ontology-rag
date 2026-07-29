"""Graphiti RAG pipeline: rewrite the query, search the graph, keep the top-ranked facts, generate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.graphiti_rag.client import build_graphiti
from legal_rag.graphiti_rag.generator import generate_answer
from legal_rag.graphiti_rag.provenance import BamlDocumentCitationClient, find_unindexed_document_citations
from legal_rag.graphiti_rag.query_rewriter import BamlQueryRewriteClient
from legal_rag.graphiti_rag.retriever import RetrievedFact, search_facts
from legal_rag.naive_rag.generator import BamlGenerationClient
from legal_rag.settings import GraphitiSettings, Settings


@dataclass(frozen=True, slots=True)
class GraphitiRAGResult:
    """Container for one Graphiti RAG pipeline response."""

    query: str
    rewritten_query: str
    answer: str
    sources: tuple[RetrievedFact, ...] = ()
    unindexed_citations: tuple[str, ...] = ()


class GraphitiRAGPipeline:
    """End-to-end rewrite-search-then-generate pipeline over the Graphiti knowledge graph.

    Retrieval order comes from Graphiti's hybrid search and cross-encoder reranker; the pipeline
    keeps the top ``max_context_facts`` of that ranking as context rather than re-scoring facts
    with any handcrafted rule.
    """

    def __init__(
        self,
        settings: Settings,
        graphiti_settings: GraphitiSettings,
        group_id: str,
        known_document_numbers: frozenset[str] | None = None,
    ) -> None:
        """Initialize the pipeline from application and Graphiti settings, scoped to one group.

        Args:
            settings: Application LLM connection settings.
            graphiti_settings: Neo4j connection, embedding profile, and retrieval settings.
            group_id: The Graphiti group to search within.
            known_document_numbers: Document numbers actually indexed into this group, or ``None`` when
                corpus coverage is unknown and citation coverage checking must be skipped.
        """
        self._settings = graphiti_settings
        self._group_id = group_id
        self._known_document_numbers = known_document_numbers
        profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
        self._graphiti_client = build_graphiti(settings, graphiti_settings, profile)
        self._generation_client = BamlGenerationClient(settings, graphiti_settings)
        self._query_rewrite_client = BamlQueryRewriteClient(settings, graphiti_settings)
        self._citation_client = BamlDocumentCitationClient(settings, graphiti_settings)
        logger.info("GraphitiRAGPipeline initialized for group '{}'", group_id)

    async def run(self, query: str, max_facts: int | None = None) -> GraphitiRAGResult:
        """Execute the full pipeline for a single query.

        Args:
            query: Natural-language question, in whatever phrasing the user typed it.
            max_facts: Override the default maximum number of selected context facts. Must be
                positive when given.

        Returns:
            Result containing the answer, the facts it was grounded in, and any citations to
            documents outside the indexed corpus.

        Raises:
            ValueError: If *max_facts* is given but not positive.
        """
        if max_facts is not None and max_facts < 1:
            msg = f"max_facts must be positive, got {max_facts}"
            raise ValueError(msg)

        rewritten_query = await self._query_rewrite_client.rewrite(query)
        if rewritten_query != query:
            logger.debug("Query rewrite completed")

        candidates = await search_facts(
            self._graphiti_client, rewritten_query, self._group_id, self._settings.search_num_results
        )
        fact_budget = self._settings.max_context_facts if max_facts is None else max_facts
        selected = candidates[:fact_budget]
        answer = await asyncio.to_thread(generate_answer, query, selected, self._generation_client)

        unindexed_citations: list[str] = []
        if self._known_document_numbers is not None:
            unindexed_citations = await asyncio.to_thread(
                find_unindexed_document_citations,
                [answer, *(fact.fact for fact in selected)],
                self._known_document_numbers,
                self._citation_client,
            )
        if unindexed_citations:
            logger.warning("Answer cites document(s) outside the indexed corpus: {}", unindexed_citations)

        logger.info("Graphiti pipeline completed")
        return GraphitiRAGResult(
            query=query,
            rewritten_query=rewritten_query,
            answer=answer,
            sources=tuple(selected),
            unindexed_citations=tuple(unindexed_citations),
        )

    async def close(self) -> None:
        """Release the underlying Graphiti/Neo4j connection."""
        await self._graphiti_client.close()
