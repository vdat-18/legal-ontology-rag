"""Graphiti RAG pipeline: rewrite the query, search the graph, select minimal facts, generate."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.graphiti_rag.client import build_graphiti
from legal_rag.graphiti_rag.generator import generate_answer
from legal_rag.graphiti_rag.provenance import find_unindexed_document_citations
from legal_rag.graphiti_rag.query_rewriter import BamlQueryRewriteClient
from legal_rag.graphiti_rag.retriever import RetrievedFact, search_facts, select_minimal_facts
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
    """End-to-end rewrite-search-select-then-generate pipeline over the Graphiti knowledge graph."""

    def __init__(
        self,
        settings: Settings,
        graphiti_settings: GraphitiSettings,
        group_id: str,
        known_so_hieu: frozenset[str] = frozenset(),
    ) -> None:
        """Initialize the pipeline from application and Graphiti settings, scoped to one group.

        Args:
            settings: Application LLM connection settings.
            graphiti_settings: Neo4j connection, embedding profile, and retrieval settings.
            group_id: The Graphiti group to search within.
            known_so_hieu: Document numbers actually indexed into this group — used to flag
                answers that cite a document outside this set (see ``graphiti_rag.provenance``).
        """
        self._settings = graphiti_settings
        self._group_id = group_id
        self._known_so_hieu = known_so_hieu
        profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
        self._graphiti_client = build_graphiti(settings, graphiti_settings, profile)
        self._generation_client = BamlGenerationClient(settings, graphiti_settings)
        self._query_rewrite_client = BamlQueryRewriteClient(settings, graphiti_settings)
        logger.info("GraphitiRAGPipeline initialized for group '{}'", group_id)

    async def run(self, query: str, max_facts: int | None = None) -> GraphitiRAGResult:
        """Execute the full pipeline for a single query.

        Args:
            query: Natural-language question, in whatever phrasing the user typed it.
            max_facts: Override the default maximum number of selected context facts.

        Returns:
            Result containing the answer, the facts it was grounded in, and any citations to
            documents outside the indexed corpus.
        """
        rewritten_query = await self._query_rewrite_client.rewrite(query)
        if rewritten_query != query:
            logger.debug("Rewrote query {!r} -> {!r}", query, rewritten_query)

        candidates = await search_facts(
            self._graphiti_client, rewritten_query, self._group_id, self._settings.search_num_results
        )
        selected = select_minimal_facts(candidates, max_facts or self._settings.max_context_facts)
        answer = generate_answer(query, selected, self._generation_client)

        unindexed_citations = find_unindexed_document_citations(
            [answer, *(fact.text for fact in selected)], self._known_so_hieu
        )
        if unindexed_citations:
            logger.warning("Answer cites document(s) outside the indexed corpus: {}", unindexed_citations)

        logger.info("Pipeline completed for: {:.60s}", query)
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
