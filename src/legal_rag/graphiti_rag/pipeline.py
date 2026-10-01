"""Graphiti RAG pipeline: rewrite the query, search the graph, keep the top-ranked facts, generate."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from loguru import logger

from legal_rag.baml_client.types import ConversationTurn
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.graphiti_rag.client import build_graphiti
from legal_rag.graphiti_rag.generator import generate_answer
from legal_rag.graphiti_rag.provenance import BamlDocumentCitationClient, find_unindexed_document_citations
from legal_rag.graphiti_rag.query_rewriter import BamlQueryRewriteClient
from legal_rag.graphiti_rag.retriever import RetrievedFact
from legal_rag.graphiti_rag.retriever import search_facts_at_dates as search_facts
from legal_rag.graphiti_rag.suggestions import BamlFollowUpClient
from legal_rag.naive_rag.generator import BamlGenerationClient
from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings


@dataclass(frozen=True, slots=True)
class GraphitiRAGResult:
    """Container for one Graphiti RAG pipeline response."""

    query: str
    rewritten_query: str
    answer: str
    sources: tuple[RetrievedFact, ...] = ()
    unindexed_citations: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()


async def _no_citations() -> list[str]:
    """Stand in for the citation check when the indexed corpus is unknown."""
    return []


class GraphitiRAGPipeline:
    """End-to-end rewrite-search-then-generate pipeline over the Graphiti knowledge graph.

    Retrieval order comes from Graphiti's hybrid search — rank fusion by default, or a
    cross-encoder when ``rerank_with_cross_encoder`` is set. The pipeline keeps the top
    ``max_context_facts`` of that ranking rather than re-scoring facts with any handcrafted rule.
    """

    def __init__(
        self,
        settings: Settings,
        neo4j_settings: Neo4jSettings,
        graphiti_settings: GraphitiSettings,
        group_id: str,
        known_document_numbers: frozenset[str] | None = None,
    ) -> None:
        """Initialize the pipeline from application and Graphiti settings, scoped to one group.

        Args:
            settings: Application LLM connection settings.
            neo4j_settings: Shared Neo4j connection settings.
            graphiti_settings: Graphiti embedding profile and retrieval settings.
            group_id: The Graphiti group to search within.
            known_document_numbers: Document numbers actually indexed into this group, or ``None`` when
                corpus coverage is unknown and citation coverage checking must be skipped.
        """
        self._settings = graphiti_settings
        self._group_id = group_id
        self._known_document_numbers = known_document_numbers
        profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
        self._graphiti_client = build_graphiti(settings, neo4j_settings, graphiti_settings, profile)
        self._generation_client = BamlGenerationClient(settings, graphiti_settings)
        self._query_rewrite_client = BamlQueryRewriteClient(settings, graphiti_settings)
        self._citation_client = BamlDocumentCitationClient(settings, graphiti_settings)
        self._follow_up_client = BamlFollowUpClient(settings, graphiti_settings)
        logger.info("GraphitiRAGPipeline initialized for group '{}'", group_id)

    async def run(
        self,
        query: str,
        max_facts: int | None = None,
        history: Sequence[ConversationTurn] = (),
        *,
        as_of_date: date | None = None,
        as_of_dates: Sequence[date] = (),
        include_auxiliary_outputs: bool = True,
    ) -> GraphitiRAGResult:
        """Execute the full pipeline for a single query.

        Args:
            query: Natural-language question, in whatever phrasing the user typed it.
            max_facts: Override the default maximum number of selected context facts. Must be
                positive when given.
            history: Earlier turns of this conversation, oldest first. A follow-up question is
                resolved against them before the graph is searched.
            as_of_date: Restrict retrieved facts to those effective at the end of this UTC date.
            as_of_dates: Search each relevant legal-version date and fuse their rankings. This is
                mutually exclusive with ``as_of_date``.
            include_auxiliary_outputs: Generate product-only citation diagnostics and follow-up
                suggestions. Evaluation disables these calls so answer latency and token usage
                cover the same retrieval-and-answer scope as the other pipelines.

        Returns:
            Result containing the answer, the facts it was grounded in, and any citations to
            documents outside the indexed corpus.

        Raises:
            ValueError: If *max_facts* is not positive or singular/plural dates are both supplied.
        """
        if max_facts is not None and max_facts < 1:
            msg = f"max_facts must be positive, got {max_facts}"
            raise ValueError(msg)
        if as_of_date is not None and as_of_dates:
            msg = "as_of_date and as_of_dates are mutually exclusive"
            raise ValueError(msg)

        rewritten_query = await self._query_rewrite_client.rewrite(query, history)
        if rewritten_query != query:
            logger.debug("Query rewrite completed")

        candidates = await search_facts(
            self._graphiti_client,
            rewritten_query,
            self._group_id,
            self._settings.search_num_results,
            as_of_dates=tuple(as_of_dates) if as_of_dates else ((as_of_date,) if as_of_date is not None else ()),
            expand_around_top_fact=self._settings.expand_around_top_fact,
            rerank_with_cross_encoder=self._settings.rerank_with_cross_encoder,
        )
        fact_budget = self._settings.max_context_facts if max_facts is None else max_facts
        selected = candidates[:fact_budget]
        answer = await asyncio.to_thread(generate_answer, rewritten_query, selected, self._generation_client)

        if not include_auxiliary_outputs:
            logger.info("Graphiti pipeline completed")
            return GraphitiRAGResult(
                query=query,
                rewritten_query=rewritten_query,
                answer=answer,
                sources=tuple(selected),
            )

        # Both only need the finished answer, so the second one is free in wall-clock terms.
        citation_task = (
            asyncio.to_thread(
                find_unindexed_document_citations,
                [answer, *(fact.fact for fact in selected)],
                self._known_document_numbers,
                self._citation_client,
            )
            if self._known_document_numbers is not None
            else _no_citations()
        )
        unindexed_citations, suggestions = await asyncio.gather(
            citation_task,
            asyncio.to_thread(self._follow_up_client.suggest, query, answer),
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
            suggestions=tuple(suggestions),
        )

    async def close(self) -> None:
        """Release the underlying Graphiti/Neo4j connection."""
        await self._graphiti_client.close()
