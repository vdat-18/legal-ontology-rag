"""Graphiti RAG pipeline: search the graph, select minimal facts, then generate."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.graphiti_rag.client import build_graphiti
from legal_rag.graphiti_rag.generator import generate_answer
from legal_rag.graphiti_rag.retriever import RetrievedFact, search_facts, select_minimal_facts
from legal_rag.naive_rag.generator import BamlGenerationClient
from legal_rag.settings import GraphitiSettings, Settings


@dataclass(frozen=True, slots=True)
class GraphitiRAGResult:
    """Container for one Graphiti RAG pipeline response."""

    query: str
    answer: str
    sources: tuple[RetrievedFact, ...] = ()


class GraphitiRAGPipeline:
    """End-to-end search-select-then-generate pipeline over the Graphiti knowledge graph."""

    def __init__(self, settings: Settings, graphiti_settings: GraphitiSettings, group_id: str) -> None:
        """Initialize the pipeline from application and Graphiti settings, scoped to one group."""
        self._settings = graphiti_settings
        self._group_id = group_id
        profile = load_configured_embedding_profile(graphiti_settings.embedding_profile)
        self._graphiti_client = build_graphiti(settings, graphiti_settings, profile)
        self._generation_client = BamlGenerationClient(settings, graphiti_settings)
        logger.info("GraphitiRAGPipeline initialized for group '{}'", group_id)

    async def run(self, query: str, max_facts: int | None = None) -> GraphitiRAGResult:
        """Execute the full pipeline for a single query.

        Args:
            query: Natural-language question.
            max_facts: Override the default maximum number of selected context facts.

        Returns:
            Result containing the answer and the facts it was grounded in.
        """
        candidates = await search_facts(
            self._graphiti_client, query, self._group_id, self._settings.search_num_results
        )
        selected = select_minimal_facts(candidates, max_facts or self._settings.max_context_facts)
        answer = generate_answer(query, selected, self._generation_client)
        logger.info("Pipeline completed for: {:.60s}", query)
        return GraphitiRAGResult(query=query, answer=answer, sources=tuple(selected))

    async def close(self) -> None:
        """Release the underlying Graphiti/Neo4j connection."""
        await self._graphiti_client.close()
