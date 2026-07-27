"""Retrieve candidate facts from the Graphiti graph and select a minimal, non-redundant set."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import EpisodicNode
from loguru import logger

EpisodeLookup = Callable[[GraphDriver, list[str]], Awaitable[list[EpisodicNode]]]


@dataclass(frozen=True, slots=True)
class RetrievedFact:
    """One fact (graph edge) retrieved for a query, duck-typed to feed the shared BAML generator.

    ``chunk_id``/``text``/``source_name`` match the fields the naive RAG generator already
    builds a ``RetrievedChunkContext`` from, so the same generation client can be reused as-is.
    """

    chunk_id: str
    text: str
    source_name: str
    source_node_uuid: str
    target_node_uuid: str


async def search_facts(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    get_episodes_by_uuids: EpisodeLookup = EpisodicNode.get_by_uuids,
) -> list[RetrievedFact]:
    """Run Graphiti's hybrid search and return the raw, relevance-ranked candidate facts.

    An edge's real provenance is the episode it was extracted from at indexing time — its own
    ``name``/``uuid`` only describe the relation type and the edge's own identity, not the
    source document or chunk. An edge can carry more than one supporting episode; the first is
    used as the cited source.
    """
    edges = await graphiti_client.search(query, group_ids=[group_id], num_results=num_results)

    episode_uuids = list({edge.episodes[0] for edge in edges})
    episodes_by_uuid = {
        episode.uuid: episode for episode in await get_episodes_by_uuids(graphiti_client.driver, episode_uuids)
    }

    facts = [
        RetrievedFact(
            chunk_id=episodes_by_uuid[edge.episodes[0]].name,
            text=edge.fact,
            source_name=episodes_by_uuid[edge.episodes[0]].source_description,
            source_node_uuid=edge.source_node_uuid,
            target_node_uuid=edge.target_node_uuid,
        )
        for edge in edges
    ]
    logger.debug("Retrieved {} candidate facts for query: {:.60s}", len(facts), query)
    return facts


def select_minimal_facts(facts: Sequence[RetrievedFact], max_facts: int) -> list[RetrievedFact]:
    """Greedily select a minimal, non-redundant subset of *facts*, up to *max_facts*.

    Approximates the "minimal hyperedge cover" idea from ontology-grounded RAG (e.g. OG-RAG,
    arXiv:2412.15235) without a full optimization solver: a fact's "coverage" is the pair of
    entity uuids it connects, and a candidate is only added once it introduces at least one
    entity uuid not already covered by a higher-ranked, already-selected fact — since *facts*
    is pre-ranked by relevance (the order Graphiti's hybrid search returns), this drops facts
    that are redundant with better-ranked ones instead of just truncating to top-N.
    """
    selected: list[RetrievedFact] = []
    covered_uuids: set[str] = set()

    for fact in facts:
        if len(selected) >= max_facts:
            break
        new_coverage = {fact.source_node_uuid, fact.target_node_uuid} - covered_uuids
        if not selected or new_coverage:
            selected.append(fact)
            covered_uuids |= {fact.source_node_uuid, fact.target_node_uuid}

    return selected
