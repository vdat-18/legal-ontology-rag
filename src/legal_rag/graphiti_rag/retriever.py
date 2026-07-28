"""Retrieve candidate facts from the Graphiti graph and select a minimal, non-redundant set."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import EpisodicNode
from loguru import logger

EpisodeLookup = Callable[[GraphDriver, list[str]], Awaitable[list[EpisodicNode]]]


class MissingProvenanceError(RuntimeError):
    """Raised when a retrieved edge's supporting episodes cannot be resolved.

    Every indexed edge is written together with the episode it was extracted from, so an
    unresolvable episode means the graph is inconsistent. Answering from the remaining facts
    would silently drop evidence and produce a confidently-cited but incomplete answer, so
    retrieval fails loudly instead.
    """

    def __init__(self, edge_uuid: str, missing_episode_uuids: Sequence[str]) -> None:
        """Record which edge could not be attributed, and which episodes were missing."""
        self.edge_uuid = edge_uuid
        self.missing_episode_uuids = tuple(missing_episode_uuids)
        super().__init__(
            f"Edge {edge_uuid} has no resolvable supporting episode "
            f"(missing episode uuids: {', '.join(self.missing_episode_uuids) or 'none recorded'})"
        )


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
    source document or chunk. An edge can carry more than one supporting episode; the first
    resolvable one is used as the cited source.

    Raises:
        MissingProvenanceError: If any retrieved edge has no resolvable supporting episode.
    """
    edges = await graphiti_client.search(query, group_ids=[group_id], num_results=num_results)

    episode_uuids = list({uuid for edge in edges for uuid in edge.episodes})
    episodes_by_uuid = {
        episode.uuid: episode for episode in await get_episodes_by_uuids(graphiti_client.driver, episode_uuids)
    }

    facts = []
    for edge in edges:
        source_episode = next((episodes_by_uuid[uuid] for uuid in edge.episodes if uuid in episodes_by_uuid), None)
        if source_episode is None:
            raise MissingProvenanceError(edge.uuid, edge.episodes)
        facts.append(
            RetrievedFact(
                chunk_id=source_episode.name,
                text=edge.fact,
                source_name=source_episode.source_description,
                source_node_uuid=edge.source_node_uuid,
                target_node_uuid=edge.target_node_uuid,
            )
        )
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
