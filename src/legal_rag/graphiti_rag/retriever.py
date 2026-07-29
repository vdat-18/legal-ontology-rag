"""Retrieve candidate facts from the Graphiti graph with the provenance of every supporting episode."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import EpisodicNode
from loguru import logger

EpisodeLookup = Callable[[GraphDriver, list[str]], Awaitable[list[EpisodicNode]]]


class MissingProvenanceError(ValueError):
    """Raised when a retrieved edge's supporting episodes cannot be fully resolved.

    Every indexed edge is written together with the episodes it was extracted from, and episodes
    are never deleted, so any episode that fails to resolve means the graph is inconsistent.
    Citing only the episodes that happen to resolve would drop evidence silently and ground the
    answer in a partial fact set, so retrieval fails loudly instead of degrading.
    """

    def __init__(self, edge_uuid: str, missing_episode_uuids: Sequence[str]) -> None:
        """Record which edge could not be attributed, and which episodes failed to resolve."""
        self.edge_uuid = edge_uuid
        self.missing_episode_uuids = tuple(missing_episode_uuids)
        super().__init__(
            f"Edge {edge_uuid} has unresolvable provenance "
            f"(missing episode uuids: {', '.join(self.missing_episode_uuids) or 'edge records no episode'})"
        )


@dataclass(frozen=True, slots=True)
class FactSource:
    """One supporting episode's provenance for a retrieved fact: its source chunk and document.

    ``chunk_id`` is the chunker's stable ``Chunk.chunk_id`` (the indexer names each episode by it),
    so a fact traces back to the exact retrieval chunk it was extracted from, not to the edge's own
    relation name.
    """

    chunk_id: str
    source_name: str


@dataclass(frozen=True, slots=True)
class RetrievedFact:
    """One fact (graph edge) with the structured provenance of every episode that supports it.

    An edge can be reinforced by more than one episode; ``sources`` lists all of them so the fact
    can cite every supporting chunk, not just the first.
    """

    fact: str
    sources: tuple[FactSource, ...]


async def search_facts(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    get_episodes_by_uuids: EpisodeLookup = EpisodicNode.get_by_uuids,
) -> list[RetrievedFact]:
    """Run Graphiti's hybrid search and return the relevance-ranked candidate facts with provenance.

    An edge's real provenance is the episode(s) it was extracted from at indexing time — its own
    ``name``/``uuid`` only describe the relation type and the edge's own identity, not the source
    document or chunk. Every supporting episode that resolves contributes one ``FactSource``.

    Raises:
        MissingProvenanceError: If an edge has no resolvable supporting episode at all.
    """
    edges = await graphiti_client.search(query, group_ids=[group_id], num_results=num_results)

    episode_uuids = list({uuid for edge in edges for uuid in edge.episodes})
    episodes_by_uuid = {
        episode.uuid: episode for episode in await get_episodes_by_uuids(graphiti_client.driver, episode_uuids)
    }

    facts: list[RetrievedFact] = []
    for edge in edges:
        missing_episode_uuids = [uuid for uuid in edge.episodes if uuid not in episodes_by_uuid]
        if missing_episode_uuids or not edge.episodes:
            raise MissingProvenanceError(edge.uuid, missing_episode_uuids)
        sources = tuple(
            FactSource(chunk_id=episodes_by_uuid[uuid].name, source_name=episodes_by_uuid[uuid].source_description)
            for uuid in edge.episodes
        )
        facts.append(RetrievedFact(fact=edge.fact, sources=sources))

    logger.debug("Retrieved {} candidate facts", len(facts))
    return facts
