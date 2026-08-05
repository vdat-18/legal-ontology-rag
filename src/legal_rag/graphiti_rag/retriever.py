"""Retrieve candidate facts from the Graphiti graph with the provenance of every supporting episode."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
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


async def _search_edges(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    expand_around_top_fact: bool,
    rerank_with_cross_encoder: bool,
) -> list[EntityEdge]:
    """Search the graph, ranking either by rank fusion or by a cross-encoder over the graph.

    ``rerank_with_cross_encoder`` selects Graphiti's cross-encoder recipe, which also adds
    breadth-first traversal to the candidate set, so relevance is judged by a model reading each
    fact against the question rather than by fusing two lexical/vector rankings. It costs a
    scoring call per candidate, which is why it is switchable and measured rather than assumed
    better. The plain path fuses BM25 and vector ranks (RRF) and never consults the cross-encoder,
    despite one being configured on the client.

    Plain hybrid search ranks an edge only by how well its own text matches the query, so it uses
    the graph as a store of independent facts. Graphiti reranks by graph distance when given a
    center node, so a second pass centred on the best fact's source entity promotes facts that are
    structurally adjacent to it — the penalty, remedy, and provision hanging off the same
    violation — which plain text similarity would rank far lower. Results from the first pass keep
    their order and the expansion is appended, so this can only add context, never displace the
    best text match.
    """
    if rerank_with_cross_encoder:
        config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
        config.limit = num_results
        results = await graphiti_client.search_(query, config, group_ids=[group_id])
        return results.edges

    edges = await graphiti_client.search(query, group_ids=[group_id], num_results=num_results)
    if not expand_around_top_fact or not edges:
        return edges

    neighbourhood = await graphiti_client.search(
        query, group_ids=[group_id], num_results=num_results, center_node_uuid=edges[0].source_node_uuid
    )
    seen = {edge.uuid for edge in edges}
    expanded = [*edges, *(edge for edge in neighbourhood if edge.uuid not in seen)]
    logger.debug("Graph expansion added {} facts around the top fact's entity", len(expanded) - len(edges))
    return expanded


async def search_facts(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    expand_around_top_fact: bool = True,
    rerank_with_cross_encoder: bool = False,
    get_episodes_by_uuids: EpisodeLookup = EpisodicNode.get_by_uuids,
) -> list[RetrievedFact]:
    """Run Graphiti's hybrid search and return the relevance-ranked candidate facts with provenance.

    An edge's real provenance is the episode(s) it was extracted from at indexing time — its own
    ``name``/``uuid`` only describe the relation type and the edge's own identity, not the source
    document or chunk. Every supporting episode that resolves contributes one ``FactSource``.

    Raises:
        MissingProvenanceError: If an edge has no resolvable supporting episode at all.
    """
    edges = await _search_edges(
        graphiti_client,
        query,
        group_id,
        num_results,
        expand_around_top_fact=expand_around_top_fact,
        rerank_with_cross_encoder=rerank_with_cross_encoder,
    )

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
