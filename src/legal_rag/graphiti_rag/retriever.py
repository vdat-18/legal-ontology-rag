"""Retrieve candidate facts from the Graphiti graph with the provenance of every supporting episode."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search import search as graphiti_search
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from loguru import logger

from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder

EpisodeLookup = Callable[[GraphDriver, list[str]], Awaitable[list[EpisodicNode]]]
_DEFAULT_EPISODE_LOOKUP = EpisodicNode.get_by_uuids
_EXPANSION_RRF_K = 60
_FactKey = tuple[str, tuple[tuple[str, str, str], ...]]


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


class MissingEffectiveProvenanceError(ValueError):
    """Raised when a dated fact has no supporting episode effective by that date."""

    def __init__(self, edge_uuid: str, as_of_date: date) -> None:
        """Identify the edge whose resolved provenance is entirely from the future."""
        self.edge_uuid = edge_uuid
        self.as_of_date = as_of_date
        super().__init__(f"Edge {edge_uuid} has no supporting episode effective by {as_of_date.isoformat()}")


@dataclass(frozen=True, slots=True)
class FactSource:
    """One supporting episode's provenance for a retrieved fact: its source chunk and document.

    ``chunk_id`` is the chunker's stable ``Chunk.chunk_id`` (the indexer names each episode by it),
    so a fact traces back to the exact retrieval chunk it was extracted from, not to the edge's own
    relation name.
    """

    chunk_id: str
    source_name: str
    text: str = ""
    """The source passage this fact was extracted from.

    An extracted fact is a distilled claim and loses the wording around it — the article heading
    that says which vehicle a penalty applies to, the clause numbering, the exceptions. Carrying
    the passage lets an answer quote the provision instead of paraphrasing a paraphrase.
    """


@dataclass(frozen=True, slots=True)
class RetrievedFact:
    """One fact (graph edge) with the structured provenance of every episode that supports it.

    An edge can be reinforced by more than one episode; ``sources`` lists all of them so the fact
    can cite every supporting chunk, not just the first.
    """

    fact: str
    sources: tuple[FactSource, ...]
    as_of_dates: tuple[date, ...] = ()


async def _search_edges(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    as_of_date: date | None,
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
    violation — which plain text similarity would rank far lower. Reciprocal Rank Fusion combines
    the text and neighbourhood rankings so graph-adjacent facts can enter the selected top-k while
    facts supported by both searches receive the strongest rank.
    """
    if not isinstance(graphiti_client.embedder, QwenGraphitiEmbedder):
        msg = "Graphiti retrieval requires QwenGraphitiEmbedder for query-side embedding"
        raise TypeError(msg)
    await graphiti_client.embedder.validate_index(graphiti_client.driver, group_id)
    query_vector = await graphiti_client.embedder.create_query(query.replace("\n", " "))

    config = (EDGE_HYBRID_SEARCH_CROSS_ENCODER if rerank_with_cross_encoder else EDGE_HYBRID_SEARCH_RRF).model_copy(
        deep=True
    )
    config.limit = num_results
    search_filters = _search_filters_for_date(as_of_date)
    results = await graphiti_search(
        graphiti_client.clients,
        query,
        [group_id],
        config,
        search_filters,
        query_vector=query_vector,
        driver=graphiti_client.driver,
    )
    edges = results.edges
    if rerank_with_cross_encoder:
        return edges
    if not expand_around_top_fact or not edges:
        return edges

    neighbourhood_config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
    neighbourhood_config.limit = num_results
    neighbourhood = await graphiti_search(
        graphiti_client.clients,
        query,
        [group_id],
        neighbourhood_config,
        search_filters,
        center_node_uuid=edges[0].source_node_uuid,
        query_vector=query_vector,
        driver=graphiti_client.driver,
    )
    expanded = _fuse_edge_rankings(edges, neighbourhood.edges, limit=num_results)
    added = len({edge.uuid for edge in expanded} - {edge.uuid for edge in edges})
    logger.debug("Graph expansion promoted {} adjacent facts into the result set", added)
    return expanded


def _search_filters_for_date(as_of_date: date | None) -> SearchFilters:
    """Restrict facts to those effective at the end of one UTC calendar date."""
    if as_of_date is None:
        return SearchFilters()

    cutoff = datetime.combine(as_of_date, time.max, tzinfo=UTC)
    return SearchFilters(
        valid_at=[
            [DateFilter(date=cutoff, comparison_operator=ComparisonOperator.less_than_equal)],
        ],
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=cutoff, comparison_operator=ComparisonOperator.greater_than)],
        ],
    )


def _fuse_edge_rankings(
    base_edges: Sequence[EntityEdge],
    neighbourhood_edges: Sequence[EntityEdge],
    *,
    limit: int,
) -> list[EntityEdge]:
    """Fuse text and graph-neighbourhood rankings so expansion can affect the selected top-k."""
    scores: dict[str, float] = {}
    edges_by_uuid: dict[str, EntityEdge] = {}
    for ranking in (base_edges, neighbourhood_edges):
        for rank, edge in enumerate(ranking, 1):
            scores[edge.uuid] = scores.get(edge.uuid, 0.0) + 1.0 / (_EXPANSION_RRF_K + rank)
            edges_by_uuid[edge.uuid] = edge
    ranked_ids = sorted(scores, key=lambda edge_uuid: (-scores[edge_uuid], edge_uuid))[:limit]
    return [edges_by_uuid[edge_uuid] for edge_uuid in ranked_ids]


async def search_facts(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    as_of_date: date | None = None,
    expand_around_top_fact: bool = True,
    rerank_with_cross_encoder: bool = False,
    get_episodes_by_uuids: EpisodeLookup = _DEFAULT_EPISODE_LOOKUP,
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
        as_of_date=as_of_date,
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
        effective_episode_uuids = edge.episodes
        if as_of_date is not None:
            cutoff = datetime.combine(as_of_date, time.max, tzinfo=UTC)
            effective_episode_uuids = [
                uuid
                for uuid in edge.episodes
                if _episode_valid_at(episodes_by_uuid[uuid], edge_uuid=edge.uuid) <= cutoff
            ]
            if not effective_episode_uuids:
                raise MissingEffectiveProvenanceError(edge.uuid, as_of_date)
        sources = tuple(
            FactSource(
                chunk_id=episodes_by_uuid[uuid].name,
                source_name=episodes_by_uuid[uuid].source_description,
                text=episodes_by_uuid[uuid].content or "",
            )
            for uuid in effective_episode_uuids
        )
        facts.append(
            RetrievedFact(
                fact=edge.fact,
                sources=sources,
                as_of_dates=(as_of_date,) if as_of_date is not None else (),
            )
        )

    logger.debug("Retrieved {} candidate facts", len(facts))
    return facts


def _episode_valid_at(episode: EpisodicNode, *, edge_uuid: str) -> datetime:
    """Return one timezone-aware episode validity timestamp for temporal provenance filtering."""
    valid_at = episode.valid_at
    if not isinstance(valid_at, datetime) or valid_at.tzinfo is None:
        msg = f"Edge {edge_uuid} has supporting episode {episode.uuid} without a timezone-aware valid_at"
        raise TypeError(msg)
    return valid_at.astimezone(UTC)


async def search_facts_at_dates(
    graphiti_client: Graphiti,
    query: str,
    group_id: str,
    num_results: int,
    *,
    as_of_dates: Sequence[date] = (),
    expand_around_top_fact: bool = True,
    rerank_with_cross_encoder: bool = False,
    get_episodes_by_uuids: EpisodeLookup = _DEFAULT_EPISODE_LOOKUP,
) -> list[RetrievedFact]:
    """Search each relevant legal date and fuse the fact rankings with reciprocal rank fusion.

    An empty date sequence intentionally performs one unfiltered search. This is required for
    temporal comparison questions whose exact target date is unknown: forcing a campaign date
    would silently discard historical or future versions the question asks to compare.
    """
    unique_dates = tuple(dict.fromkeys(as_of_dates))
    episode_lookup_kwargs = (
        {"get_episodes_by_uuids": get_episodes_by_uuids} if get_episodes_by_uuids is not _DEFAULT_EPISODE_LOOKUP else {}
    )
    if len(unique_dates) <= 1:
        return await search_facts(
            graphiti_client,
            query,
            group_id,
            num_results,
            as_of_date=unique_dates[0] if unique_dates else None,
            expand_around_top_fact=expand_around_top_fact,
            rerank_with_cross_encoder=rerank_with_cross_encoder,
            **episode_lookup_kwargs,
        )

    rankings = [
        await search_facts(
            graphiti_client,
            query,
            group_id,
            num_results,
            as_of_date=as_of_date,
            expand_around_top_fact=expand_around_top_fact,
            rerank_with_cross_encoder=rerank_with_cross_encoder,
            **episode_lookup_kwargs,
        )
        for as_of_date in unique_dates
    ]
    return _fuse_fact_rankings(rankings, limit=num_results)


def _fuse_fact_rankings(rankings: Sequence[Sequence[RetrievedFact]], *, limit: int) -> list[RetrievedFact]:
    """Fuse dated rankings while reserving evidence unique to every compared version."""
    scores: dict[_FactKey, float] = {}
    facts_by_key: dict[_FactKey, RetrievedFact] = {}
    dates_by_key: dict[_FactKey, set[date]] = {}
    ranking_keys: list[list[_FactKey]] = []
    for ranking in rankings:
        keys: list[_FactKey] = []
        for rank, fact in enumerate(ranking, 1):
            source_key = tuple(sorted((source.chunk_id, source.source_name, source.text) for source in fact.sources))
            key = (fact.fact, source_key)
            scores[key] = max(scores.get(key, 0.0), 1.0 / (_EXPANSION_RRF_K + rank))
            facts_by_key[key] = fact
            dates_by_key.setdefault(key, set()).update(fact.as_of_dates)
            if key not in keys:
                keys.append(key)
        ranking_keys.append(keys)

    occurrence_count = {key: sum(key in keys for keys in ranking_keys) for key in scores}
    ranked_keys: list[_FactKey] = []
    for keys in ranking_keys:
        exclusive = [key for key in keys if occurrence_count[key] == 1 and key not in ranked_keys]
        fallback = [key for key in keys if key not in ranked_keys]
        candidate = (exclusive or fallback)[:1]
        ranked_keys.extend(candidate)
        if len(ranked_keys) == limit:
            break
    if len(ranked_keys) < limit:
        remaining = sorted(
            (key for key in scores if key not in ranked_keys),
            key=lambda key: (-scores[key], key),
        )
        ranked_keys.extend(remaining[: limit - len(ranked_keys)])
    return [
        RetrievedFact(
            fact=facts_by_key[key].fact,
            sources=facts_by_key[key].sources,
            as_of_dates=tuple(sorted(dates_by_key[key])),
        )
        for key in ranked_keys
    ]
