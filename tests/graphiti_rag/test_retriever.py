"""Tests for searching facts and resolving their structured provenance from the Graphiti graph."""

from __future__ import annotations

import asyncio

import pytest

from legal_rag.graphiti_rag.retriever import (
    FactSource,
    MissingProvenanceError,
    RetrievedFact,
    search_facts,
)


class _FakeEdge:
    """A minimal stand-in for graphiti_core's EntityEdge."""

    def __init__(
        self,
        uuid: str,
        fact: str,
        name: str,
        episodes: list[str],
        source_node_uuid: str = "node-source",
        target_node_uuid: str = "node-target",
    ) -> None:
        self.uuid = uuid
        self.fact = fact
        self.name = name
        self.episodes = episodes
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid


class _FakeEpisode:
    """A minimal stand-in for graphiti_core's EpisodicNode."""

    def __init__(self, uuid: str, name: str, source_description: str, content: str = "") -> None:
        self.uuid = uuid
        self.name = name
        self.source_description = source_description
        self.content = content


class _FakeGraphiti:
    """Record search() calls and return a fixed list of fake edges."""

    def __init__(self, edges: list[_FakeEdge]) -> None:
        self._edges = edges
        self.calls: list[dict[str, object]] = []
        self.driver = object()

    async def search(self, query: str, **kwargs: object) -> list[_FakeEdge]:
        """Record the call and return the pre-seeded edges."""
        self.calls.append({"query": query, **kwargs})
        return self._edges


def _fake_episode_lookup(episodes: list[_FakeEpisode]):
    """Build a fake ``get_episodes_by_uuids`` that resolves only the uuids it knows about.

    Like the real lookup, it returns fewer nodes than requested when some uuids don't exist,
    exercising the retriever's guard against missing episodes instead of raising ``KeyError``.
    """
    by_uuid = {episode.uuid: episode for episode in episodes}

    async def _lookup(_driver: object, uuids: list[str]) -> list[_FakeEpisode]:
        return [by_uuid[uuid] for uuid in uuids if uuid in by_uuid]

    return _lookup


def test_search_facts_cites_the_source_episode_not_the_edges_own_name_and_uuid() -> None:
    """Provenance comes from the edge's supporting episode, not its own relation name/uuid."""
    edges = [_FakeEdge("e1", "Phạt tiền 4-6 triệu", "PENALIZES", episodes=["ep1"])]
    episodes = [_FakeEpisode("ep1", "chunk-abc123", "168_2024_ND-CP.docx")]
    graphiti = _FakeGraphiti(edges)

    facts = asyncio.run(
        search_facts(
            graphiti,
            "vượt đèn đỏ bị phạt bao nhiêu?",
            "group-1",
            30,
            expand_around_top_fact=False,
            get_episodes_by_uuids=_fake_episode_lookup(episodes),
        )
    )

    assert facts == [
        RetrievedFact(
            fact="Phạt tiền 4-6 triệu",
            sources=(FactSource(chunk_id="chunk-abc123", source_name="168_2024_ND-CP.docx"),),
        )
    ]
    assert graphiti.calls == [{"query": "vượt đèn đỏ bị phạt bao nhiêu?", "group_ids": ["group-1"], "num_results": 30}]


def test_search_facts_cites_every_supporting_episode_of_an_edge() -> None:
    """An edge reinforced across several episodes carries the provenance of all of them."""
    edges = [_FakeEdge("e1", "some fact", "PENALIZES", episodes=["ep1", "ep2"])]
    episodes = [
        _FakeEpisode("ep1", "chunk-1", "doc.docx"),
        _FakeEpisode("ep2", "chunk-9", "doc.docx"),
    ]
    graphiti = _FakeGraphiti(edges)

    facts = asyncio.run(
        search_facts(
            graphiti,
            "q",
            "group-1",
            30,
            expand_around_top_fact=False,
            get_episodes_by_uuids=_fake_episode_lookup(episodes),
        )
    )

    assert facts[0].sources == (
        FactSource(chunk_id="chunk-1", source_name="doc.docx"),
        FactSource(chunk_id="chunk-9", source_name="doc.docx"),
    )


def test_search_facts_raises_when_only_some_episodes_of_an_edge_resolve() -> None:
    """Partial provenance is still a graph inconsistency: it must not silently lose a source."""
    edges = [_FakeEdge("e1", "some fact", "PENALIZES", episodes=["ep1", "ep-missing"])]
    episodes = [_FakeEpisode("ep1", "chunk-1", "doc.docx")]
    graphiti = _FakeGraphiti(edges)

    with pytest.raises(MissingProvenanceError) as error:
        asyncio.run(
            search_facts(
                graphiti,
                "q",
                "group-1",
                30,
                expand_around_top_fact=False,
                get_episodes_by_uuids=_fake_episode_lookup(episodes),
            )
        )

    assert error.value.edge_uuid == "e1"
    assert error.value.missing_episode_uuids == ("ep-missing",)


def test_search_facts_raises_for_an_edge_that_records_no_episode() -> None:
    """An edge carrying no supporting episode fails loudly instead of being silently dropped."""
    edges = [_FakeEdge("e1", "unciteable fact", "PENALIZES", episodes=[])]
    graphiti = _FakeGraphiti(edges)

    with pytest.raises(MissingProvenanceError) as error:
        asyncio.run(
            search_facts(
                graphiti,
                "q",
                "group-1",
                30,
                expand_around_top_fact=False,
                get_episodes_by_uuids=_fake_episode_lookup([]),
            )
        )

    assert error.value.edge_uuid == "e1"
    assert error.value.missing_episode_uuids == ()


def test_search_facts_raises_when_every_episode_of_an_edge_is_unresolvable() -> None:
    """A graph inconsistency surfaces with the edge and missing episode uuids, not a degraded answer."""
    edges = [_FakeEdge("e1", "unciteable fact", "PENALIZES", episodes=["ep-missing"])]
    graphiti = _FakeGraphiti(edges)

    with pytest.raises(MissingProvenanceError) as error:
        asyncio.run(
            search_facts(
                graphiti,
                "q",
                "group-1",
                30,
                expand_around_top_fact=False,
                get_episodes_by_uuids=_fake_episode_lookup([]),
            )
        )

    assert error.value.edge_uuid == "e1"
    assert error.value.missing_episode_uuids == ("ep-missing",)


class _TwoPassGraphiti:
    """Return different edges for the plain pass and the graph-proximity pass."""

    def __init__(self, first: list[_FakeEdge], neighbourhood: list[_FakeEdge]) -> None:
        self._first = first
        self._neighbourhood = neighbourhood
        self.calls: list[dict[str, object]] = []
        self.driver = object()

    async def search(self, query: str, **kwargs: object) -> list[_FakeEdge]:
        """Return the neighbourhood only when reranking around a center node."""
        self.calls.append({"query": query, **kwargs})
        return self._neighbourhood if "center_node_uuid" in kwargs else self._first


def test_graph_expansion_reranks_around_the_top_facts_entity() -> None:
    """The second pass is centred on the best fact's entity, which is what uses the graph."""
    first = [_FakeEdge("e1", "vi phạm", "PENALIZES", episodes=["ep1"], source_node_uuid="violation-1")]
    graphiti = _TwoPassGraphiti(first, neighbourhood=[])

    asyncio.run(
        search_facts(
            graphiti,
            "q",
            "group-1",
            30,
            get_episodes_by_uuids=_fake_episode_lookup([_FakeEpisode("ep1", "chunk-1", "doc.docx")]),
        )
    )

    assert graphiti.calls[1]["center_node_uuid"] == "violation-1"


def test_graph_expansion_appends_structurally_adjacent_facts_without_duplicates() -> None:
    """Adjacent facts are added after the text matches, and an edge seen twice is kept once."""
    first = [_FakeEdge("e1", "vi phạm", "PENALIZES", episodes=["ep1"], source_node_uuid="violation-1")]
    neighbourhood = [
        _FakeEdge("e1", "vi phạm", "PENALIZES", episodes=["ep1"], source_node_uuid="violation-1"),
        _FakeEdge("e2", "biện pháp khắc phục", "REQUIRES_REMEDY", episodes=["ep2"], source_node_uuid="violation-1"),
    ]
    graphiti = _TwoPassGraphiti(first, neighbourhood)
    episodes = [_FakeEpisode("ep1", "chunk-1", "doc.docx"), _FakeEpisode("ep2", "chunk-2", "doc.docx")]

    facts = asyncio.run(
        search_facts(graphiti, "q", "group-1", 30, get_episodes_by_uuids=_fake_episode_lookup(episodes))
    )

    assert [fact.fact for fact in facts] == ["vi phạm", "biện pháp khắc phục"]


def test_graph_expansion_is_skipped_when_nothing_matched() -> None:
    """With no seed fact there is no entity to expand around, so no second query is issued."""
    graphiti = _TwoPassGraphiti([], neighbourhood=[])

    facts = asyncio.run(search_facts(graphiti, "q", "group-1", 30, get_episodes_by_uuids=_fake_episode_lookup([])))

    assert facts == []
    assert len(graphiti.calls) == 1
