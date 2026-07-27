"""Tests for searching and minimally selecting facts from the Graphiti graph."""

from __future__ import annotations

import asyncio

from legal_rag.graphiti_rag.retriever import RetrievedFact, search_facts, select_minimal_facts


class _FakeEdge:
    """A minimal stand-in for graphiti_core's EntityEdge."""

    def __init__(
        self, uuid: str, fact: str, name: str, source_node_uuid: str, target_node_uuid: str, episodes: list[str]
    ) -> None:
        self.uuid = uuid
        self.fact = fact
        self.name = name
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.episodes = episodes


class _FakeEpisode:
    """A minimal stand-in for graphiti_core's EpisodicNode."""

    def __init__(self, uuid: str, name: str, source_description: str) -> None:
        self.uuid = uuid
        self.name = name
        self.source_description = source_description


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
    """Build a fake ``get_episodes_by_uuids`` that resolves from a pre-seeded episode list."""
    by_uuid = {episode.uuid: episode for episode in episodes}

    async def _lookup(_driver: object, uuids: list[str]) -> list[_FakeEpisode]:
        return [by_uuid[uuid] for uuid in uuids]

    return _lookup


def _fact(chunk_id: str, source: str, target: str, text: str = "some fact") -> RetrievedFact:
    """Build one retrieved fact for selection tests."""
    return RetrievedFact(
        chunk_id=chunk_id, text=text, source_name="RELATION", source_node_uuid=source, target_node_uuid=target
    )


def test_search_facts_cites_the_source_episode_not_the_edges_own_name_and_uuid() -> None:
    """Provenance comes from the edge's supporting episode, not its own relation name/uuid."""
    edges = [_FakeEdge("e1", "Phạt tiền 4-6 triệu", "PENALIZES", "n1", "n2", episodes=["ep1"])]
    episodes = [_FakeEpisode("ep1", "168_2024_ND-CP.docx#3", "168_2024_ND-CP.docx")]
    graphiti = _FakeGraphiti(edges)

    facts = asyncio.run(
        search_facts(
            graphiti,
            "vượt đèn đỏ bị phạt bao nhiêu?",
            "group-1",
            30,
            get_episodes_by_uuids=_fake_episode_lookup(episodes),
        )
    )

    assert facts == [RetrievedFact("168_2024_ND-CP.docx#3", "Phạt tiền 4-6 triệu", "168_2024_ND-CP.docx", "n1", "n2")]
    assert graphiti.calls == [{"query": "vượt đèn đỏ bị phạt bao nhiêu?", "group_ids": ["group-1"], "num_results": 30}]


def test_search_facts_uses_the_first_episode_when_an_edge_has_several() -> None:
    """An edge reinforced across multiple episodes is cited by its first supporting episode."""
    edges = [_FakeEdge("e1", "some fact", "PENALIZES", "n1", "n2", episodes=["ep1", "ep2"])]
    episodes = [
        _FakeEpisode("ep1", "doc.docx#1", "doc.docx"),
        _FakeEpisode("ep2", "doc.docx#9", "doc.docx"),
    ]
    graphiti = _FakeGraphiti(edges)

    facts = asyncio.run(
        search_facts(graphiti, "q", "group-1", 30, get_episodes_by_uuids=_fake_episode_lookup(episodes))
    )

    assert facts[0].chunk_id == "doc.docx#1"


def test_select_minimal_facts_keeps_the_first_fact_even_with_no_new_coverage() -> None:
    """The top-ranked fact is always kept, even though it "covers" only itself."""
    facts = [_fact("f1", "a", "b")]

    selected = select_minimal_facts(facts, max_facts=10)

    assert selected == facts


def test_select_minimal_facts_drops_fully_redundant_facts() -> None:
    """A lower-ranked fact whose entities are already covered is dropped as redundant."""
    facts = [_fact("f1", "a", "b"), _fact("f2", "a", "b")]

    selected = select_minimal_facts(facts, max_facts=10)

    assert selected == [facts[0]]


def test_select_minimal_facts_keeps_facts_that_introduce_new_entities() -> None:
    """A fact that connects to at least one new entity is kept, even if it shares one entity."""
    facts = [_fact("f1", "a", "b"), _fact("f2", "b", "c")]

    selected = select_minimal_facts(facts, max_facts=10)

    assert selected == facts


def test_select_minimal_facts_respects_max_facts_budget() -> None:
    """Selection stops once max_facts is reached, even if more non-redundant facts remain."""
    facts = [_fact("f1", "a", "b"), _fact("f2", "c", "d"), _fact("f3", "e", "f")]

    selected = select_minimal_facts(facts, max_facts=2)

    assert selected == facts[:2]
