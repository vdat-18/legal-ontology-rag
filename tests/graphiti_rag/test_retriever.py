"""Tests for searching and minimally selecting facts from the Graphiti graph."""

from __future__ import annotations

import asyncio

from legal_rag.graphiti_rag.retriever import RetrievedFact, search_facts, select_minimal_facts


class _FakeEdge:
    """A minimal stand-in for graphiti_core's EntityEdge."""

    def __init__(self, uuid: str, fact: str, name: str, source_node_uuid: str, target_node_uuid: str) -> None:
        self.uuid = uuid
        self.fact = fact
        self.name = name
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid


class _FakeGraphiti:
    """Record search() calls and return a fixed list of fake edges."""

    def __init__(self, edges: list[_FakeEdge]) -> None:
        self._edges = edges
        self.calls: list[dict[str, object]] = []

    async def search(self, query: str, **kwargs: object) -> list[_FakeEdge]:
        """Record the call and return the pre-seeded edges."""
        self.calls.append({"query": query, **kwargs})
        return self._edges


def _fact(chunk_id: str, source: str, target: str, text: str = "some fact") -> RetrievedFact:
    """Build one retrieved fact for selection tests."""
    return RetrievedFact(
        chunk_id=chunk_id, text=text, source_name="RELATION", source_node_uuid=source, target_node_uuid=target
    )


def test_search_facts_maps_edges_and_passes_through_group_and_limit() -> None:
    """search_facts forwards group_id/num_results to Graphiti's search and maps edges to facts."""
    edges = [_FakeEdge("e1", "Phạt tiền 4-6 triệu", "PENALIZES", "n1", "n2")]
    graphiti = _FakeGraphiti(edges)

    facts = asyncio.run(search_facts(graphiti, "vượt đèn đỏ bị phạt bao nhiêu?", "group-1", 30))

    assert facts == [RetrievedFact("e1", "Phạt tiền 4-6 triệu", "PENALIZES", "n1", "n2")]
    assert graphiti.calls == [{"query": "vượt đèn đỏ bị phạt bao nhiêu?", "group_ids": ["group-1"], "num_results": 30}]


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
