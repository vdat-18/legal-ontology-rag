"""Tests for searching facts and resolving their structured provenance from the Graphiti graph."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time
from types import SimpleNamespace

import numpy as np
import pytest
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag import embedder as embedder_module
from legal_rag.graphiti_rag import retriever as retriever_module
from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder
from legal_rag.graphiti_rag.retriever import (
    FactSource,
    MissingEffectiveProvenanceError,
    MissingProvenanceError,
    RetrievedFact,
    search_facts,
    search_facts_at_dates,
)


class _FakeEncoderBackend:
    """Return a deterministic vector that exposes the exact encoded text length."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return one fixed-size vector per input text."""
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _query_embedder() -> QwenGraphitiEmbedder:
    """Build the query-aware embedder required by production Graphiti retrieval."""
    profile = EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision="abc123",
        tokenizer_id="test/model",
        tokenizer_revision="abc123",
        dimensions=2,
        max_input_tokens=512,
        query_instruction="Instruct: ",
    )
    return QwenGraphitiEmbedder(EmbeddingEncoder(profile, _FakeEncoderBackend()))


@pytest.fixture(autouse=True)
def _use_fake_graphiti_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise this project's query-vector wiring without contacting Neo4j."""

    async def fake_validate_index(driver: object, group_id: str, _profile: EmbeddingProfile) -> None:
        driver.validated_group_ids.append(group_id)

    async def fake_search(
        clients: object,
        query: str,
        group_ids: list[str],
        config: object,
        _search_filter: object,
        center_node_uuid: str | None = None,
        query_vector: list[float] | None = None,
        driver: object | None = None,
    ) -> SimpleNamespace:
        clients.query_vectors.append(query_vector)
        clients.search_filters.append(_search_filter)
        kwargs: dict[str, object] = {
            "group_ids": group_ids,
            "num_results": config.limit,
        }
        if center_node_uuid is not None:
            kwargs["center_node_uuid"] = center_node_uuid
        edges = await clients.search(query, **kwargs)
        return SimpleNamespace(edges=edges)

    monkeypatch.setattr(retriever_module, "graphiti_search", fake_search)
    monkeypatch.setattr(embedder_module, "validate_graphiti_index_metadata", fake_validate_index)


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

    def __init__(
        self,
        uuid: str,
        name: str,
        source_description: str,
        content: str = "",
        valid_at: datetime = datetime(2020, 1, 1, tzinfo=UTC),
    ) -> None:
        self.uuid = uuid
        self.name = name
        self.source_description = source_description
        self.content = content
        self.valid_at = valid_at


class _FakeGraphiti:
    """Record search() calls and return a fixed list of fake edges."""

    def __init__(self, edges: list[_FakeEdge]) -> None:
        self._edges = edges
        self.calls: list[dict[str, object]] = []
        self.query_vectors: list[list[float] | None] = []
        self.search_filters: list[SearchFilters] = []
        self.driver = SimpleNamespace(validated_group_ids=[])
        self.clients = self
        self.embedder = _query_embedder()

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
    assert graphiti.query_vectors == [[40.0, 0.0]]
    assert graphiti.search_filters == [SearchFilters()]
    assert graphiti.driver.validated_group_ids == ["group-1"]


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
        self.query_vectors: list[list[float] | None] = []
        self.search_filters: list[SearchFilters] = []
        self.driver = SimpleNamespace(validated_group_ids=[])
        self.clients = self
        self.embedder = _query_embedder()

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
    assert graphiti.query_vectors == [[11.0, 0.0], [11.0, 0.0]]
    assert [call["query"] for call in graphiti.calls] == ["q", "q"]
    assert graphiti.driver.validated_group_ids == ["group-1"]


def test_as_of_date_filters_both_search_passes_at_end_of_utc_date() -> None:
    """Text and neighbourhood search use the same date-level effective-time boundary."""
    edge = _FakeEdge("e1", "vi phạm", "PENALIZES", episodes=["ep1"], source_node_uuid="violation-1")
    graphiti = _TwoPassGraphiti([edge], neighbourhood=[])
    as_of_date = date(2024, 12, 31)

    asyncio.run(
        search_facts(
            graphiti,
            "q",
            "group-1",
            30,
            as_of_date=as_of_date,
            get_episodes_by_uuids=_fake_episode_lookup([_FakeEpisode("ep1", "chunk-1", "doc.docx")]),
        )
    )

    cutoff = datetime.combine(as_of_date, time.max, tzinfo=UTC)
    expected = SearchFilters(
        valid_at=[[DateFilter(date=cutoff, comparison_operator=ComparisonOperator.less_than_equal)]],
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=cutoff, comparison_operator=ComparisonOperator.greater_than)],
        ],
    )
    assert graphiti.search_filters == [expected, expected]
    assert graphiti.search_filters[0] is graphiti.search_filters[1]


def test_as_of_date_excludes_future_reinforcement_from_fact_provenance() -> None:
    """A historical query cannot cite a later episode that reinforced the same edge."""
    edge = _FakeEdge("e1", "quy định", "REGULATES", episodes=["old", "future"])
    graphiti = _FakeGraphiti([edge])
    cutoff = date(2025, 12, 31)

    [fact] = asyncio.run(
        search_facts(
            graphiti,
            "q",
            "group-1",
            10,
            as_of_date=cutoff,
            expand_around_top_fact=False,
            get_episodes_by_uuids=_fake_episode_lookup(
                [
                    _FakeEpisode("old", "chunk-old", "old.docx", valid_at=datetime(2025, 1, 1, tzinfo=UTC)),
                    _FakeEpisode(
                        "future",
                        "chunk-future",
                        "future.docx",
                        valid_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                ]
            ),
        )
    )

    assert [source.chunk_id for source in fact.sources] == ["chunk-old"]


def test_as_of_date_rejects_fact_supported_only_by_future_episodes() -> None:
    """An effective edge without date-eligible source text fails rather than leaking future evidence."""
    edge = _FakeEdge("e1", "quy định", "REGULATES", episodes=["future"])
    graphiti = _FakeGraphiti([edge])

    with pytest.raises(MissingEffectiveProvenanceError, match="no supporting episode effective"):
        asyncio.run(
            search_facts(
                graphiti,
                "q",
                "group-1",
                10,
                as_of_date=date(2025, 12, 31),
                expand_around_top_fact=False,
                get_episodes_by_uuids=_fake_episode_lookup(
                    [
                        _FakeEpisode(
                            "future",
                            "chunk-future",
                            "future.docx",
                            valid_at=datetime(2026, 1, 1, tzinfo=UTC),
                        )
                    ]
                ),
            )
        )


def test_search_facts_at_dates_fuses_each_exact_target_and_keeps_date_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-version questions retrieve every target and label facts with matching dates."""
    first_date = date(2025, 1, 1)
    second_date = date(2026, 3, 1)
    calls: list[date | None] = []

    async def fake_search(*_args: object, as_of_date: date | None = None, **_kwargs: object) -> list[RetrievedFact]:
        calls.append(as_of_date)
        assert as_of_date is not None
        if as_of_date == first_date:
            return [
                RetrievedFact("shared", (FactSource("shared", "shared.docx"),), (as_of_date,)),
                RetrievedFact("same wording", (FactSource("old", "old.docx"),), (as_of_date,)),
            ]
        return [
            RetrievedFact("shared", (FactSource("shared", "shared.docx"),), (as_of_date,)),
            RetrievedFact("same wording", (FactSource("new", "new.docx"),), (as_of_date,)),
        ]

    monkeypatch.setattr(retriever_module, "search_facts", fake_search)

    facts = asyncio.run(
        search_facts_at_dates(object(), "question", "group", 3, as_of_dates=(first_date, second_date))  # type: ignore[arg-type]
    )

    assert calls == [first_date, second_date]
    assert [fact.fact for fact in facts] == ["same wording", "same wording", "shared"]
    assert facts[0].sources[0].chunk_id == "old"
    assert facts[0].as_of_dates == (first_date,)
    assert facts[1].sources[0].chunk_id == "new"
    assert facts[1].as_of_dates == (second_date,)
    assert facts[2].as_of_dates == (first_date, second_date)


def test_search_facts_at_dates_reserves_version_specific_top_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facts common to every snapshot cannot starve the change evidence from top-k."""
    first_date = date(2025, 1, 1)
    second_date = date(2026, 3, 1)

    async def fake_search(*_args: object, as_of_date: date | None = None, **_kwargs: object) -> list[RetrievedFact]:
        assert as_of_date is not None
        version = "old" if as_of_date == first_date else "new"
        specific = RetrievedFact(
            f"{version} version",
            (FactSource(version, f"{version}.docx"),),
            (as_of_date,),
        )
        common = [
            RetrievedFact(
                f"common {index}",
                (FactSource(f"common-{index}", "common.docx"),),
                (as_of_date,),
            )
            for index in range(10)
        ]
        return [specific, *common]

    monkeypatch.setattr(retriever_module, "search_facts", fake_search)

    facts = asyncio.run(
        search_facts_at_dates(
            object(),  # type: ignore[arg-type]
            "question",
            "group",
            10,
            as_of_dates=(first_date, second_date),
        )
    )

    assert len(facts) == 10
    assert [fact.fact for fact in facts[:2]] == ["old version", "new version"]


def test_cross_encoder_search_uses_the_instructed_query_vector_once() -> None:
    """The cross-encoder recipe receives the same explicit query-side vector contract."""
    edge = _FakeEdge("e1", "vi phạm", "PENALIZES", episodes=["ep1"])
    graphiti = _FakeGraphiti([edge])

    facts = asyncio.run(
        search_facts(
            graphiti,
            "q",
            "group-1",
            30,
            rerank_with_cross_encoder=True,
            get_episodes_by_uuids=_fake_episode_lookup([_FakeEpisode("ep1", "chunk-1", "doc.docx")]),
        )
    )

    assert [fact.fact for fact in facts] == ["vi phạm"]
    assert graphiti.query_vectors == [[11.0, 0.0]]
    assert graphiti.calls == [{"query": "q", "group_ids": ["group-1"], "num_results": 30}]


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


def test_graph_expansion_can_promote_an_adjacent_fact_into_a_full_top_k() -> None:
    """RRF lets graph proximity affect generation even when text search already filled top-k."""
    first = [_FakeEdge(f"e{index}", f"text fact {index}", "RELATED", episodes=[f"ep{index}"]) for index in range(1, 6)]
    neighbourhood = [
        _FakeEdge("adjacent", "adjacent remedy", "REQUIRES_REMEDY", episodes=["ep-adjacent"]),
        first[0],
    ]
    episodes = [
        *[_FakeEpisode(f"ep{index}", f"chunk-{index}", "doc.docx") for index in range(1, 6)],
        _FakeEpisode("ep-adjacent", "chunk-adjacent", "doc.docx"),
    ]
    graphiti = _TwoPassGraphiti(first, neighbourhood)

    facts = asyncio.run(search_facts(graphiti, "q", "group-1", 2, get_episodes_by_uuids=_fake_episode_lookup(episodes)))

    assert [fact.fact for fact in facts] == ["text fact 1", "adjacent remedy"]


def test_graph_expansion_is_skipped_when_nothing_matched() -> None:
    """With no seed fact there is no entity to expand around, so no second query is issued."""
    graphiti = _TwoPassGraphiti([], neighbourhood=[])

    facts = asyncio.run(search_facts(graphiti, "q", "group-1", 30, get_episodes_by_uuids=_fake_episode_lookup([])))

    assert facts == []
    assert len(graphiti.calls) == 1
