"""Tests for deterministic snapshots of persisted Graphiti evaluation groups."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from datetime import date

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.graphiti_rag.snapshot import (
    GraphitiGroupSnapshotError,
    snapshot_graphiti_group,
)


def _chunk(
    *,
    chunk_id: str = "chunk-stable-1",
    text: str = "Phạt tiền từ 4 đến 6 triệu đồng.",
) -> Chunk:
    """Build the local chunk represented by the complete fake graph group."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content",
        chunker_config_sha256="config",
        scope_id="scope",
        chunk_index=0,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-1",),
        source_refs=("ref-1",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text=text,
        embedding_text=text,
        text_token_count=8,
        embedding_token_count=8,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


class _FakeNeo4jDriver:
    """Return bounded pages from in-memory records keyed by snapshot query marker."""

    def __init__(self, records: Mapping[str, list[dict[str, object]]]) -> None:
        self.records = {key: deepcopy(value) for key, value in records.items()}
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, query: str, **params: object) -> tuple[list[dict[str, object]], None, None]:
        """Return the requested offset/limit page for the query's record kind."""
        self.calls.append((query, params))
        kind = next(key for key in self.records if f"graphiti_group_snapshot:{key}" in query)
        offset = int(params["offset"])
        page_size = int(params["page_size"])
        return self.records[kind][offset : offset + page_size], None, None


def _group_records(*, episode_uuid: str = "episode-random-1") -> dict[str, list[dict[str, object]]]:
    """Build one complete group snapshot with random persisted identifiers."""
    return {
        "episodes": [
            {
                "cursor": episode_uuid,
                "name": "chunk-stable-1",
                "source": "text",
                "source_description": "168_2024_ND-CP.docx",
                "content": "Phạt tiền từ 4 đến 6 triệu đồng.",
                "valid_at": "2025-01-01T00:00:00+00:00",
            }
        ],
        "entities": [
            {
                "cursor": "entity-random-1",
                "name": "vượt đèn đỏ",
                "summary": "Một hành vi vi phạm giao thông.",
                "labels": ["Violation", "Entity"],
            },
            {
                "cursor": "entity-random-2",
                "name": "4 đến 6 triệu đồng",
                "summary": "Mức phạt tiền.",
                "labels": ["Penalty", "Entity"],
            },
        ],
        "mentions": [
            {
                "cursor": "mention-random-1",
                "episode_group_id": "g1",
                "episode_name": "chunk-stable-1",
                "entity_group_id": "g1",
                "entity_name": "vượt đèn đỏ",
                "entity_labels": ["Violation", "Entity"],
            },
            {
                "cursor": "mention-random-2",
                "episode_group_id": "g1",
                "episode_name": "chunk-stable-1",
                "entity_group_id": "g1",
                "entity_name": "4 đến 6 triệu đồng",
                "entity_labels": ["Penalty", "Entity"],
            },
        ],
        "facts": [
            {
                "cursor": "fact-random-1",
                "source_group_id": "g1",
                "source_name": "vượt đèn đỏ",
                "source_labels": ["Violation", "Entity"],
                "target_group_id": "g1",
                "target_name": "4 đến 6 triệu đồng",
                "target_labels": ["Penalty", "Entity"],
                "name": "PENALIZES",
                "fact": "Vượt đèn đỏ bị phạt từ 4 đến 6 triệu đồng.",
                "episodes": [episode_uuid],
                "valid_at": "2025-01-01T00:00:00+00:00",
                "invalid_at": None,
                "expired_at": None,
                "reference_time": "2025-01-01T00:00:00+00:00",
            }
        ],
    }


def test_snapshot_records_counts_and_uses_bounded_parameterized_queries() -> None:
    """Every Graphiti record class is counted and read through bounded parameters."""
    driver = _FakeNeo4jDriver(_group_records())

    snapshot = asyncio.run(snapshot_graphiti_group(driver, "g1", page_size=1))

    assert snapshot.to_dict() == {
        "schema_version": "1.0.0",
        "group_id": "g1",
        "episode_count": 1,
        "entity_count": 2,
        "edge_count": 3,
        "fact_count": 1,
        "sha256": snapshot.sha256,
    }
    assert len(snapshot.sha256) == 64
    snapshot.validate_for_evaluation()
    assert len(driver.calls) == 10
    for query, params in driver.calls:
        assert "LIMIT $page_size" in query
        assert "g1" not in query
        assert params["group_id"] == "g1"
        assert params["page_size"] == 1
        assert params["routing_"] == "r"


def test_snapshot_digest_ignores_random_ids_database_order_and_page_size() -> None:
    """A logical re-index has one fingerprint even when Graphiti assigns new UUIDs."""
    first_records = _group_records(episode_uuid="episode-first")
    second_records = _group_records(episode_uuid="episode-second")
    for records in second_records.values():
        records.reverse()
    second_records["entities"][0]["cursor"] = "new-entity-2"
    second_records["entities"][1]["cursor"] = "new-entity-1"
    second_records["mentions"][0]["cursor"] = "new-mention-2"
    second_records["mentions"][1]["cursor"] = "new-mention-1"
    second_records["facts"][0]["cursor"] = "new-fact"

    first = asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(first_records), "g1", page_size=1))
    second = asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(second_records), "g1", page_size=100))

    assert first.sha256 == second.sha256


def test_snapshot_digest_changes_with_query_relevant_content() -> None:
    """Changing persisted episode text changes the group identity."""
    original_records = _group_records()
    changed_records = _group_records()
    changed_records["episodes"][0]["content"] = "Nội dung đã thay đổi."

    original = asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(original_records), "g1"))
    changed = asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(changed_records), "g1"))

    assert original.sha256 != changed.sha256


def test_snapshot_validates_exact_local_episode_corpus() -> None:
    """Evaluation preflight rejects missing, unexpected, or stale active episodes."""
    snapshot = asyncio.run(
        snapshot_graphiti_group(_FakeNeo4jDriver(_group_records()), "g1", expected_chunks=[_chunk()])
    )
    assert snapshot.episode_count == 1

    with pytest.raises(GraphitiGroupSnapshotError, match="1 missing and 1 unexpected"):
        asyncio.run(
            snapshot_graphiti_group(
                _FakeNeo4jDriver(_group_records()),
                "g1",
                expected_chunks=[_chunk(chunk_id="different-chunk")],
            )
        )

    with pytest.raises(GraphitiGroupSnapshotError, match="is stale"):
        asyncio.run(
            snapshot_graphiti_group(
                _FakeNeo4jDriver(_group_records()),
                "g1",
                expected_chunks=[_chunk(text="stale local text")],
            )
        )


def test_snapshot_validates_authoritative_episode_effective_dates() -> None:
    """An episode indexed under the wrong legal effective date fails preflight."""
    snapshot = asyncio.run(
        snapshot_graphiti_group(
            _FakeNeo4jDriver(_group_records()),
            "g1",
            expected_chunks=[_chunk()],
            expected_source_dates={"168_2024_ND-CP.docx": date(2025, 1, 1)},
        )
    )

    assert snapshot.episode_count == 1
    with pytest.raises(GraphitiGroupSnapshotError, match="stale effective-date metadata"):
        asyncio.run(
            snapshot_graphiti_group(
                _FakeNeo4jDriver(_group_records()),
                "g1",
                expected_chunks=[_chunk()],
                expected_source_dates={"168_2024_ND-CP.docx": date(2026, 1, 1)},
            )
        )


def test_snapshot_rejects_fact_provenance_outside_the_group() -> None:
    """A fact cannot silently fingerprint an episode from another Graphiti group."""
    records = _group_records()
    records["facts"][0]["episodes"] = ["episode-from-another-group"]

    with pytest.raises(GraphitiGroupSnapshotError, match="references episodes outside the group"):
        asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(records), "g1"))


def test_snapshot_rejects_fact_without_episode_provenance() -> None:
    """A fact with no supporting episode cannot satisfy the retriever's provenance contract."""
    records = _group_records()
    records["facts"][0]["episodes"] = []

    with pytest.raises(GraphitiGroupSnapshotError, match="has no episode provenance"):
        asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(records), "g1"))


def test_snapshot_rejects_cross_group_edge_endpoint() -> None:
    """An edge whose endpoint escaped the partition is not a valid group snapshot."""
    records = _group_records()
    records["mentions"][0]["entity_group_id"] = "other-group"

    with pytest.raises(GraphitiGroupSnapshotError, match="connects an endpoint from another group"):
        asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(records), "g1"))


def test_snapshot_rejects_blank_fact_text() -> None:
    """A RELATES_TO edge without a retrievable fact cannot identify an evaluation graph."""
    records = _group_records()
    records["facts"][0]["fact"] = " "

    with pytest.raises(GraphitiGroupSnapshotError, match="has no fact text"):
        asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(records), "g1"))


def test_snapshot_rejects_fact_not_normalized_from_its_reference_time() -> None:
    """A dated reference_time cannot mask a null valid_at that as-of search would exclude."""
    records = _group_records()
    records["facts"][0]["valid_at"] = None

    with pytest.raises(GraphitiGroupSnapshotError, match="no effective valid_at"):
        asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(records), "g1"))


def test_validate_for_evaluation_rejects_an_empty_group() -> None:
    """Empty groups can be recorded diagnostically but cannot be evaluated."""
    empty = {"episodes": [], "entities": [], "mentions": [], "facts": []}

    snapshot = asyncio.run(snapshot_graphiti_group(_FakeNeo4jDriver(empty), "empty"))

    with pytest.raises(GraphitiGroupSnapshotError, match="no episodes, entities, facts"):
        snapshot.validate_for_evaluation()


@pytest.mark.parametrize(
    ("group_id", "page_size", "error"),
    [("", 1, "group_id must not be blank"), ("g1", 0, "page_size must be positive")],
)
def test_snapshot_rejects_invalid_request_bounds(group_id: str, page_size: int, error: str) -> None:
    """Invalid caller input fails before any Neo4j query is attempted."""
    driver = _FakeNeo4jDriver(_group_records())

    with pytest.raises(ValueError, match=error):
        asyncio.run(snapshot_graphiti_group(driver, group_id, page_size=page_size))

    assert driver.calls == []
