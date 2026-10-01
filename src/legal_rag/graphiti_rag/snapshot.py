"""Capture a deterministic snapshot of one Graphiti evaluation group in Neo4j."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from neo4j import AsyncDriver

from legal_rag.chunker.models import Chunk

_SNAPSHOT_SCHEMA_VERSION = "1.0.0"
_DEFAULT_PAGE_SIZE = 500

_EPISODE_QUERY = """
// graphiti_group_snapshot:episodes
MATCH (episode:Episodic {group_id: $group_id})
RETURN episode.uuid AS cursor,
       episode.name AS name,
       episode.source AS source,
       episode.source_description AS source_description,
       episode.content AS content,
       episode.valid_at AS valid_at
ORDER BY episode.uuid
SKIP $offset
LIMIT $page_size
"""

_ENTITY_QUERY = """
// graphiti_group_snapshot:entities
MATCH (entity:Entity {group_id: $group_id})
RETURN entity.uuid AS cursor,
       entity.name AS name,
       entity.summary AS summary,
       labels(entity) AS labels
ORDER BY entity.uuid
SKIP $offset
LIMIT $page_size
"""

_MENTION_QUERY = """
// graphiti_group_snapshot:mentions
MATCH (episode:Episodic)-[edge:MENTIONS {group_id: $group_id}]->(entity:Entity)
RETURN edge.uuid AS cursor,
       episode.group_id AS episode_group_id,
       episode.name AS episode_name,
       entity.group_id AS entity_group_id,
       entity.name AS entity_name,
       labels(entity) AS entity_labels
ORDER BY edge.uuid
SKIP $offset
LIMIT $page_size
"""

_FACT_QUERY = """
// graphiti_group_snapshot:facts
MATCH (source:Entity)-[edge:RELATES_TO {group_id: $group_id}]->(target:Entity)
RETURN edge.uuid AS cursor,
       source.group_id AS source_group_id,
       source.name AS source_name,
       labels(source) AS source_labels,
       target.group_id AS target_group_id,
       target.name AS target_name,
       labels(target) AS target_labels,
       edge.name AS name,
       edge.fact AS fact,
       edge.episodes AS episodes,
       edge.valid_at AS valid_at,
       edge.invalid_at AS invalid_at,
       edge.expired_at AS expired_at,
       edge.reference_time AS reference_time
ORDER BY edge.uuid
SKIP $offset
LIMIT $page_size
"""


class GraphitiGroupSnapshotError(ValueError):
    """Raised when a persisted Graphiti group cannot be fingerprinted safely."""


@dataclass(frozen=True, slots=True)
class GraphitiGroupSnapshot:
    """Stable counts and content fingerprint for one persisted Graphiti group.

    ``edge_count`` includes both provenance ``MENTIONS`` edges and fact-bearing
    ``RELATES_TO`` edges. ``fact_count`` is the latter subset.
    """

    schema_version: str
    group_id: str
    episode_count: int
    entity_count: int
    edge_count: int
    fact_count: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        """Return the JSON-compatible identity fields stored in an evaluation manifest."""
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "episode_count": self.episode_count,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "fact_count": self.fact_count,
            "sha256": self.sha256,
        }

    def validate_for_evaluation(self) -> None:
        """Reject an empty or incomplete group before an evaluation queries it."""
        missing = [
            label
            for label, count in (
                ("episodes", self.episode_count),
                ("entities", self.entity_count),
                ("facts", self.fact_count),
            )
            if count == 0
        ]
        if missing:
            joined = ", ".join(missing)
            msg = f"Graphiti group {self.group_id!r} is not evaluable because it has no {joined}."
            raise GraphitiGroupSnapshotError(msg)


async def snapshot_graphiti_group(
    neo4j_driver: AsyncDriver,
    group_id: str,
    *,
    page_size: int = _DEFAULT_PAGE_SIZE,
    expected_chunks: Sequence[Chunk] | None = None,
    expected_source_dates: Mapping[str, date] | None = None,
) -> GraphitiGroupSnapshot:
    """Read and fingerprint the exact Graphiti group selected for evaluation.

    Database responses are bounded by ``page_size``. The digest excludes random UUIDs,
    ingestion timestamps, and embeddings; it includes the persisted text, temporal values,
    entity labels/summaries, fact endpoints, and fact provenance. As a result, re-indexing
    the same logical graph has the same digest while any query-relevant content change does not.

    This function validates group boundaries and fact-to-episode references while reading.
    Call :meth:`GraphitiGroupSnapshot.validate_for_evaluation` as well when an empty group
    should fail rather than be recorded.
    """
    if not group_id.strip():
        msg = "Graphiti group_id must not be blank."
        raise ValueError(msg)
    if page_size <= 0:
        msg = "Graphiti snapshot page_size must be positive."
        raise ValueError(msg)

    episode_records = await _read_all_pages(neo4j_driver, _EPISODE_QUERY, group_id, page_size)
    if expected_chunks is not None:
        _validate_episode_corpus(
            episode_records,
            expected_chunks,
            group_id,
            expected_source_dates=expected_source_dates or {},
        )
    episode_names_by_uuid = _episode_names_by_uuid(episode_records)
    entity_records = await _read_all_pages(neo4j_driver, _ENTITY_QUERY, group_id, page_size)
    mention_records = await _read_all_pages(neo4j_driver, _MENTION_QUERY, group_id, page_size)
    fact_records = await _read_all_pages(neo4j_driver, _FACT_QUERY, group_id, page_size)

    _validate_edge_groups(mention_records, group_id, "episode_group_id", "entity_group_id")
    _validate_edge_groups(fact_records, group_id, "source_group_id", "target_group_id")

    episode_hashes = _record_hashes(episode_records, excluded_fields=frozenset({"cursor"}))
    entity_hashes = _record_hashes(entity_records, excluded_fields=frozenset({"cursor"}))
    mention_hashes = _record_hashes(
        mention_records,
        excluded_fields=frozenset({"cursor", "episode_group_id", "entity_group_id"}),
    )
    fact_hashes = _fact_hashes(fact_records, episode_names_by_uuid, group_id)

    fingerprint_payload = {
        "schema_version": _SNAPSHOT_SCHEMA_VERSION,
        "group_id": group_id,
        "episodes": episode_hashes,
        "entities": entity_hashes,
        "mentions": mention_hashes,
        "facts": fact_hashes,
    }
    digest = hashlib.sha256(_canonical_json(fingerprint_payload)).hexdigest()
    return GraphitiGroupSnapshot(
        schema_version=_SNAPSHOT_SCHEMA_VERSION,
        group_id=group_id,
        episode_count=len(episode_records),
        entity_count=len(entity_records),
        edge_count=len(mention_records) + len(fact_records),
        fact_count=len(fact_records),
        sha256=digest,
    )


def _validate_episode_corpus(
    episode_records: Sequence[Mapping[str, Any]],
    expected_chunks: Sequence[Chunk],
    group_id: str,
    *,
    expected_source_dates: Mapping[str, date],
) -> None:
    """Require the active episode set to match the selected local chunks exactly."""
    expected: dict[str, Chunk] = {}
    for chunk in expected_chunks:
        if chunk.chunk_id in expected:
            msg = f"Local Graphiti corpus contains duplicate chunk_id {chunk.chunk_id!r}."
            raise GraphitiGroupSnapshotError(msg)
        expected[chunk.chunk_id] = chunk

    active: dict[str, Mapping[str, Any]] = {}
    for record in episode_records:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            msg = f"Graphiti group {group_id!r} contains an episode without a stable chunk name."
            raise GraphitiGroupSnapshotError(msg)
        if name in active:
            msg = f"Graphiti group {group_id!r} contains duplicate episode name {name!r}."
            raise GraphitiGroupSnapshotError(msg)
        active[name] = record

    missing = sorted(set(expected) - set(active))
    unexpected = sorted(set(active) - set(expected))
    if missing or unexpected:
        msg = (
            f"Graphiti group {group_id!r} does not match the selected local corpus: "
            f"{len(missing)} missing and {len(unexpected)} unexpected episode(s)."
        )
        raise GraphitiGroupSnapshotError(msg)

    for chunk_id, chunk in expected.items():
        record = active[chunk_id]
        if record.get("content") != chunk.text or record.get("source_description") != chunk.source_name:
            msg = f"Graphiti episode {chunk_id!r} is stale; rebuild group {group_id!r} before evaluation."
            raise GraphitiGroupSnapshotError(msg)
        expected_date = expected_source_dates.get(chunk.source_name)
        if expected_date is not None and _calendar_date(record.get("valid_at")) != expected_date:
            msg = (
                f"Graphiti episode {chunk_id!r} has stale effective-date metadata; "
                f"rebuild group {group_id!r} before evaluation."
            )
            raise GraphitiGroupSnapshotError(msg)


def _calendar_date(value: Any) -> date | None:
    """Normalize built-in and Neo4j temporal values to one calendar date."""
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        value = to_native()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


async def _read_all_pages(
    neo4j_driver: AsyncDriver,
    query: str,
    group_id: str,
    page_size: int,
) -> list[Mapping[str, Any]]:
    """Read one record type with bounded Neo4j responses."""
    all_records: list[Mapping[str, Any]] = []
    offset = 0
    while True:
        records, _, _ = await neo4j_driver.execute_query(
            query,
            group_id=group_id,
            offset=offset,
            page_size=page_size,
            routing_="r",
        )
        page = list(records)
        all_records.extend(page)
        if len(page) < page_size:
            return all_records
        offset += len(page)


def _episode_names_by_uuid(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map persisted episode UUIDs to their stable chunk names."""
    names: dict[str, str] = {}
    for record in records:
        uuid = record.get("cursor")
        name = record.get("name")
        if not isinstance(uuid, str) or not uuid:
            msg = "Graphiti snapshot found an episode without a UUID."
            raise GraphitiGroupSnapshotError(msg)
        if not isinstance(name, str) or not name:
            msg = f"Graphiti episode {uuid!r} has no stable chunk name."
            raise GraphitiGroupSnapshotError(msg)
        names[uuid] = name
    return names


def _validate_edge_groups(
    records: Sequence[Mapping[str, Any]],
    group_id: str,
    source_group_field: str,
    target_group_field: str,
) -> None:
    """Reject edges whose endpoint nodes cross the requested group boundary."""
    for record in records:
        if record.get(source_group_field) != group_id or record.get(target_group_field) != group_id:
            edge_uuid = record.get("cursor")
            msg = f"Graphiti edge {edge_uuid!r} in group {group_id!r} connects an endpoint from another group."
            raise GraphitiGroupSnapshotError(msg)


def _fact_hashes(
    records: Sequence[Mapping[str, Any]],
    episode_names_by_uuid: Mapping[str, str],
    group_id: str,
) -> list[str]:
    """Hash fact rows after replacing random provenance UUIDs with stable chunk ids."""
    hashes: list[str] = []
    for record in records:
        edge_uuid = record.get("cursor")
        fact = record.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            msg = f"Graphiti fact edge {edge_uuid!r} in group {group_id!r} has no fact text."
            raise GraphitiGroupSnapshotError(msg)
        if _calendar_date(record.get("valid_at")) is None:
            msg = (
                f"Graphiti fact edge {edge_uuid!r} in group {group_id!r} has no effective valid_at; "
                "rerun indexing to normalize temporal metadata before evaluation."
            )
            raise GraphitiGroupSnapshotError(msg)

        episode_uuids = record.get("episodes")
        if not isinstance(episode_uuids, Sequence) or isinstance(episode_uuids, str):
            msg = f"Graphiti fact edge {edge_uuid!r} in group {group_id!r} has invalid episode provenance."
            raise GraphitiGroupSnapshotError(msg)
        if not episode_uuids:
            msg = f"Graphiti fact edge {edge_uuid!r} in group {group_id!r} has no episode provenance."
            raise GraphitiGroupSnapshotError(msg)
        missing = sorted(
            str(uuid) for uuid in episode_uuids if not isinstance(uuid, str) or uuid not in episode_names_by_uuid
        )
        if missing:
            msg = (
                f"Graphiti fact edge {edge_uuid!r} in group {group_id!r} references episodes outside the group: "
                f"{', '.join(missing)}"
            )
            raise GraphitiGroupSnapshotError(msg)

        stable_record = {
            key: value
            for key, value in record.items()
            if key not in {"cursor", "source_group_id", "target_group_id", "episodes"}
        }
        stable_record["episode_names"] = sorted(episode_names_by_uuid[uuid] for uuid in episode_uuids)
        hashes.append(hashlib.sha256(_canonical_json(stable_record)).hexdigest())
    return sorted(hashes)


def _record_hashes(records: Sequence[Mapping[str, Any]], *, excluded_fields: frozenset[str]) -> list[str]:
    """Return sorted per-record hashes so database row order cannot affect the snapshot."""
    hashes = []
    for record in records:
        stable_record = {key: value for key, value in record.items() if key not in excluded_fields}
        hashes.append(hashlib.sha256(_canonical_json(stable_record)).hexdigest())
    return sorted(hashes)


def _canonical_json(value: object) -> bytes:
    """Serialize persisted Neo4j values to deterministic UTF-8 JSON."""
    normalized = _normalize_value(value)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _normalize_value(value: object) -> object:
    """Normalize Neo4j temporal values and unordered containers for hashing."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return sorted((_normalize_value(item) for item in value), key=_sort_key)
    msg = f"Unsupported Neo4j value in Graphiti snapshot: {type(value).__name__}"
    raise GraphitiGroupSnapshotError(msg)


def _sort_key(value: object) -> str:
    """Build a stable ordering key for normalized sequence elements."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
