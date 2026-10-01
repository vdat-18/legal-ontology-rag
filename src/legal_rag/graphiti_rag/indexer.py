"""Load chunked JSONL data and corpus metadata, then index episodes into Graphiti."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from loguru import logger
from neo4j import AsyncDriver

from legal_rag.benchmark_synthesis.writer import read_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.graphiti_rag.effective_date import EffectiveDateClient, extract_effective_date
from legal_rag.graphiti_rag.ontology import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """Temporal and identifying metadata for one legal document in a ground-truth corpus."""

    document_number: str
    effective_date: date | None
    amends: str | None


@dataclass(frozen=True, slots=True)
class CorpusMetadata:
    """One corpus's shared Graphiti group id and per-document temporal metadata."""

    group_id: str
    documents: dict[str, CorpusDocument]


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    """Counts of episodes indexed and skipped for one indexing run."""

    indexed: int
    skipped_no_metadata: int
    skipped_no_date: int
    skipped_already_indexed: int


def load_corpus_metadata(corpus_config_path: Path) -> CorpusMetadata:
    """Read a ground-truth corpus config into its shared group id and per-document metadata."""
    config = json.loads(corpus_config_path.read_text(encoding="utf-8"))
    documents: dict[str, CorpusDocument] = {}
    for document in config["documents"]:
        source_name = document["file"]
        effective_date_value = document.get("effective_date")
        authoritative_number, authoritative_valid_from = _load_document_metadata(corpus_config_path, source_name)
        documents[source_name] = CorpusDocument(
            document_number=authoritative_number
            or document.get("so_hieu")
            or _document_number_from_filename(source_name),
            effective_date=(
                date.fromisoformat(effective_date_value) if effective_date_value else authoritative_valid_from
            ),
            amends=document.get("amends"),
        )
    return CorpusMetadata(group_id=config["group_id"], documents=documents)


def _load_document_metadata(corpus_config_path: Path, source_name: str) -> tuple[str | None, date | None]:
    """Load authoritative document number and validity from prepared metadata when available.

    Ground-truth corpus configs select source files but do not always duplicate their legal
    document numbers or effective dates. Preparation writes those authoritative values beside
    processed chunks.
    Support both a config-local artifact and the repository's ``data/processed/groundtruth``
    layout so copied/self-contained configs continue to work.
    """
    metadata_name = f"{Path(source_name).stem}.document_metadata.jsonl"
    candidate_paths = (
        corpus_config_path.parent / metadata_name,
        corpus_config_path.parent.parent / "processed" / "groundtruth" / metadata_name,
    )
    existing_paths = tuple(dict.fromkeys(path for path in candidate_paths if path.is_file()))
    if not existing_paths:
        return None, None

    records = tuple(read_document_metadata(path) for path in existing_paths)
    document_numbers = {record.document_number.strip() for record in records}
    if "" in document_numbers:
        msg = f"Document metadata for {source_name!r} has a blank document_number."
        raise ValueError(msg)
    valid_from_values = {record.valid_from for record in records}
    if len(document_numbers) != 1 or len(valid_from_values) != 1:
        rendered = ", ".join(sorted(f"{record.document_number.strip()}@{record.valid_from}" for record in records))
        msg = f"Conflicting document metadata for {source_name!r}: {rendered}"
        raise ValueError(msg)
    return document_numbers.pop(), valid_from_values.pop()


def _document_number_from_filename(source_name: str) -> str:
    """Recover the conventional legal number from repository document filenames."""
    parts = Path(source_name).stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    if len(parts) > 1 and parts[-1] == "m":
        parts.pop()
    return "/".join(parts)


async def _episode_already_indexed(neo4j_driver: AsyncDriver, chunk: Chunk, group_id: str) -> bool:
    """Return whether a current Episodic node for *chunk* already exists in *group_id*.

    Graphiti has no upsert-by-uuid for new episodes: passing ``uuid=`` to ``add_episode``
    means "re-fetch this existing episode", and raises ``NodeNotFoundError`` if it is not
    already present. So the idempotency check has to key on something set before the episode
    exists — the chunk's stable ``chunk_id`` this indexer uses as the episode name, scoped to
    the corpus group. A stable id alone is insufficient when the chunk corpus has been rebuilt:
    accepting stale content or provenance would silently mix corpus versions in one graph.
    """
    records, _, _ = await neo4j_driver.execute_query(
        """
        MATCH (e:Episodic {name: $name, group_id: $group_id})
        RETURN e.uuid AS uuid,
               e.content = $content AS content_matches,
               e.source_description = $source_description AS source_description_matches
        LIMIT 1
        """,
        name=chunk.chunk_id,
        group_id=group_id,
        content=chunk.text,
        source_description=chunk.source_name,
    )
    if not records:
        return False

    record = records[0]
    mismatched_fields = [
        field
        for field, matches in (
            ("content", record["content_matches"]),
            ("source_description", record["source_description_matches"]),
        )
        if matches is not True
    ]
    if mismatched_fields:
        fields = ", ".join(mismatched_fields)
        msg = (
            f"Graphiti episode {chunk.chunk_id!r} in group {group_id!r} does not match the current chunk "
            f"({fields}). Rebuild the Graphiti index for this group before indexing the updated corpus."
        )
        raise ValueError(msg)
    return True


async def _normalize_fact_effective_dates(neo4j_driver: AsyncDriver, chunk: Chunk, group_id: str) -> None:
    """Persist the episode date on facts whose extraction did not identify a narrower date.

    Graphiti records ``reference_time`` for every extracted fact, while ``valid_at`` remains
    optional when the fact text contains no explicit timestamp. Retrieval filters on
    ``valid_at``, so leaving it null would silently hide ordinary legal facts from every as-of
    query. The supporting episode's legal effective date is the final fallback.
    """
    await neo4j_driver.execute_query(
        """
        MATCH (episode:Episodic {name: $name, group_id: $group_id})
        MATCH ()-[edge:RELATES_TO {group_id: $group_id}]->()
        WHERE episode.uuid IN edge.episodes AND edge.valid_at IS NULL
        SET edge.valid_at = coalesce(edge.reference_time, episode.valid_at)
        """,
        name=chunk.chunk_id,
        group_id=group_id,
    )


def _resolve_effective_dates(
    chunks: Sequence[Chunk],
    corpus_metadata: CorpusMetadata,
    date_extraction_client: EffectiveDateClient,
    window_token_budget: int,
) -> dict[str, date | None]:
    """Resolve one effective date per source document: the corpus config's, else extracted.

    The corpus config's ``effective_date`` is the authoritative source when present. When it is
    ``null``, fall back to extracting the document's own stated commencement date from its
    "Hiệu lực thi hành" article (see ``effective_date.extract_effective_date``) rather than
    skipping the document outright — most legal documents state this themselves.
    """
    chunks_by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source_name, []).append(chunk)

    resolved: dict[str, date | None] = {}
    for source_name, document_chunks in chunks_by_source.items():
        document = corpus_metadata.documents.get(source_name)
        if document is None:
            continue
        if document.effective_date is not None:
            resolved[source_name] = document.effective_date
            continue
        extracted = extract_effective_date(document_chunks, date_extraction_client, window_token_budget)
        if extracted is not None:
            logger.info("Extracted effective_date {} for {} from its own text", extracted, source_name)
        resolved[source_name] = extracted
    return resolved


async def _add_episode(graphiti_client: Graphiti, chunk: Chunk, reference_time: datetime, group_id: str) -> None:
    """Add one chunk to the graph as a single Graphiti episode named by its stable ``chunk_id``.

    Errors from ``add_episode`` (LLM, Neo4j, or Graphiti internals) propagate to the caller rather
    than being swallowed: a failed episode is a real failure that must surface, not silently
    counted as "skipped".
    """
    await graphiti_client.add_episode(
        name=chunk.chunk_id,
        episode_body=chunk.text,
        source_description=chunk.source_name,
        reference_time=reference_time,
        source=EpisodeType.text,
        group_id=group_id,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
    )


def _filter_eligible_chunks(
    chunks: Sequence[Chunk],
    corpus_metadata: CorpusMetadata,
    effective_dates: dict[str, date | None],
) -> tuple[list[Chunk], int, int]:
    """Split *chunks* into those eligible for indexing and counts of the two skip reasons."""
    eligible: list[Chunk] = []
    skipped_no_metadata = 0
    skipped_no_date = 0

    for chunk in chunks:
        if corpus_metadata.documents.get(chunk.source_name) is None:
            logger.warning("Skipping chunk {}: {} is not in the corpus config", chunk.chunk_id, chunk.source_name)
            skipped_no_metadata += 1
            continue
        if effective_dates.get(chunk.source_name) is None:
            logger.warning("Skipping chunk {}: {} has no known effective_date", chunk.chunk_id, chunk.source_name)
            skipped_no_date += 1
            continue
        eligible.append(chunk)

    return eligible, skipped_no_metadata, skipped_no_date


async def index_chunks(
    chunks: Sequence[Chunk],
    corpus_metadata: CorpusMetadata,
    graphiti_client: Graphiti,
    neo4j_driver: AsyncDriver,
    date_extraction_client: EffectiveDateClient,
    window_token_budget: int,
) -> IndexingSummary:
    """Add one Graphiti episode per chunk, skipping chunks without known temporal metadata."""
    effective_dates = await asyncio.to_thread(
        _resolve_effective_dates, chunks, corpus_metadata, date_extraction_client, window_token_budget
    )
    eligible, skipped_no_metadata, skipped_no_date = _filter_eligible_chunks(chunks, corpus_metadata, effective_dates)

    indexed = 0
    skipped_already_indexed = 0

    for chunk in eligible:
        if await _episode_already_indexed(neo4j_driver, chunk, corpus_metadata.group_id):
            await _normalize_fact_effective_dates(neo4j_driver, chunk, corpus_metadata.group_id)
            skipped_already_indexed += 1
            continue

        reference_time = datetime.combine(effective_dates[chunk.source_name], time.min, tzinfo=UTC)
        await _add_episode(graphiti_client, chunk, reference_time, corpus_metadata.group_id)
        await _normalize_fact_effective_dates(neo4j_driver, chunk, corpus_metadata.group_id)
        indexed += 1

    return IndexingSummary(
        indexed=indexed,
        skipped_no_metadata=skipped_no_metadata,
        skipped_no_date=skipped_no_date,
        skipped_already_indexed=skipped_already_indexed,
    )


async def index_chunks_concurrent(
    chunks: Sequence[Chunk],
    corpus_metadata: CorpusMetadata,
    graphiti_clients: Sequence[Graphiti],
    neo4j_driver: AsyncDriver,
    date_extraction_client: EffectiveDateClient,
    window_token_budget: int,
) -> IndexingSummary:
    """Index chunks using several Graphiti clients (e.g. one per LLM API key) sharing one backlog.

    Unlike statically splitting chunks into one fixed batch per client up front, every client
    takes its next chunk from a shared backlog the moment it finishes the previous one — so a
    client whose batch happened to be a short document doesn't sit idle while others still have
    a long backlog. This is the concurrency-safe counterpart to :func:`index_chunks`; entity/edge
    resolution correctness across concurrent writers to one Neo4j group is a known, accepted
    tradeoff (see the Graphiti baseline project notes), not something this function fixes.
    """
    effective_dates = await asyncio.to_thread(
        _resolve_effective_dates, chunks, corpus_metadata, date_extraction_client, window_token_budget
    )
    eligible, skipped_no_metadata, skipped_no_date = _filter_eligible_chunks(chunks, corpus_metadata, effective_dates)

    # Workers share one iterator: advancing it never awaits, so on asyncio's single thread each
    # worker is handed a distinct chunk without needing a queue or a lock.
    backlog = iter(eligible)
    counts = {"indexed": 0, "skipped_already_indexed": 0}

    async def worker(graphiti_client: Graphiti) -> None:
        for chunk in backlog:
            if await _episode_already_indexed(neo4j_driver, chunk, corpus_metadata.group_id):
                await _normalize_fact_effective_dates(neo4j_driver, chunk, corpus_metadata.group_id)
                counts["skipped_already_indexed"] += 1
                continue
            reference_time = datetime.combine(effective_dates[chunk.source_name], time.min, tzinfo=UTC)
            await _add_episode(graphiti_client, chunk, reference_time, corpus_metadata.group_id)
            await _normalize_fact_effective_dates(neo4j_driver, chunk, corpus_metadata.group_id)
            counts["indexed"] += 1

    tasks = [asyncio.create_task(worker(client)) for client in graphiti_clients]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        failures: list[BaseException] = []
        for task in tasks:
            if task in done and not task.cancelled():
                failure = task.exception()
                if failure is not None:
                    failures.append(failure)
        if failures:
            raise failures[0]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return IndexingSummary(
        indexed=counts["indexed"],
        skipped_no_metadata=skipped_no_metadata,
        skipped_no_date=skipped_no_date,
        skipped_already_indexed=counts["skipped_already_indexed"],
    )
