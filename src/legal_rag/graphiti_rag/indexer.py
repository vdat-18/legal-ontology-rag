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
from neo4j import AsyncDriver, AsyncGraphDatabase

from legal_rag.chunker.models import Chunk
from legal_rag.graphiti_rag.effective_date import EffectiveDateClient, extract_effective_date
from legal_rag.graphiti_rag.ontology import EDGE_TYPES, ENTITY_TYPES
from legal_rag.settings import GraphitiSettings


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """Temporal and identifying metadata for one legal document in a ground-truth corpus."""

    so_hieu: str
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
    documents = {
        document["file"]: CorpusDocument(
            so_hieu=document["so_hieu"],
            effective_date=date.fromisoformat(document["effective_date"]) if document["effective_date"] else None,
            amends=document["amends"],
        )
        for document in config["documents"]
    }
    return CorpusMetadata(group_id=config["group_id"], documents=documents)


def build_neo4j_driver(settings: GraphitiSettings) -> AsyncDriver:
    """Open an async Neo4j driver used for the episode-existence idempotency check."""
    return AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


async def _episode_already_indexed(neo4j_driver: AsyncDriver, name: str, group_id: str) -> bool:
    """Return whether an Episodic node with this *name* in *group_id* already exists.

    Graphiti has no upsert-by-uuid for new episodes: passing ``uuid=`` to ``add_episode``
    means "re-fetch this existing episode", and raises ``NodeNotFoundError`` if it is not
    already present. So the idempotency check has to key on something set before the episode
    exists — the chunk's stable ``chunk_id`` this indexer uses as the episode name, scoped to
    the corpus group.
    """
    records, _, _ = await neo4j_driver.execute_query(
        "MATCH (e:Episodic {name: $name, group_id: $group_id}) RETURN e.uuid LIMIT 1",
        name=name,
        group_id=group_id,
    )
    return len(records) > 0


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
        if await _episode_already_indexed(neo4j_driver, chunk.chunk_id, corpus_metadata.group_id):
            skipped_already_indexed += 1
            continue

        reference_time = datetime.combine(effective_dates[chunk.source_name], time.min, tzinfo=UTC)
        await _add_episode(graphiti_client, chunk, reference_time, corpus_metadata.group_id)
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
            if await _episode_already_indexed(neo4j_driver, chunk.chunk_id, corpus_metadata.group_id):
                counts["skipped_already_indexed"] += 1
                continue
            reference_time = datetime.combine(effective_dates[chunk.source_name], time.min, tzinfo=UTC)
            await _add_episode(graphiti_client, chunk, reference_time, corpus_metadata.group_id)
            counts["indexed"] += 1

    await asyncio.gather(*(worker(client) for client in graphiti_clients))

    return IndexingSummary(
        indexed=counts["indexed"],
        skipped_no_metadata=skipped_no_metadata,
        skipped_no_date=skipped_no_date,
        skipped_already_indexed=counts["skipped_already_indexed"],
    )
