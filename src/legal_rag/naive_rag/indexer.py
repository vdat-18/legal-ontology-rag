"""Load chunked JSONL data, embed with the configured profile, and index into Milvus."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.settings import NaiveRagSettings

_MAX_VARCHAR_BYTES = 65_535


def load_chunks(chunks_dir: Path) -> tuple[Chunk, ...]:
    """Read all ``*.chunks.jsonl`` files from a directory into typed chunk records."""
    chunks: list[Chunk] = []
    for chunks_path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        chunks.extend(read_chunks(chunks_path))
    logger.info("Loaded {} chunks from {}", len(chunks), chunks_dir)
    return tuple(chunks)


def _truncate_varchar(text: str, max_bytes: int = _MAX_VARCHAR_BYTES) -> str:
    """Truncate text so its UTF-8 encoding fits within *max_bytes*."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    while len(text.encode("utf-8")) > max_bytes:
        text = text[: len(text) - 100]
    return text


def _ensure_collection(client: MilvusClient, settings: NaiveRagSettings, dim: int) -> None:
    """Create the Milvus collection and index if it does not already exist."""
    if client.has_collection(settings.milvus_collection_name):
        stats = client.get_collection_stats(settings.milvus_collection_name)
        logger.info("Collection '{}' exists with {} rows", settings.milvus_collection_name, stats["row_count"])
        return

    schema = CollectionSchema(
        [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_VARCHAR_BYTES),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=_MAX_VARCHAR_BYTES),
            FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="record_scope", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=settings.milvus_index_type,
        metric_type=settings.milvus_metric_type,
        params={"nlist": settings.milvus_nlist},
    )
    client.create_collection(
        collection_name=settings.milvus_collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '{}' with {} index", settings.milvus_collection_name, settings.milvus_metric_type)


def index_chunks(
    chunks: Sequence[Chunk],
    encoder: EmbeddingEncoder,
    milvus_client: MilvusClient,
    settings: NaiveRagSettings,
) -> int:
    """Embed and insert *chunks* into Milvus, skipping entirely if already populated."""
    _ensure_collection(milvus_client, settings, encoder.profile.dimensions)

    stats = milvus_client.get_collection_stats(settings.milvus_collection_name)
    if stats["row_count"] > 0:
        logger.info("Collection already has {} rows, skipping indexing", stats["row_count"])
        return stats["row_count"]

    total_inserted = 0
    batch: list[Chunk] = []

    for position, chunk in enumerate(chunks):
        batch.append(chunk)

        if len(batch) >= settings.milvus_insert_batch_size or position == len(chunks) - 1:
            vectors = encoder.encode([item.embedding_text for item in batch])
            data = [
                {
                    "chunk_id": item.chunk_id,
                    "vector": vectors[offset].tolist(),
                    "text": _truncate_varchar(item.text),
                    "embedding_text": _truncate_varchar(item.embedding_text),
                    "source_name": item.source_name,
                    "kind": item.kind,
                    "record_scope": item.record_scope,
                    "chunk_index": item.chunk_index,
                }
                for offset, item in enumerate(batch)
            ]
            milvus_client.insert(collection_name=settings.milvus_collection_name, data=data)
            total_inserted += len(data)
            logger.debug("Inserted batch {}/{}", total_inserted, len(chunks))
            batch = []

    # Flush so a subsequent run's row-count check (used to skip re-indexing) sees this data;
    # a standalone Milvus server does not guarantee inserted rows are counted until flushed.
    # Milvus Lite (a local ".db" file, matching pymilvus's own dispatch rule) is already
    # synchronous here and its flush RPC is broken on Windows, so it is skipped for Lite.
    if not settings.milvus_uri.endswith(".db"):
        milvus_client.flush(settings.milvus_collection_name)
    logger.info("Indexed {} chunks into '{}'", total_inserted, settings.milvus_collection_name)
    return total_inserted
