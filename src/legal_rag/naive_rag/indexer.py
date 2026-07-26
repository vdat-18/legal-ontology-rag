"""Load chunked JSONL data, embed with BGE-M3, and index into Milvus."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from legal_rag.naive_rag.config import MilvusConfig, NaiveRAGConfig
from legal_rag.naive_rag.embedder import BGEM3Embedder

_MAX_VARCHAR_BYTES = 65_535


def load_chunks(chunks_dir: Path) -> list[dict]:
    """Read all ``*.chunks.jsonl`` files from a directory.

    Args:
        chunks_dir: Directory containing chunked JSONL files.

    Returns:
        List of chunk dictionaries in file-name order.
    """
    chunks: list[dict] = []
    for jsonl_file in sorted(chunks_dir.glob("*.chunks.jsonl")):
        chunks.extend(json.loads(line) for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line.strip())
    logger.info("Loaded {} chunks from {}", len(chunks), chunks_dir)
    return chunks


def _truncate_varchar(text: str, max_bytes: int = _MAX_VARCHAR_BYTES) -> str:
    """Truncate text so its UTF-8 encoding fits within *max_bytes*."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    while len(text.encode("utf-8")) > max_bytes:
        text = text[: len(text) - 100]
    return text


def _ensure_collection(client: MilvusClient, cfg: MilvusConfig, dim: int) -> None:
    """Create the Milvus collection and index if it does not already exist."""
    if client.has_collection(cfg.collection_name):
        stats = client.get_collection_stats(cfg.collection_name)
        logger.info("Collection '{}' exists with {} rows", cfg.collection_name, stats["row_count"])
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
        index_type=cfg.index_type,
        metric_type=cfg.metric_type,
        params={"nlist": cfg.nlist},
    )
    client.create_collection(
        collection_name=cfg.collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '{}' with {} index", cfg.collection_name, cfg.metric_type)


def index_chunks(config: NaiveRAGConfig | None = None) -> int:
    """Run the full indexing pipeline: load → embed → insert.

    Args:
        config: Pipeline configuration. Uses defaults when *None*.

    Returns:
        Number of chunks indexed.
    """
    cfg = config or NaiveRAGConfig()
    embedder = BGEM3Embedder(cfg.embedder)
    client = MilvusClient(uri=cfg.milvus.uri)
    _ensure_collection(client, cfg.milvus, embedder.dim)

    # Skip if already indexed
    stats = client.get_collection_stats(cfg.milvus.collection_name)
    if stats["row_count"] > 0:
        logger.info("Collection already has {} rows, skipping indexing", stats["row_count"])
        return stats["row_count"]

    chunks = load_chunks(cfg.chunks_dir)
    total_inserted = 0
    batch_records: list[dict] = []
    batch_texts: list[str] = []

    for i, chunk in enumerate(chunks):
        batch_records.append(chunk)
        batch_texts.append(chunk["embedding_text"])

        if len(batch_records) >= cfg.milvus.insert_batch_size or i == len(chunks) - 1:
            vectors = embedder.encode(batch_texts)
            data = [
                {
                    "chunk_id": rec["chunk_id"],
                    "vector": vectors[j].tolist(),
                    "text": _truncate_varchar(rec["text"]),
                    "embedding_text": _truncate_varchar(rec["embedding_text"]),
                    "source_name": rec["source_name"],
                    "kind": rec["kind"],
                    "record_scope": rec.get("record_scope", "root"),
                    "chunk_index": rec.get("chunk_index", 0),
                }
                for j, rec in enumerate(batch_records)
            ]
            client.insert(collection_name=cfg.milvus.collection_name, data=data)
            total_inserted += len(data)
            logger.debug("Inserted batch {}/{}", total_inserted, len(chunks))
            batch_records = []
            batch_texts = []

    logger.info("Indexed {} chunks into '{}'", total_inserted, cfg.milvus.collection_name)
    return total_inserted
