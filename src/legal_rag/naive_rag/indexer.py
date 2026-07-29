"""Load chunked JSONL data, embed with the configured profile, and index into Milvus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.settings import NaiveRagSettings

_MAX_VARCHAR_BYTES = 65_535
_CHUNK_ID_MAX_LENGTH = 256
_SOURCE_NAME_MAX_LENGTH = 512
_KIND_MAX_LENGTH = 64
_RECORD_SCOPE_MAX_LENGTH = 64
_HASH_MAX_LENGTH = 64
_EMBEDDING_PROFILE_MAX_LENGTH = 256
_MODEL_ID_MAX_LENGTH = 512
_MODEL_REVISION_MAX_LENGTH = 256
_EXPECTED_FIELD_TYPES = {
    "chunk_id": DataType.VARCHAR,
    "vector": DataType.FLOAT_VECTOR,
    "text": DataType.VARCHAR,
    "embedding_text": DataType.VARCHAR,
    "source_name": DataType.VARCHAR,
    "kind": DataType.VARCHAR,
    "record_scope": DataType.VARCHAR,
    "chunk_index": DataType.INT64,
    "content_sha256": DataType.VARCHAR,
    "chunker_config_sha256": DataType.VARCHAR,
    "embedding_profile_id": DataType.VARCHAR,
    "embedding_model_id": DataType.VARCHAR,
    "embedding_model_revision": DataType.VARCHAR,
}
_EXPECTED_VARCHAR_LENGTHS = {
    "chunk_id": _CHUNK_ID_MAX_LENGTH,
    "text": _MAX_VARCHAR_BYTES,
    "embedding_text": _MAX_VARCHAR_BYTES,
    "source_name": _SOURCE_NAME_MAX_LENGTH,
    "kind": _KIND_MAX_LENGTH,
    "record_scope": _RECORD_SCOPE_MAX_LENGTH,
    "content_sha256": _HASH_MAX_LENGTH,
    "chunker_config_sha256": _HASH_MAX_LENGTH,
    "embedding_profile_id": _EMBEDDING_PROFILE_MAX_LENGTH,
    "embedding_model_id": _MODEL_ID_MAX_LENGTH,
    "embedding_model_revision": _MODEL_REVISION_MAX_LENGTH,
}
_METADATA_FIELDS = (
    "chunk_id",
    "content_sha256",
    "chunker_config_sha256",
    "embedding_profile_id",
    "embedding_model_id",
    "embedding_model_revision",
)


def load_chunks(chunks_dir: Path) -> tuple[Chunk, ...]:
    """Read all ``*.chunks.jsonl`` files from a directory into typed chunk records."""
    chunks: list[Chunk] = []
    for chunks_path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        chunks.extend(read_chunks(chunks_path))
    logger.info("Loaded {} chunks from {}", len(chunks), chunks_dir)
    return tuple(chunks)


def _truncate_varchar(text: str, max_bytes: int = _MAX_VARCHAR_BYTES) -> str:
    """Truncate text so its UTF-8 encoding fits within *max_bytes*."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _collection_schema(dim: int) -> CollectionSchema:
    """Build the persisted Milvus schema for naive RAG chunks."""
    return CollectionSchema(
        [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=_CHUNK_ID_MAX_LENGTH),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_VARCHAR_BYTES),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=_MAX_VARCHAR_BYTES),
            FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=_SOURCE_NAME_MAX_LENGTH),
            FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=_KIND_MAX_LENGTH),
            FieldSchema(name="record_scope", dtype=DataType.VARCHAR, max_length=_RECORD_SCOPE_MAX_LENGTH),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content_sha256", dtype=DataType.VARCHAR, max_length=_HASH_MAX_LENGTH),
            FieldSchema(name="chunker_config_sha256", dtype=DataType.VARCHAR, max_length=_HASH_MAX_LENGTH),
            FieldSchema(name="embedding_profile_id", dtype=DataType.VARCHAR, max_length=_EMBEDDING_PROFILE_MAX_LENGTH),
            FieldSchema(name="embedding_model_id", dtype=DataType.VARCHAR, max_length=_MODEL_ID_MAX_LENGTH),
            FieldSchema(name="embedding_model_revision", dtype=DataType.VARCHAR, max_length=_MODEL_REVISION_MAX_LENGTH),
        ]
    )


def _validate_field_schema(fields: Mapping[str, Mapping[str, Any]], collection_name: str, dim: int) -> None:
    """Validate required field types, capacities, primary key, and vector dimension."""
    for name, expected_type in _EXPECTED_FIELD_TYPES.items():
        field = fields.get(name)
        if field is None or field["type"] != expected_type:
            msg = f"Milvus collection '{collection_name}' has an incompatible field '{name}'"
            raise ValueError(msg)

    for name, minimum_length in _EXPECTED_VARCHAR_LENGTHS.items():
        if int(fields[name]["params"]["max_length"]) < minimum_length:
            msg = f"Milvus collection '{collection_name}' has an incompatible max_length for '{name}'"
            raise ValueError(msg)

    if not fields["chunk_id"].get("is_primary", False):
        msg = f"Milvus collection '{collection_name}' must use chunk_id as its primary key"
        raise ValueError(msg)
    if int(fields["vector"]["params"]["dim"]) != dim:
        msg = (
            f"Milvus collection '{collection_name}' has vector dimension "
            f"{fields['vector']['params']['dim']}, expected {dim}"
        )
        raise ValueError(msg)


def _validate_vector_index(client: MilvusClient, settings: NaiveRagSettings) -> None:
    """Validate the existing vector index type and distance metric."""
    vector_index: Mapping[str, Any] | None = None
    for index_name in client.list_indexes(settings.milvus_collection_name):
        index = client.describe_index(settings.milvus_collection_name, index_name)
        if index.get("field_name") == "vector":
            vector_index = index
            break
    if vector_index is None:
        msg = f"Milvus collection '{settings.milvus_collection_name}' has no vector index"
        raise ValueError(msg)
    if str(vector_index["index_type"]).upper() != settings.milvus_index_type.upper():
        msg = (
            f"Milvus collection '{settings.milvus_collection_name}' uses index type "
            f"{vector_index['index_type']}, expected {settings.milvus_index_type}"
        )
        raise ValueError(msg)
    if str(vector_index["metric_type"]).upper() != settings.milvus_metric_type.upper():
        msg = (
            f"Milvus collection '{settings.milvus_collection_name}' uses metric "
            f"{vector_index['metric_type']}, expected {settings.milvus_metric_type}"
        )
        raise ValueError(msg)


def _validate_collection(client: MilvusClient, settings: NaiveRagSettings, dim: int) -> None:
    """Fail when an existing collection cannot safely store the configured corpus."""
    description = client.describe_collection(settings.milvus_collection_name)
    fields = {field["name"]: field for field in description["fields"]}
    _validate_field_schema(fields, settings.milvus_collection_name, dim)
    _validate_vector_index(client, settings)


def _ensure_collection(client: MilvusClient, settings: NaiveRagSettings, dim: int) -> None:
    """Create the configured collection or validate its persisted schema and index."""
    if client.has_collection(settings.milvus_collection_name):
        _validate_collection(client, settings, dim)
        return

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=settings.milvus_index_type,
        metric_type=settings.milvus_metric_type,
        params={"nlist": settings.milvus_nlist},
    )
    client.create_collection(
        collection_name=settings.milvus_collection_name,
        schema=_collection_schema(dim),
        index_params=index_params,
    )
    logger.info("Created collection '{}' with {} index", settings.milvus_collection_name, settings.milvus_metric_type)


def _read_existing_metadata(client: MilvusClient, settings: NaiveRagSettings) -> dict[str, dict[str, Any]]:
    """Read persisted chunk identity metadata without loading vectors or legal text."""
    iterator = client.query_iterator(
        collection_name=settings.milvus_collection_name,
        batch_size=settings.milvus_insert_batch_size,
        output_fields=list(_METADATA_FIELDS),
    )
    rows: dict[str, dict[str, Any]] = {}
    try:
        while batch := iterator.next():
            rows.update((row["chunk_id"], row) for row in batch)
    finally:
        iterator.close()
    return rows


def _chunk_metadata(chunk: Chunk, encoder: EmbeddingEncoder) -> dict[str, str]:
    """Return persisted identity metadata used to detect stale embeddings."""
    return {
        "content_sha256": chunk.content_sha256,
        "chunker_config_sha256": chunk.chunker_config_sha256,
        "embedding_profile_id": encoder.profile.profile_id,
        "embedding_model_id": encoder.profile.model_id,
        "embedding_model_revision": encoder.profile.model_revision,
    }


def _validate_existing_metadata(
    chunks_by_id: Mapping[str, Chunk],
    existing_rows: Mapping[str, Mapping[str, Any]],
    encoder: EmbeddingEncoder,
) -> None:
    """Reject stale, mixed, or out-of-scope rows before indexing any missing chunks."""
    unexpected_ids = sorted(existing_rows.keys() - chunks_by_id.keys())
    if unexpected_ids:
        sample = ", ".join(unexpected_ids[:5])
        msg = f"Milvus collection contains chunk IDs outside the current corpus: {sample}; rebuild the collection"
        raise ValueError(msg)

    for chunk_id, row in existing_rows.items():
        expected = _chunk_metadata(chunks_by_id[chunk_id], encoder)
        mismatched_fields = [name for name, value in expected.items() if row[name] != value]
        if mismatched_fields:
            fields = ", ".join(mismatched_fields)
            msg = f"Milvus chunk '{chunk_id}' has stale metadata ({fields}); rebuild the collection"
            raise ValueError(msg)


def _row(chunk: Chunk, vector: Sequence[float], encoder: EmbeddingEncoder) -> dict[str, Any]:
    """Build one Milvus row from a chunk and its embedding."""
    return {
        "chunk_id": chunk.chunk_id,
        "vector": list(vector),
        "text": _truncate_varchar(chunk.text),
        "embedding_text": _truncate_varchar(chunk.embedding_text),
        "source_name": chunk.source_name,
        "kind": chunk.kind,
        "record_scope": chunk.record_scope,
        "chunk_index": chunk.chunk_index,
        **_chunk_metadata(chunk, encoder),
    }


def index_chunks(
    chunks: Sequence[Chunk],
    encoder: EmbeddingEncoder,
    milvus_client: MilvusClient,
    settings: NaiveRagSettings,
) -> int:
    """Index missing chunks and return the verified row count for the current corpus."""
    if not chunks:
        msg = "Input corpus contains no chunks"
        raise ValueError(msg)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        msg = "Input corpus contains duplicate chunk_id values"
        raise ValueError(msg)

    _ensure_collection(milvus_client, settings, encoder.profile.dimensions)
    existing_rows = _read_existing_metadata(milvus_client, settings)
    _validate_existing_metadata(chunks_by_id, existing_rows, encoder)
    missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_rows]

    total_inserted = 0
    for batch_start in range(0, len(missing_chunks), settings.milvus_insert_batch_size):
        batch = missing_chunks[batch_start : batch_start + settings.milvus_insert_batch_size]
        vectors = encoder.encode([item.embedding_text for item in batch])
        data = [_row(item, vectors[offset].tolist(), encoder) for offset, item in enumerate(batch)]
        milvus_client.insert(collection_name=settings.milvus_collection_name, data=data)
        total_inserted += len(data)
        logger.debug("Inserted batch {}/{}", total_inserted, len(missing_chunks))

    if total_inserted and settings.milvus_flush_after_insert:
        milvus_client.flush(settings.milvus_collection_name)

    indexed_ids = _read_existing_metadata(milvus_client, settings).keys()
    expected_ids = chunks_by_id.keys()
    if indexed_ids != expected_ids:
        missing_ids = sorted(expected_ids - indexed_ids)
        unexpected_ids = sorted(indexed_ids - expected_ids)
        msg = f"Milvus indexing verification failed; missing={missing_ids[:5]}, unexpected={unexpected_ids[:5]}"
        raise ValueError(msg)

    logger.info(
        "Naive RAG collection '{}' is complete with {} rows ({} inserted)",
        settings.milvus_collection_name,
        len(chunks),
        total_inserted,
    )
    return len(chunks)
