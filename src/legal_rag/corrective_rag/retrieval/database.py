"""Milvus database adapter for indexing and retrieving chunk documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks
from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel

_MAX_VARCHAR_BYTES = 65_535
_CHUNK_ID_MAX_LENGTH = 256
_SOURCE_NAME_MAX_LENGTH = 512
_KIND_MAX_LENGTH = 64
_RECORD_SCOPE_MAX_LENGTH = 64
_SCOPE_ID_MAX_LENGTH = 256
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
    "scope_id": DataType.VARCHAR,
    "chunk_index": DataType.INT64,
    "previous_chunk_id": DataType.VARCHAR,
    "next_chunk_id": DataType.VARCHAR,
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
    "scope_id": _SCOPE_ID_MAX_LENGTH,
    "previous_chunk_id": _CHUNK_ID_MAX_LENGTH,
    "next_chunk_id": _CHUNK_ID_MAX_LENGTH,
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
_SEARCH_OUTPUT_FIELDS = (
    "chunk_id",
    "text",
    "embedding_text",
    "source_name",
    "scope_id",
    "chunk_index",
    "previous_chunk_id",
    "next_chunk_id",
)


class LegalVectorDB:
    """Manages the Milvus collection, document indexing, and similarity search."""

    def __init__(self, config: RAGConfig, embedding_model: EmbeddingModel) -> None:
        """Initialize the Milvus database adapter.

        Args:
            config: RAG config loaded from settings.
            embedding_model: Initialized embedding model adapter.
        """
        self.config = config
        self.embedder = embedding_model

        logger.info("Connecting to Milvus at: {}", self.config.milvus_uri)
        connection_options: dict[str, Any] = {
            "uri": self.config.milvus_uri,
            "grpc_options": {
                "grpc.keepalive_time_ms": 60000,
                "grpc.keepalive_timeout_ms": 20000,
                "grpc.keepalive_permit_without_calls": False,
            },
        }
        if self.config.milvus_token is not None:
            connection_options["token"] = self.config.milvus_token
        self.client = MilvusClient(**connection_options)
        if self.config.milvus_database not in self.client.list_databases():
            self.client.create_database(self.config.milvus_database)
        self.client.use_database(self.config.milvus_database)

        self.collection_name = self.config.milvus_collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection or validate its existing schema and vector index."""
        has_collection = self.client.has_collection(collection_name=self.collection_name)

        if not has_collection:
            logger.info(f"Creating Milvus collection: {self.collection_name}")
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=self.config.milvus_index_type,
                metric_type=self.config.milvus_metric_type,
                params={"nlist": self.config.milvus_nlist},
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=_collection_schema(self.embedder.dimensions),
                index_params=index_params,
            )

        else:
            logger.info(f"Using existing Milvus collection: {self.collection_name}")
            self._validate_collection()

        self.client.load_collection(collection_name=self.collection_name)

    def _validate_collection(self) -> None:
        """Fail when the persisted Milvus collection is incompatible."""
        description = self.client.describe_collection(collection_name=self.collection_name)
        fields = {field["name"]: field for field in description["fields"]}
        _validate_field_schema(fields, self.collection_name, self.embedder.dimensions)
        self._validate_vector_index()

    def _validate_vector_index(self) -> None:
        """Validate the persisted vector index type and distance metric."""
        vector_index: Mapping[str, Any] | None = None
        for index_name in self.client.list_indexes(collection_name=self.collection_name):
            index = self.client.describe_index(collection_name=self.collection_name, index_name=index_name)
            if index.get("field_name") == "vector":
                vector_index = index
                break
        if vector_index is None:
            msg = f"Milvus collection '{self.collection_name}' has no vector index"
            raise ValueError(msg)
        if str(vector_index["index_type"]).upper() != self.config.milvus_index_type.upper():
            msg = (
                f"Milvus collection '{self.collection_name}' uses index type "
                f"{vector_index['index_type']}, expected {self.config.milvus_index_type}"
            )
            raise ValueError(msg)
        if str(vector_index["metric_type"]).upper() != self.config.milvus_metric_type.upper():
            msg = (
                f"Milvus collection '{self.collection_name}' uses metric "
                f"{vector_index['metric_type']}, expected {self.config.milvus_metric_type}"
            )
            raise ValueError(msg)

    def index_processed_data(self, processed_dir: str = "data/processed/groundtruth") -> int:
        """Index all chunk files in the processed folder.

        Args:
            processed_dir: Path to directory containing chunk JSONL files.
        """
        path = Path(processed_dir)
        if not path.exists():
            msg = f"Processed data directory does not exist: {processed_dir}"
            raise FileNotFoundError(msg)

        chunk_files = sorted(path.glob("*.chunks.jsonl"))
        if not chunk_files:
            msg = f"No *.chunks.jsonl files found in {processed_dir}"
            raise ValueError(msg)

        logger.info(f"Found {len(chunk_files)} chunk files to index.")
        chunks: list[Chunk] = []
        for file_path in chunk_files:
            chunks.extend(read_chunks(file_path))

        total = self.index_chunks(tuple(chunks))

        logger.info("Indexing process completed.")
        return total

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Index missing chunks and reject incompatible existing rows."""
        if not chunks:
            msg = "Input corpus contains no chunks"
            raise ValueError(msg)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            msg = "Input corpus contains duplicate chunk_id values"
            raise ValueError(msg)

        self._ensure_collection()
        existing_rows = self._read_existing_metadata()
        _validate_existing_metadata(chunks_by_id, existing_rows, self.embedder)
        missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_rows]
        index_batch_size = max(1, self.config.index_batch_size)
        logger.info(
            "Indexing {} missing chunks with batch size {}...",
            len(missing_chunks),
            index_batch_size,
        )

        indexed_count = 0
        for i in range(0, len(missing_chunks), index_batch_size):
            chunk_batch = missing_chunks[i : i + index_batch_size]
            logger.info(
                "Embedding and inserting chunks {}-{} of {}...",
                i + 1,
                i + len(chunk_batch),
                len(missing_chunks),
            )

            embeddings = self.embedder.embed_documents([chunk.embedding_text for chunk in chunk_batch])
            data_to_insert = [
                _row(chunk, vector, self.embedder) for chunk, vector in zip(chunk_batch, embeddings, strict=True)
            ]
            self.client.insert(collection_name=self.collection_name, data=data_to_insert)
            indexed_count += len(data_to_insert)

        if indexed_count and self.config.milvus_flush_after_insert:
            self.client.flush(collection_name=self.collection_name)

        indexed_ids = self._read_existing_metadata().keys()
        expected_ids = chunks_by_id.keys()
        if indexed_ids != expected_ids:
            missing_ids = sorted(expected_ids - indexed_ids)
            unexpected_ids = sorted(indexed_ids - expected_ids)
            msg = f"Milvus indexing verification failed; missing={missing_ids[:5]}, unexpected={unexpected_ids[:5]}"
            raise ValueError(msg)

        logger.info(
            "Corrective RAG collection '{}' is complete with {} rows ({} inserted)",
            self.collection_name,
            len(chunks),
            indexed_count,
        )
        return len(chunks)

    def _read_existing_metadata(self) -> dict[str, dict[str, Any]]:
        """Read persisted chunk identity metadata without legal text or vectors."""
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=max(1, self.config.index_batch_size),
            output_fields=list(_METADATA_FIELDS),
        )
        rows: dict[str, dict[str, Any]] = {}
        try:
            while batch := iterator.next():
                rows.update((row["chunk_id"], row) for row in batch)
        finally:
            iterator.close()
        return rows

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Perform vector similarity search in the Milvus database.

        Args:
            query: The user query string.
            limit: Maximum number of search results to retrieve.

        Returns:
            A list of dictionary objects representing the retrieved chunks with scores.
        """
        logger.info("Performing similarity search for one query.")

        query_vector = self.embedder.embed_query(query)
        self.client.load_collection(collection_name=self.collection_name)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=list(_SEARCH_OUTPUT_FIELDS),
            search_params={
                "metric_type": self.config.milvus_metric_type,
                "params": {"nprobe": self.config.milvus_nprobe},
            },
        )

        formatted_results = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit.get("entity", {})
                formatted_results.append(
                    {
                        "chunk_id": hit.get("id") or entity.get("chunk_id"),
                        "score": hit.get("distance"),
                        "text": entity.get("text", ""),
                        "embedding_text": entity.get("embedding_text", ""),
                        "source_name": entity.get("source_name", ""),
                        "scope_id": entity.get("scope_id", ""),
                        "chunk_index": entity.get("chunk_index", 0),
                        "previous_chunk_id": entity.get("previous_chunk_id") or None,
                        "next_chunk_id": entity.get("next_chunk_id") or None,
                    }
                )

        logger.info(f"Retrieved {len(formatted_results)} results from vector search.")
        return formatted_results

    def close(self) -> None:
        """Close the Milvus connection."""
        logger.info("Closing Milvus database client connection...")
        self.client.close()


def _collection_schema(dim: int) -> CollectionSchema:
    """Build the persisted Milvus schema for CRAG chunks."""
    return CollectionSchema(
        [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=_CHUNK_ID_MAX_LENGTH),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_VARCHAR_BYTES),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=_MAX_VARCHAR_BYTES),
            FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=_SOURCE_NAME_MAX_LENGTH),
            FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=_KIND_MAX_LENGTH),
            FieldSchema(name="record_scope", dtype=DataType.VARCHAR, max_length=_RECORD_SCOPE_MAX_LENGTH),
            FieldSchema(name="scope_id", dtype=DataType.VARCHAR, max_length=_SCOPE_ID_MAX_LENGTH),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="previous_chunk_id", dtype=DataType.VARCHAR, max_length=_CHUNK_ID_MAX_LENGTH),
            FieldSchema(name="next_chunk_id", dtype=DataType.VARCHAR, max_length=_CHUNK_ID_MAX_LENGTH),
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


def _chunk_metadata(chunk: Chunk, embedder: EmbeddingModel) -> dict[str, str]:
    """Return persisted identity metadata used to detect stale embeddings."""
    return {
        "content_sha256": chunk.content_sha256,
        "chunker_config_sha256": chunk.chunker_config_sha256,
        "embedding_profile_id": embedder.profile_id,
        "embedding_model_id": embedder.model_id,
        "embedding_model_revision": embedder.model_revision,
    }


def _validate_existing_metadata(
    chunks_by_id: Mapping[str, Chunk],
    existing_rows: Mapping[str, Mapping[str, Any]],
    embedder: EmbeddingModel,
) -> None:
    """Reject stale, mixed, or out-of-scope rows before inserting missing chunks."""
    unexpected_ids = sorted(existing_rows.keys() - chunks_by_id.keys())
    if unexpected_ids:
        sample = ", ".join(unexpected_ids[:5])
        msg = f"Milvus collection contains chunk IDs outside the current corpus: {sample}; rebuild the collection"
        raise ValueError(msg)

    for chunk_id, row in existing_rows.items():
        expected = _chunk_metadata(chunks_by_id[chunk_id], embedder)
        mismatched_fields = [name for name, value in expected.items() if row[name] != value]
        if mismatched_fields:
            fields = ", ".join(mismatched_fields)
            msg = f"Milvus chunk '{chunk_id}' has stale metadata ({fields}); rebuild the collection"
            raise ValueError(msg)


def _row(chunk: Chunk, vector: Sequence[float], embedder: EmbeddingModel) -> dict[str, Any]:
    """Build one Milvus row from a chunk and its embedding."""
    return {
        "chunk_id": chunk.chunk_id,
        "vector": list(vector),
        "text": _truncate_varchar(chunk.text),
        "embedding_text": _truncate_varchar(chunk.embedding_text),
        "source_name": chunk.source_name,
        "kind": chunk.kind,
        "record_scope": chunk.record_scope,
        "scope_id": chunk.scope_id,
        "chunk_index": chunk.chunk_index,
        "previous_chunk_id": chunk.previous_chunk_id or "",
        "next_chunk_id": chunk.next_chunk_id or "",
        **_chunk_metadata(chunk, embedder),
    }


def _truncate_varchar(text: str, max_bytes: int = _MAX_VARCHAR_BYTES) -> str:
    """Truncate text so its UTF-8 encoding fits within Milvus VARCHAR limits."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
