"""Milvus database adapter for indexing and retrieving chunk documents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel


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
        """Ensure that the legal_chunks collection exists with the correct schema."""
        has_collection = self.client.has_collection(collection_name=self.collection_name)

        if not has_collection:
            logger.info(f"Creating Milvus collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedder.dimensions,
                primary_field_name="chunk_id",
                id_type="string",
                max_length=128,
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )

        else:
            logger.info(f"Using existing Milvus collection: {self.collection_name}")

        self.client.load_collection(collection_name=self.collection_name)

    def index_processed_data(self, processed_dir: str = "data/processed/groundtruth") -> None:
        """Index all chunk files in the processed folder.

        Args:
            processed_dir: Path to directory containing chunk JSONL files.
        """
        path = Path(processed_dir)
        if not path.exists():
            logger.error(f"Processed data directory does not exist: {processed_dir}")
            return

        chunk_files = list(path.glob("*.chunks.jsonl"))
        if not chunk_files:
            logger.warning(f"No *.chunks.jsonl files found in {processed_dir}")
            return

        logger.info(f"Found {len(chunk_files)} chunk files to index.")

        for file_path in chunk_files:
            self._index_file(file_path)

        logger.info("Indexing process completed.")

    def _index_file(self, file_path: Path) -> None:
        """Index a single JSONL chunk file into Milvus.

        Args:
            file_path: Path to the *.chunks.jsonl file.
        """
        logger.info(f"Processing chunk file: {file_path.name}")

        # Read the file's lines to load all chunk data
        with file_path.open(encoding="utf-8") as f:
            chunks_data: list[Mapping[str, Any]] = [json.loads(line) for line in f if line.strip()]

        if not chunks_data:
            logger.warning(f"Chunk file {file_path.name} is empty.")
            return

        # Filter out chunks that are already indexed to avoid duplicates
        existing_ids = set()
        chunk_ids = [c["chunk_id"] for c in chunks_data]

        for i in range(0, len(chunk_ids), 100):
            batch_ids = chunk_ids[i : i + 100]
            results = self.client.get(collection_name=self.collection_name, ids=batch_ids, output_fields=["chunk_id"])
            for res in results:
                existing_ids.add(res["chunk_id"])

        new_chunks = [c for c in chunks_data if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            logger.info(f"All chunks from {file_path.name} are already indexed.")
            return

        index_batch_size = max(1, self.config.index_batch_size)
        logger.info(
            "Indexing {} new chunks from {} with batch size {}...",
            len(new_chunks),
            file_path.name,
            index_batch_size,
        )

        indexed_count = 0
        for i in range(0, len(new_chunks), index_batch_size):
            chunk_batch = new_chunks[i : i + index_batch_size]
            logger.info(
                "Embedding and inserting chunks {}-{} of {} from {}...",
                i + 1,
                i + len(chunk_batch),
                len(new_chunks),
                file_path.name,
            )

            texts_to_embed = [chunk.get("embedding_text") or chunk.get("text") or "" for chunk in chunk_batch]
            embeddings = self.embedder.embed_documents(texts_to_embed)
            data_to_insert = []
            for chunk, vector in zip(chunk_batch, embeddings, strict=True):
                data_to_insert.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "vector": vector,
                        "text": chunk.get("text", ""),
                        "embedding_text": chunk.get("embedding_text", ""),
                        "source_name": chunk.get("source_name", ""),
                        "scope_id": chunk.get("scope_id", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "previous_chunk_id": chunk.get("previous_chunk_id"),
                        "next_chunk_id": chunk.get("next_chunk_id"),
                    }
                )

            self.client.insert(collection_name=self.collection_name, data=data_to_insert)
            indexed_count += len(data_to_insert)

        self.client.flush(collection_name=self.collection_name)

        logger.info("Successfully indexed {} chunks from {}", indexed_count, file_path.name)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Perform vector similarity search in the Milvus database.

        Args:
            query: The user query string.
            limit: Maximum number of search results to retrieve.

        Returns:
            A list of dictionary objects representing the retrieved chunks with scores.
        """
        logger.info(f"Performing similarity search for query: '{query}'")

        query_vector = self.embedder.embed_query(query)
        self.client.load_collection(collection_name=self.collection_name)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=[
                "chunk_id",
                "text",
                "embedding_text",
                "source_name",
                "scope_id",
                "chunk_index",
                "previous_chunk_id",
                "next_chunk_id",
            ],
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
                        "previous_chunk_id": entity.get("previous_chunk_id"),
                        "next_chunk_id": entity.get("next_chunk_id"),
                    }
                )

        logger.info(f"Retrieved {len(formatted_results)} results from vector search.")
        return formatted_results

    def close(self) -> None:
        """Close the Milvus connection."""
        logger.info("Closing Milvus database client connection...")
        self.client.close()
