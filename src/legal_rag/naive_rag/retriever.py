"""Retrieve relevant chunks from Milvus by keyword, semantic, or hybrid search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.settings import NaiveRagSettings


class SearchMode(Enum):
    """Supported retrieval strategies."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens for BM25 indexing."""
    return text.lower().split()


class Retriever:
    """Search Milvus chunks by keyword, semantic vector, or hybrid fusion."""

    def __init__(self, encoder: EmbeddingEncoder, milvus_client: MilvusClient, settings: NaiveRagSettings) -> None:
        """Initialize with an embedding encoder, Milvus client, and naive RAG settings."""
        self.encoder = encoder
        self.client = milvus_client
        self.settings = settings
        self._bm25: object | None = None
        self._bm25_chunks: list[dict] | None = None

    def _ensure_bm25_index(self) -> None:
        """Build the BM25 index lazily from Milvus on first keyword or hybrid query."""
        if self._bm25_chunks is not None:
            return

        from rank_bm25 import BM25Okapi

        iterator = self.client.query_iterator(
            collection_name=self.settings.milvus_collection_name,
            batch_size=self.settings.milvus_insert_batch_size,
            output_fields=["chunk_id", "text", "source_name", "kind"],
        )
        all_chunks: list[dict] = []
        try:
            while batch := iterator.next():
                all_chunks.extend(batch)
        finally:
            iterator.close()
        corpus = [_tokenize(chunk["text"]) for chunk in all_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_chunks = all_chunks
        logger.info("Built BM25 index over {} chunks", len(all_chunks))

    def _semantic_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Dense vector search via Milvus."""
        query_vec = self.encoder.encode_query([query])[0].tolist()
        results = self.client.search(
            collection_name=self.settings.milvus_collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["chunk_id", "text", "source_name", "kind"],
            search_params={
                "metric_type": self.settings.milvus_metric_type,
                "params": {"nprobe": self.settings.milvus_nprobe},
            },
        )
        return [
            RetrievedChunk(
                chunk_id=hit["entity"]["chunk_id"],
                text=hit["entity"]["text"],
                source_name=hit["entity"]["source_name"],
                kind=hit["entity"]["kind"],
                score=hit["distance"],
            )
            for hit in results[0]
        ]

    def _keyword_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25 keyword search over all indexed chunks."""
        self._ensure_bm25_index()
        if not self._bm25_chunks:
            return []
        bm25 = self._bm25
        chunks = self._bm25_chunks

        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        return [
            RetrievedChunk(
                chunk_id=chunks[i]["chunk_id"],
                text=chunks[i]["text"],
                source_name=chunks[i]["source_name"],
                kind=chunks[i]["kind"],
                score=float(scores[i]),
            )
            for i in ranked_indices
            if scores[i] > 0.0
        ]

    def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion of semantic and keyword results."""
        pool_size = top_k * self.settings.hybrid_pool_multiplier
        semantic_hits = self._semantic_search(query, pool_size)
        keyword_hits = self._keyword_search(query, pool_size)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, hit in enumerate(semantic_hits, start=1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (self.settings.hybrid_rrf_k + rank)
            chunk_map[hit.chunk_id] = hit

        for rank, hit in enumerate(keyword_hits, start=1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (self.settings.hybrid_rrf_k + rank)
            if hit.chunk_id not in chunk_map:
                chunk_map[hit.chunk_id] = hit

        ranked_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:top_k]
        return [
            RetrievedChunk(
                chunk_id=chunk_map[cid].chunk_id,
                text=chunk_map[cid].text,
                source_name=chunk_map[cid].source_name,
                kind=chunk_map[cid].kind,
                score=rrf_scores[cid],
            )
            for cid in ranked_ids
        ]

    def retrieve(self, query: str, top_k: int = 10, mode: SearchMode | None = None) -> list[RetrievedChunk]:
        """Dispatch to the configured or overridden search mode."""
        active_mode = mode or SearchMode(self.settings.search_mode)

        if active_mode == SearchMode.KEYWORD:
            hits = self._keyword_search(query, top_k)
        elif active_mode == SearchMode.HYBRID:
            hits = self._hybrid_search(query, top_k)
        else:
            hits = self._semantic_search(query, top_k)

        logger.debug("Retrieved {} chunks via {} search", len(hits), active_mode.value)
        return hits

    def close(self) -> None:
        """Close the Milvus client."""
        self.client.close()
