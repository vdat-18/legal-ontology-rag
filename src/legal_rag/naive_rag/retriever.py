"""Retrieve relevant chunks from Milvus by keyword, semantic, or hybrid search."""

from __future__ import annotations

from dataclasses import replace
from threading import Lock

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.benchmark_synthesis.vietnamese_lexical import VietnameseWordSegmenter
from legal_rag.datastores.milvus import MilvusCorpusFingerprint
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.naive_rag.indexer import load_chunks, preflight_collection_corpus
from legal_rag.naive_rag.models import RetrievedChunk, SearchMode
from legal_rag.naive_rag.reranker import ChunkReranker, CrossEncoderReranker
from legal_rag.naive_rag.structure import ChunkCatalog
from legal_rag.retrieval.rrf import reciprocal_rank_fusion
from legal_rag.retrieval.vietnamese_bm25 import vietnamese_legal_tokens
from legal_rag.settings import NaiveRagSettings


class Retriever:
    """Search Milvus chunks by keyword, semantic vector, or hybrid fusion."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        milvus_client: MilvusClient,
        settings: NaiveRagSettings,
        *,
        word_segmenter: VietnameseWordSegmenter | None = None,
        reranker: ChunkReranker | None = None,
        chunk_catalog: ChunkCatalog | None = None,
    ) -> None:
        """Initialize with encoder, Milvus client, settings, and optional word segmenter."""
        self.encoder = encoder
        self.client = milvus_client
        self.settings = settings
        self._word_segmenter = word_segmenter
        self._bm25: object | None = None
        self._bm25_chunks: list[dict] | None = None
        self._bm25_lock = Lock()
        self._reranker = reranker
        self._reranker_lock = Lock()
        self._chunk_catalog = chunk_catalog
        self._chunk_catalog_lock = Lock()

    def preflight_corpus(self) -> MilvusCorpusFingerprint:
        """Verify that this retriever's active collection matches its configured chunk corpus."""
        chunks = load_chunks(self.settings.chunks_dir)
        return preflight_collection_corpus(chunks, self.encoder, self.client, self.settings)

    def _ensure_bm25_index(self) -> None:
        """Build the BM25L index lazily from Milvus on first keyword or hybrid query."""
        if self._bm25_chunks is not None:
            return

        from rank_bm25 import BM25L

        with self._bm25_lock:
            if self._bm25_chunks is not None:
                return

            iterator = self.client.query_iterator(
                collection_name=self.settings.milvus_collection_name,
                batch_size=self.settings.milvus_insert_batch_size,
                output_fields=[
                    "chunk_id",
                    "text",
                    "source_name",
                    "kind",
                    "content_sha256",
                    "chunker_config_sha256",
                ],
            )
            all_chunks: list[dict] = []
            try:
                while batch := iterator.next():
                    all_chunks.extend(batch)
            finally:
                iterator.close()

            if not all_chunks:
                return

            texts = [chunk["text"] for chunk in all_chunks]
            if self._word_segmenter is not None:
                segmented = self._word_segmenter.segment(texts)
                corpus = [vietnamese_legal_tokens(tokens) for tokens in segmented]
            else:
                corpus = [[token for token in text.lower().split() if token] for text in texts]

            self._bm25 = BM25L(
                corpus,
                k1=self.settings.bm25_k1,
                b=self.settings.bm25_b,
                delta=self.settings.bm25_delta,
            )
            self._bm25_chunks = all_chunks
            logger.info("Built BM25L index over {} chunks", len(all_chunks))

    def _semantic_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Dense vector search via Milvus."""
        query_vec = self.encoder.encode_query([query])[0].tolist()
        results = self.client.search(
            collection_name=self.settings.milvus_collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=[
                "chunk_id",
                "text",
                "source_name",
                "kind",
                "content_sha256",
                "chunker_config_sha256",
            ],
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
                content_sha256=hit["entity"].get("content_sha256"),
                chunker_config_sha256=hit["entity"].get("chunker_config_sha256"),
            )
            for hit in results[0]
        ]

    def _keyword_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_bm25_index()
        if not self._bm25_chunks:
            return []
        bm25 = self._bm25
        chunks = self._bm25_chunks

        if self._word_segmenter is not None:
            segmented_query = self._word_segmenter.segment([query])
            if not segmented_query:
                return []
            query_tokens = vietnamese_legal_tokens(segmented_query[0])
        else:
            query_tokens = [token for token in query.lower().split() if token]

        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        return [
            RetrievedChunk(
                chunk_id=chunks[i]["chunk_id"],
                text=chunks[i]["text"],
                source_name=chunks[i]["source_name"],
                kind=chunks[i]["kind"],
                score=float(scores[i]),
                content_sha256=chunks[i].get("content_sha256"),
                chunker_config_sha256=chunks[i].get("chunker_config_sha256"),
            )
            for i in ranked_indices
            if scores[i] > 0.0
        ]

    def _hybrid_candidates(self, query: str, pool_size: int, limit: int) -> list[RetrievedChunk]:
        """Fuse semantic and keyword candidate pools with Reciprocal Rank Fusion."""
        semantic_hits = self._semantic_search(query, pool_size)
        keyword_hits = self._keyword_search(query, pool_size)

        fused = reciprocal_rank_fusion(
            (semantic_hits, keyword_hits),
            key=lambda hit: hit.chunk_id,
            rrf_k=self.settings.hybrid_rrf_k,
        )
        return [replace(result.item, score=result.score) for result in fused[:limit]]

    def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top RRF-fused semantic and keyword results."""
        pool_size = top_k * self.settings.hybrid_pool_multiplier
        return self._hybrid_candidates(query, pool_size, top_k)

    def _get_reranker(self) -> ChunkReranker:
        """Load the Cross-Encoder only when hybrid reranking is requested."""
        if self._reranker is not None:
            return self._reranker

        with self._reranker_lock:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker(
                    self.settings.reranker_model_id,
                    self.settings.reranker_model_revision,
                    self.settings.reranker_batch_size,
                )
        return self._reranker

    def _hybrid_rerank_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Rerank a bounded RRF candidate pool with a Cross-Encoder."""
        candidate_limit = top_k * self.settings.reranker_pool_multiplier
        candidates = self._hybrid_candidates(query, candidate_limit, candidate_limit)
        if not candidates:
            return []
        return self._get_reranker().rerank(query, candidates)[:top_k]

    def _get_chunk_catalog(self) -> ChunkCatalog:
        """Load the local structural catalog only for hierarchical retrieval."""
        if self._chunk_catalog is not None:
            return self._chunk_catalog

        with self._chunk_catalog_lock:
            if self._chunk_catalog is None:
                self._chunk_catalog = ChunkCatalog.from_directory(self.settings.chunks_dir)
        return self._chunk_catalog

    def _hierarchical_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Expand RRF seeds into bounded article-level retrieval units."""
        candidate_limit = top_k * self.settings.hybrid_pool_multiplier
        candidates = self._hybrid_candidates(query, candidate_limit, candidate_limit)
        if not candidates:
            return []

        catalog = self._get_chunk_catalog()
        hits: list[RetrievedChunk] = []
        seen_unit_ids: set[str] = set()
        for seed in candidates:
            unit = catalog.build_structural_unit(
                seed,
                max_chunks=self.settings.hierarchical_max_chunks_per_unit,
                max_tokens=self.settings.hierarchical_max_tokens_per_unit,
            )
            hit = (
                replace(
                    seed,
                    text=unit.text,
                    source_name=unit.source_name,
                    unit_id=unit.unit_id,
                    unit_kind="article",
                    evidence_chunk_ids=unit.evidence_chunk_ids,
                )
                if unit is not None
                else seed
            )
            if hit.effective_unit_id in seen_unit_ids:
                continue
            seen_unit_ids.add(hit.effective_unit_id)
            hits.append(hit)
            if len(hits) == top_k:
                break
        return hits

    def retrieve(self, query: str, top_k: int = 10, mode: SearchMode | None = None) -> list[RetrievedChunk]:
        """Dispatch to the configured or overridden search mode."""
        active_mode = mode or SearchMode(self.settings.search_mode)

        if active_mode == SearchMode.KEYWORD:
            hits = self._keyword_search(query, top_k)
        elif active_mode == SearchMode.HYBRID:
            hits = self._hybrid_search(query, top_k)
        elif active_mode == SearchMode.HYBRID_RERANK:
            hits = self._hybrid_rerank_search(query, top_k)
        elif active_mode == SearchMode.HIERARCHICAL:
            hits = self._hierarchical_search(query, top_k)
        else:
            hits = self._semantic_search(query, top_k)

        logger.debug("Retrieved {} chunks via {} search", len(hits), active_mode.value)
        return hits

    def fetch_by_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> list[RetrievedChunk]:
        """Fetch persisted source chunks by primary key, preserving caller order."""
        ordered_ids = tuple(dict.fromkeys(chunk_ids))
        if not ordered_ids:
            return []
        rows = self.client.get(
            collection_name=self.settings.milvus_collection_name,
            ids=list(ordered_ids),
            output_fields=[
                "chunk_id",
                "text",
                "source_name",
                "kind",
                "content_sha256",
                "chunker_config_sha256",
            ],
        )
        rows_by_id = {str(row["chunk_id"]): row for row in rows}
        missing = [chunk_id for chunk_id in ordered_ids if chunk_id not in rows_by_id]
        if missing:
            logger.warning("{} ontology source chunks are missing from Milvus", len(missing))
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=str(rows_by_id[chunk_id]["text"]),
                source_name=str(rows_by_id[chunk_id]["source_name"]),
                kind=str(rows_by_id[chunk_id]["kind"]),
                score=1.0,
                content_sha256=rows_by_id[chunk_id].get("content_sha256"),
                chunker_config_sha256=rows_by_id[chunk_id].get("chunker_config_sha256"),
            )
            for chunk_id in ordered_ids
            if chunk_id in rows_by_id
        ]

    def close(self) -> None:
        """Close the Milvus client."""
        self.client.close()
