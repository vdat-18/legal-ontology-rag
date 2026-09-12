"""Ontology-enhanced RAG pipeline combining vector, keyword, and ontology retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from legal_rag.naive_rag.generator import BamlGenerationClient, generate_answer
from legal_rag.naive_rag.models import RetrievedChunk, SearchMode
from legal_rag.naive_rag.retriever import Retriever
from legal_rag.ontology_rag.context.context_builder import ContextBuilder
from legal_rag.ontology_rag.fusion.rrf_fusion import RRFFusion
from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.index.ontology_index import OntologyIndex
from legal_rag.ontology_rag.index.temporal_index import TemporalIndex
from legal_rag.ontology_rag.models import OntologyInstance, PenaltyRule
from legal_rag.ontology_rag.retrieval.instance_retriever import InstanceRetriever
from legal_rag.ontology_rag.retrieval.linker import ChunkLinker
from legal_rag.ontology_rag.retrieval.sparql_search import SPARQLSearch
from legal_rag.ontology_rag.retrieval.temporal_resolver import TemporalResolver
from legal_rag.query_rewriting import SyncQueryRewriter
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings


@dataclass(frozen=True, slots=True)
class OntologyRAGResult:
    """Container for ontology-enhanced RAG result."""

    query: str
    rewritten_query: str | None = None
    answer: str = ""
    rules: tuple[PenaltyRule, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    answer_contexts: tuple[RetrievedChunk, ...] = ()


class OntologyRAGPipeline:
    """End-to-end ontology-enhanced RAG pipeline."""

    def __init__(
        self,
        settings: Settings,
        milvus_settings: MilvusSettings,
        naive_rag_settings: NaiveRagSettings,
        ontology_paths: list[Path],
    ) -> None:
        """Initialize the pipeline with settings and ontology paths."""
        self._naive_rag_settings = naive_rag_settings
        self._settings = settings

        ontology_index = OntologyIndex(ontology_paths)
        self._chunk_index = ChunkIndex(ontology_index.graph)
        self._temporal_index = TemporalIndex(ontology_index.graph)

        self._retriever = Retriever(
            self._build_encoder(naive_rag_settings),
            self._build_milvus_client(milvus_settings),
            naive_rag_settings,
        )
        self._generation_client = BamlGenerationClient(settings, naive_rag_settings)
        self._query_rewriter = SyncQueryRewriter(settings)
        self._linker = ChunkLinker(self._chunk_index)
        self._instance_retriever = InstanceRetriever(ontology_index.graph)
        self._sparql_search = SPARQLSearch(ontology_index.graph)
        self._temporal_resolver = TemporalResolver()
        self._rrf_fusion = RRFFusion()
        self._context_builder = ContextBuilder()

        logger.info("OntologyRAGPipeline initialized")

    def _build_encoder(self, settings: NaiveRagSettings):
        """Build embedding encoder."""
        from legal_rag.embedding.encoder import EmbeddingEncoder
        from legal_rag.embedding.profiles import load_configured_embedding_profile

        profile = load_configured_embedding_profile(settings.embedding_profile)
        return EmbeddingEncoder.from_profile(profile)

    def _build_milvus_client(self, settings: MilvusSettings):
        """Build Milvus client."""
        from legal_rag.datastores.milvus import build_milvus_client

        return build_milvus_client(settings)

    def run(
        self,
        query: str,
        query_date: datetime | None = None,
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
        rewrite_query: bool = True,
    ) -> OntologyRAGResult:
        """Execute the full ontology-enhanced RAG pipeline.

        Args:
            query: The user's raw question, possibly context-dependent.
            query_date: Target date for temporal filtering.
            top_k: Maximum number of results.
            history: Sequence of (question, answer) pairs from earlier turns.
            rewrite_query: Whether to rewrite the query before retrieval.

        Returns:
            OntologyRAGResult with answer and supporting evidence.
        """
        k = top_k or self._naive_rag_settings.top_k
        target_date = query_date or datetime.now(UTC).date()

        rewritten_query = query
        if rewrite_query and (history or self._should_rewrite_single_turn(query)):
            result = self._query_rewriter.rewrite(query, tuple(history))
            rewritten_query = result.rewritten
            if rewritten_query != query:
                logger.debug("Query rewritten: '{}' -> '{}'", query, rewritten_query)

        semantic_chunks = self._retriever.retrieve(rewritten_query, top_k=k, mode=SearchMode.SEMANTIC)
        keyword_chunks = self._retriever.retrieve(rewritten_query, top_k=k, mode=SearchMode.KEYWORD)
        ontology_instances = self._sparql_search.search([rewritten_query])
        linked_instances = self._linker.link([*semantic_chunks, *keyword_chunks])

        candidates = self._merge_instances([*linked_instances, *ontology_instances])
        ranked_instances = self._rrf_fusion.fuse(semantic_chunks, keyword_chunks, candidates)
        rules = self._instance_retriever.retrieve_batch([inst.uri for inst in ranked_instances[:k]])

        valid_rules = self._temporal_resolver.filter_valid(rules, target_date)
        self._context_builder.build(valid_rules, [*semantic_chunks, *keyword_chunks])
        answer_contexts = self._build_answer_contexts(valid_rules)
        answer = generate_answer(query, answer_contexts, self._generation_client)

        logger.info(
            "Pipeline completed: {} rules, {} chunks", len(valid_rules), len(semantic_chunks) + len(keyword_chunks)
        )
        return OntologyRAGResult(
            query=query,
            rewritten_query=rewritten_query if rewritten_query != query else None,
            answer=answer,
            rules=tuple(valid_rules),
            sources=tuple(semantic_chunks[:5] + keyword_chunks[:5]),
            answer_contexts=tuple(answer_contexts),
        )

    def _should_rewrite_single_turn(self, query: str) -> bool:
        """Check if a single-turn query should be rewritten for terminology normalization."""
        colloquial_markers = [
            "vượt đèn",
            "đi ngược chiều",
            "không đội mũ",
            "chạy quá tốc độ",
            "đèn đỏ",
            "bỏ chạy",
        ]
        return any(marker in query.lower() for marker in colloquial_markers)

    @staticmethod
    def _merge_instances(instances: list[OntologyInstance]) -> list[OntologyInstance]:
        """Merge per-chunk instance duplicates by URI, unioning their source chunk ids."""
        merged: dict[object, OntologyInstance] = {}
        for instance in instances:
            previous = merged.get(instance.uri)
            if previous is None:
                merged[instance.uri] = instance
            else:
                chunk_ids = tuple(dict.fromkeys((*previous.source_chunk_ids, *instance.source_chunk_ids)))
                merged[instance.uri] = OntologyInstance(
                    uri=instance.uri, type=previous.type, source_chunk_ids=chunk_ids
                )
        return list(merged.values())

    def _build_answer_contexts(self, rules: list[PenaltyRule]) -> list[RetrievedChunk]:
        """Build the rule-summary contexts actually supplied to answer generation."""
        return [
            RetrievedChunk(
                chunk_id=rule.uri.fragment or str(rule.uri),
                text=f"{rule.conduct_label_vi}: {self._format_consequences(rule)}",
                source_name=rule.provision_path,
                kind="ontology",
                score=1.0,
            )
            for rule in rules[:5]
        ]

    def _format_consequences(self, rule: PenaltyRule) -> str:
        """Format consequences for answer generation."""
        parts = []
        if rule.monetary_penalty_min and rule.monetary_penalty_max:
            parts.append(f"Phạt {rule.monetary_penalty_min}-{rule.monetary_penalty_max} {rule.currency or 'VND'}")
        if rule.point_value:
            parts.append(f"Trừ {rule.point_value} điểm")
        return ", ".join(parts) if parts else "Không có thông tin phạt"

    def close(self) -> None:
        """Close the pipeline connections."""
        self._retriever.close()
