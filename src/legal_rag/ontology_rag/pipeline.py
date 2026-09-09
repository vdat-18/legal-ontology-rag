"""Ontology-enhanced RAG pipeline combining vector, keyword, and ontology retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from legal_rag.naive_rag.generator import BamlGenerationClient, generate_answer
from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever, SearchMode
from legal_rag.ontology_rag.context.context_builder import ContextBuilder
from legal_rag.ontology_rag.fusion.rrf_fusion import RRFFusion
from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.index.ontology_index import OntologyIndex
from legal_rag.ontology_rag.index.temporal_index import TemporalIndex
from legal_rag.ontology_rag.models import PenaltyRule
from legal_rag.ontology_rag.retrieval.instance_retriever import InstanceRetriever
from legal_rag.ontology_rag.retrieval.linker import ChunkLinker
from legal_rag.ontology_rag.retrieval.sparql_search import SPARQLSearch
from legal_rag.ontology_rag.retrieval.temporal_resolver import TemporalResolver
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings


@dataclass(frozen=True, slots=True)
class OntologyRAGResult:
    """Container for ontology-enhanced RAG result."""

    query: str
    answer: str
    rules: tuple[PenaltyRule, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()


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

        ontology_index = OntologyIndex(ontology_paths)
        self._chunk_index = ChunkIndex(ontology_index.graph)
        self._temporal_index = TemporalIndex(ontology_index.graph)

        self._retriever = Retriever(
            self._build_encoder(naive_rag_settings),
            self._build_milvus_client(milvus_settings),
            naive_rag_settings,
        )
        self._generation_client = BamlGenerationClient(settings, naive_rag_settings)
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
    ) -> OntologyRAGResult:
        """Execute the full ontology-enhanced RAG pipeline."""
        k = top_k or self._naive_rag_settings.top_k
        target_date = query_date or datetime.now(UTC).date()

        semantic_chunks = self._retriever.retrieve(query, top_k=k, mode=SearchMode.SEMANTIC)
        keyword_chunks = self._retriever.retrieve(query, top_k=k, mode=SearchMode.KEYWORD)
        ontology_instances = self._sparql_search.search([query])

        linked_instances = self._linker.link([*semantic_chunks, *keyword_chunks])
        ranked_instances = self._rrf_fusion.fuse(semantic_chunks, keyword_chunks, ontology_instances)

        all_instances = list(set(linked_instances) | set(ranked_instances))
        rules = self._instance_retriever.retrieve_batch([inst.uri for inst in all_instances[:k]])

        valid_rules = self._temporal_resolver.filter_valid(rules, target_date)
        self._context_builder.build(valid_rules, [*semantic_chunks, *keyword_chunks])
        answer = self._generate_answer(query, valid_rules)

        logger.info(
            "Pipeline completed: {} rules, {} chunks", len(valid_rules), len(semantic_chunks) + len(keyword_chunks)
        )
        return OntologyRAGResult(
            query=query,
            answer=answer,
            rules=tuple(valid_rules),
            sources=tuple(semantic_chunks[:5] + keyword_chunks[:5]),
        )

    def _generate_answer(self, query: str, rules: list[PenaltyRule]) -> str:
        """Generate answer from rules context."""
        chunk_contexts = [
            RetrievedChunk(
                chunk_id=rule.uri.fragment or str(rule.uri),
                text=f"{rule.conduct_label_vi}: {self._format_consequences(rule)}",
                source_name=rule.provision_path,
                kind="ontology",
                score=1.0,
            )
            for rule in rules[:5]
        ]
        return generate_answer(query, chunk_contexts, self._generation_client)

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
