"""Ontology-enhanced RAG pipeline combining vector, keyword, and ontology retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from loguru import logger

from legal_rag.datastores.milvus import MilvusCorpusFingerprint
from legal_rag.naive_rag.generator import BamlGenerationClient, generate_answer
from legal_rag.naive_rag.indexer import load_chunks
from legal_rag.naive_rag.models import RetrievedChunk, SearchMode
from legal_rag.naive_rag.retriever import Retriever
from legal_rag.ontology_rag.context.context_builder import ContextBuilder
from legal_rag.ontology_rag.fusion.rrf_fusion import RRFFusion
from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.index.ontology_index import OntologyIndex
from legal_rag.ontology_rag.index.temporal_index import TemporalIndex
from legal_rag.ontology_rag.models import OntologyInstance, PenaltyRule
from legal_rag.ontology_rag.query_planner import (
    BamlOntologyQueryPlanner,
    OntologyQueryPlan,
    OntologyQueryPlanner,
    TemporalQueryMode,
)
from legal_rag.ontology_rag.retrieval.instance_retriever import InstanceRetriever
from legal_rag.ontology_rag.retrieval.linker import ChunkLinker
from legal_rag.ontology_rag.retrieval.sparql_search import SPARQLSearch
from legal_rag.ontology_rag.retrieval.temporal_resolver import TemporalResolver
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings

_ONTOLOGY_PROVENANCE_FINGERPRINT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class OntologyCorpusFingerprint:
    """Verified identity of the active corpus and ontology provenance links."""

    milvus: MilvusCorpusFingerprint
    ontology_provenance_fingerprint_version: str
    ontology_provenance_chunk_count: int
    ontology_provenance_chunk_ids_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a stable manifest-ready representation."""
        return {
            **self.milvus.to_dict(),
            "ontology_provenance_fingerprint_version": self.ontology_provenance_fingerprint_version,
            "ontology_provenance_chunk_count": self.ontology_provenance_chunk_count,
            "ontology_provenance_chunk_ids_sha256": self.ontology_provenance_chunk_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class OntologyRAGResult:
    """Container for ontology-enhanced RAG result."""

    query: str
    rewritten_query: str | None = None
    answer: str = ""
    rules: tuple[PenaltyRule, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    answer_contexts: tuple[RetrievedChunk, ...] = ()
    temporal_mode: str = TemporalQueryMode.CURRENT.value
    query_dates: tuple[date, ...] = ()
    temporal_status: dict[str, str] = field(default_factory=dict)
    candidate_count: int = 0
    invalid_rule_count: int = 0
    indeterminate_rule_count: int = 0


class OntologyRAGPipeline:
    """End-to-end ontology-enhanced RAG pipeline."""

    def __init__(
        self,
        settings: Settings,
        milvus_settings: MilvusSettings,
        naive_rag_settings: NaiveRagSettings,
        ontology_paths: list[Path],
        *,
        query_planner: OntologyQueryPlanner | None = None,
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
        self._query_planner = query_planner or BamlOntologyQueryPlanner(settings)
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
        query_date: date | datetime | None = None,
        query_dates: Sequence[date | datetime] = (),
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
        rewrite_query: bool = True,
        reference_date: date | None = None,
    ) -> OntologyRAGResult:
        """Execute the full ontology-enhanced RAG pipeline.

        Args:
            query: The user's raw question, possibly context-dependent.
            query_date: Target date for temporal filtering.
            query_dates: Authoritative dates for a multi-version evaluation. Mutually exclusive
                with ``query_date``.
            top_k: Maximum number of results.
            history: Sequence of (question, answer) pairs from earlier turns.
            rewrite_query: Whether to rewrite the query before retrieval.
            reference_date: Fixed current date for reproducible temporal planning.

        Returns:
            OntologyRAGResult with answer and supporting evidence.
        """
        if query_date is not None and query_dates:
            msg = "query_date and query_dates are mutually exclusive"
            raise ValueError(msg)
        if any(not isinstance(item, date) for item in query_dates):
            msg = "query_dates must contain date values"
            raise TypeError(msg)
        explicit_query_dates = tuple(item.date() if isinstance(item, datetime) else item for item in query_dates)
        if len(set(explicit_query_dates)) != len(explicit_query_dates):
            msg = "query_dates must not contain duplicates"
            raise ValueError(msg)
        k = top_k or self._naive_rag_settings.top_k
        current_date = reference_date or datetime.now(UTC).date()
        plan = self._plan_query(
            query,
            tuple(history),
            current_date,
            query_date,
            explicit_query_dates,
        )
        retrieval_query = plan.search_query if rewrite_query else query
        if retrieval_query != query:
            logger.debug("Context-dependent ontology query rewritten")

        candidate_limit = k * self._naive_rag_settings.hybrid_pool_multiplier
        semantic_chunks = self._retriever.retrieve(
            retrieval_query,
            top_k=candidate_limit,
            mode=SearchMode.SEMANTIC,
        )
        keyword_chunks = self._retriever.retrieve(
            retrieval_query,
            top_k=candidate_limit,
            mode=SearchMode.KEYWORD,
        )
        ontology_instances = self._sparql_search.search([retrieval_query])
        semantic_instances = self._linker.link(semantic_chunks)
        keyword_instances = self._linker.link(keyword_chunks)
        ranked_instances = self._rrf_fusion.fuse(
            semantic_chunks,
            keyword_chunks,
            ontology_instances,
            semantic_instances=semantic_instances,
            keyword_instances=keyword_instances,
        )
        rules = self._instance_retriever.retrieve_batch(
            [instance.uri for instance in ranked_instances[:candidate_limit]]
        )
        selected_rules, temporal_status, invalid_count, indeterminate_count = self._select_temporal_rules(
            rules,
            plan.query_dates,
            k,
        )

        source_chunk_ids = tuple(
            dict.fromkeys(chunk_id for rule in selected_rules for chunk_id in rule.source_chunk_ids)
        )
        source_chunks = self._fetch_source_chunks(source_chunk_ids)
        fallback_chunks = self._filter_fallback_chunks(
            [*semantic_chunks[:k], *keyword_chunks[:k]],
            ranked_instances,
            selected_rules,
        )
        context_chunks = [*source_chunks, *fallback_chunks]
        answer_contexts = self._context_builder.build_answer_contexts(
            selected_rules,
            context_chunks,
            temporal_status,
            max_characters=self._naive_rag_settings.answer_max_context_characters,
        )
        answer = generate_answer(retrieval_query, answer_contexts, self._generation_client)

        logger.info(
            "Pipeline completed: {} selected rules from {} candidates, {} source chunks, {} indeterminate",
            len(selected_rules),
            len(ranked_instances),
            len(source_chunks),
            indeterminate_count,
        )
        return OntologyRAGResult(
            query=query,
            rewritten_query=retrieval_query if retrieval_query != query else None,
            answer=answer,
            rules=tuple(selected_rules),
            sources=tuple(self._unique_chunks(context_chunks)),
            answer_contexts=tuple(answer_contexts),
            temporal_mode=plan.temporal_mode.value,
            query_dates=plan.query_dates,
            temporal_status=temporal_status,
            candidate_count=len(ranked_instances),
            invalid_rule_count=invalid_count,
            indeterminate_rule_count=indeterminate_count,
        )

    def _plan_query(
        self,
        query: str,
        history: tuple[tuple[str, str], ...],
        reference_date: date,
        explicit_query_date: date | datetime | None,
        explicit_query_dates: tuple[date, ...],
    ) -> OntologyQueryPlan:
        """Plan retrieval once and let explicit API dates override inferred temporal intent."""
        plan = self._query_planner.plan(query, history, reference_date)
        if explicit_query_dates:
            return OntologyQueryPlan(
                search_query=plan.search_query,
                temporal_mode=(
                    TemporalQueryMode.AT_DATE if len(explicit_query_dates) == 1 else TemporalQueryMode.COMPARE_DATES
                ),
                query_dates=explicit_query_dates,
            )
        if explicit_query_date is None:
            if plan.query_dates:
                return plan
            return OntologyQueryPlan(
                search_query=plan.search_query,
                temporal_mode=plan.temporal_mode,
                query_dates=(reference_date,),
            )
        target_date = explicit_query_date.date() if isinstance(explicit_query_date, datetime) else explicit_query_date
        return OntologyQueryPlan(
            search_query=plan.search_query,
            temporal_mode=TemporalQueryMode.AT_DATE,
            query_dates=(target_date,),
        )

    def _select_temporal_rules(
        self,
        rules: Sequence[PenaltyRule],
        query_dates: tuple[date, ...],
        limit: int,
    ) -> tuple[list[PenaltyRule], dict[str, str], int, int]:
        """Resolve every requested date before limiting the final ranked rules."""
        status_by_uri: dict[str, list[tuple[date, str]]] = {str(rule.uri): [] for rule in rules}
        preferred_by_date: list[list[PenaltyRule]] = []
        for query_date in query_dates:
            resolution = self._temporal_resolver.partition(rules, query_date)
            preferred_by_date.append([*resolution.valid, *resolution.indeterminate])
            for status, resolved_rules in (
                ("valid", resolution.valid),
                ("invalid", resolution.invalid),
                ("indeterminate", resolution.indeterminate),
            ):
                for rule in resolved_rules:
                    status_by_uri[str(rule.uri)].append((query_date, status))

        overall_status: dict[str, str] = {}
        context_status: dict[str, str] = {}
        for rule_uri, dated_statuses in status_by_uri.items():
            statuses = {status for _, status in dated_statuses}
            overall_status[rule_uri] = (
                "valid" if "valid" in statuses else "indeterminate" if "indeterminate" in statuses else "invalid"
            )
            context_status[rule_uri] = (
                dated_statuses[0][1]
                if len(query_dates) == 1
                else ";".join(f"{query_date.isoformat()}={status}" for query_date, status in dated_statuses)
            )

        definitely_valid = [rule for rule in rules if overall_status[str(rule.uri)] == "valid"]
        indeterminate = [rule for rule in rules if overall_status[str(rule.uri)] == "indeterminate"]
        selected = self._round_robin_unique(preferred_by_date, limit)
        selected_uris = {rule.uri for rule in selected}
        for rule in (*definitely_valid, *indeterminate):
            if len(selected) == limit:
                break
            if rule.uri in selected_uris:
                continue
            selected.append(rule)
            selected_uris.add(rule.uri)
        selected_status = {str(rule.uri): context_status[str(rule.uri)] for rule in selected}
        invalid_count = sum(status == "invalid" for status in overall_status.values())
        return selected, selected_status, invalid_count, len(indeterminate)

    @staticmethod
    def _round_robin_unique(groups: Sequence[Sequence[PenaltyRule]], limit: int) -> list[PenaltyRule]:
        """Give each requested date a chance to contribute a ranked rule."""
        selected: list[PenaltyRule] = []
        selected_uris: set[object] = set()
        offsets = [0] * len(groups)
        while len(selected) < limit:
            added = False
            for index, group in enumerate(groups):
                while offsets[index] < len(group) and group[offsets[index]].uri in selected_uris:
                    offsets[index] += 1
                if offsets[index] >= len(group):
                    continue
                rule = group[offsets[index]]
                offsets[index] += 1
                selected.append(rule)
                selected_uris.add(rule.uri)
                added = True
                if len(selected) == limit:
                    break
            if not added:
                break
        return selected

    @staticmethod
    def _filter_fallback_chunks(
        chunks: Sequence[RetrievedChunk],
        candidate_instances: Sequence[OntologyInstance],
        selected_rules: Sequence[PenaltyRule],
    ) -> list[RetrievedChunk]:
        """Exclude fallback chunks that evidence any unselected candidate rule."""
        selected_uris = {rule.uri for rule in selected_rules}
        excluded_chunk_ids = {
            chunk_id
            for instance in candidate_instances
            if instance.uri not in selected_uris
            for chunk_id in instance.source_chunk_ids
        }
        filtered = [chunk for chunk in chunks if excluded_chunk_ids.isdisjoint(chunk.effective_evidence_chunk_ids)]
        omitted_count = len(chunks) - len(filtered)
        if omitted_count:
            logger.info("Excluded {} fallback chunks belonging to unselected candidate rules", omitted_count)
        return filtered

    @staticmethod
    def _unique_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Deduplicate result sources without changing their first-seen order."""
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique.append(chunk)
        return unique

    def _fetch_source_chunks(self, source_chunk_ids: tuple[str, ...]) -> list[RetrievedChunk]:
        """Fetch every selected rule's provenance chunk or fail the prediction."""
        source_chunks = self._retriever.fetch_by_ids(source_chunk_ids)
        returned_ids = {chunk.chunk_id for chunk in source_chunks}
        missing_ids = [chunk_id for chunk_id in source_chunk_ids if chunk_id not in returned_ids]
        if missing_ids:
            msg = (
                "Selected ontology rules reference chunks missing from the active corpus; "
                f"count={len(missing_ids)}, missing={missing_ids[:5]}"
            )
            raise ValueError(msg)
        return source_chunks

    def preflight_corpus(self) -> OntologyCorpusFingerprint:
        """Verify active Milvus rows and every ontology provenance chunk against local data."""
        milvus_fingerprint = self._retriever.preflight_corpus()
        local_chunk_ids = {chunk.chunk_id for chunk in load_chunks(self._retriever.settings.chunks_dir)}
        ontology_chunk_ids = self._chunk_index.all_chunk_ids()
        missing_ids = sorted(ontology_chunk_ids - local_chunk_ids)
        if missing_ids:
            msg = (
                "Ontology provenance references chunks absent from the configured local corpus; "
                f"count={len(missing_ids)}, missing={missing_ids[:5]}"
            )
            raise ValueError(msg)
        encoded_ids = json.dumps(sorted(ontology_chunk_ids), ensure_ascii=False, separators=(",", ":")).encode()
        return OntologyCorpusFingerprint(
            milvus=milvus_fingerprint,
            ontology_provenance_fingerprint_version=_ONTOLOGY_PROVENANCE_FINGERPRINT_VERSION,
            ontology_provenance_chunk_count=len(ontology_chunk_ids),
            ontology_provenance_chunk_ids_sha256=hashlib.sha256(encoded_ids).hexdigest(),
        )

    def close(self) -> None:
        """Close the pipeline connections."""
        self._retriever.close()
