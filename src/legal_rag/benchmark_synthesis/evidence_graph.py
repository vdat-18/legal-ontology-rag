"""Confirm candidate evidence relations and sample bundles for scenario generation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal, Protocol
from unicodedata import normalize

import numpy as np
from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ConfirmedRelation as BamlConfirmedRelation
from legal_rag.baml_client.types import EvidenceRelationBatchItem, LegalFactSummary
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import EvidenceRetrievalSettings, SynthesisSettings
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.vietnamese_lexical import (
    VIETNAMESE_LEXICAL_TOKENIZER_VERSION,
    VietnameseWordSegmenter,
)
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

LEGACY_EVIDENCE_RELATION_SCHEMA_VERSION = "3.0.0"
EVIDENCE_RELATION_SCHEMA_VERSION = "3.1.0"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "3.0.0"
LEGACY_EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION = "1.1.0"
EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION = "1.2.0"
LEGACY_EVIDENCE_RELATION_PROMPT_VERSION = "5.0.0"
EVIDENCE_RELATION_PROMPT_VERSION = "5.1.0"
EVIDENCE_CANDIDATE_SELECTION_VERSION = "4.1.0"
EVIDENCE_CANDIDATE_SET_SCHEMA_VERSION = "1.0.0"
FACT_RETRIEVAL_TEXT_VERSION = "4.0.0"

BundleKind = Literal["single_hop", "multi_hop", "temporal"]
RelationBundleKind = Literal["multi_hop", "temporal"]

_BUNDLE_KIND_BY_BAML_ENUM: dict[str, RelationBundleKind] = {
    "MultiHop": "multi_hop",
    "Temporal": "temporal",
}


@dataclass(frozen=True, slots=True)
class EvidenceRelation:
    """One confirmed relation between two legal facts."""

    relation_id: str
    fact_ids: tuple[str, str]
    bundle_kind: RelationBundleKind
    bridge: str
    confirmed_by_model: str
    run_id: str
    prompt_version: str = EVIDENCE_RELATION_PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-relation record."""
        return {
            "schema_version": EVIDENCE_RELATION_SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "fact_ids": list(self.fact_ids),
            "bundle_kind": self.bundle_kind,
            "bridge": self.bridge,
            "confirmed_by_model": self.confirmed_by_model,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """One group of facts sampled as evidence for a scenario."""

    bundle_id: str
    kind: BundleKind
    fact_ids: tuple[str, ...]
    bridge: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-bundle record."""
        return {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "kind": self.kind,
            "fact_ids": list(self.fact_ids),
            "bridge": self.bridge,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRelationProgressRecord:
    """One candidate fact pair evidence-relation confirmation has finished checking, related or not."""

    fact_id_a: str
    fact_id_b: str
    run_id: str
    prompt_version: str = EVIDENCE_RELATION_PROMPT_VERSION
    confirmed_by_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-relation-progress record."""
        return {
            "schema_version": EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
            "fact_id_a": self.fact_id_a,
            "fact_id_b": self.fact_id_b,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "confirmed_by_model": self.confirmed_by_model,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCandidateSet:
    """One fingerprinted candidate-pair set reusable across relation-confirmation resumes."""

    run_id: str
    pairs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-candidate-set record."""
        return {
            "schema_version": EVIDENCE_CANDIDATE_SET_SCHEMA_VERSION,
            "run_id": self.run_id,
            "pairs": [list(pair) for pair in self.pairs],
        }


def _fact_embedding_text(fact: LegalFact) -> str:
    """Return the legal statement used by both retrieval legs."""
    return fact.statement


def _lexical_units(text: str) -> list[str]:
    """Split normalized text on Unicode punctuation and separators while preserving letters and digits."""
    return "".join(character if character.isalnum() else " " for character in text).split()


def _vietnamese_legal_tokens(segmented_tokens: Sequence[str]) -> list[str]:
    """Return namespaced VnCoreNLP words and adjacent word bigrams for Vietnamese legal BM25."""
    word_tokens: list[str] = []
    bigram_tokens: list[str] = []
    previous_word: str | None = None
    for segmented_token in segmented_tokens:
        units = _lexical_units(normalize("NFC", segmented_token).casefold())
        if not units:
            previous_word = None
            continue
        word = "_".join(units)
        word_tokens.append(f"w:{word}")
        if previous_word is not None:
            bigram_tokens.append(f"b:{previous_word}|{word}")
        previous_word = word
    return [*word_tokens, *bigram_tokens]


def _lexical_corpus(texts: Sequence[str], word_segmenter: VietnameseWordSegmenter) -> list[list[str]]:
    """Build the Vietnamese BM25 corpus from complete legal statements."""
    segmented = word_segmenter.segment(texts)
    if len(segmented) != len(texts):
        msg = f"Word segmenter returned {len(segmented)} outputs for {len(texts)} statements"
        raise ValueError(msg)
    return [_vietnamese_legal_tokens(tokens) for tokens in segmented]


class _FactEncoder(Protocol):
    """Embedding behavior required by evidence candidate retrieval."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized vector per input text."""


class _Bm25Scorer(Protocol):
    """BM25 scoring behavior required by lexical candidate retrieval."""

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        """Return one lexical score per corpus item."""


def _top_indices(scored: Sequence[tuple[int, float]], fact_ids: Sequence[str], limit: int) -> list[int]:
    """Return indices ordered by descending score then fact ID, capped at limit."""
    return [index for index, _ in sorted(scored, key=lambda item: (-item[1], fact_ids[item[0]]))[:limit]]


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], rrf_k: int) -> dict[int, float]:
    """Fuse equally weighted rankings with standard reciprocal rank fusion."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_index in enumerate(ranking, start=1):
            fused[item_index] = fused.get(item_index, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def _dense_rankings(
    vectors: np.ndarray,
    fact_ids: Sequence[str],
    pool_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Rank dense neighbors for every fact in corpus order."""
    return tuple(
        tuple(
            _top_indices(
                [(other, float(score)) for other, score in enumerate(np.dot(vectors, vector)) if other != index],
                fact_ids,
                pool_size,
            )
        )
        for index, vector in enumerate(vectors)
    )


def _lexical_rankings(
    bm25: _Bm25Scorer,
    lexical_corpus: Sequence[Sequence[str]],
    fact_ids: Sequence[str],
    pool_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Rank positive-score BM25L neighbors for every fact in corpus order."""
    rankings: list[tuple[int, ...]] = []
    for index, query in enumerate(lexical_corpus):
        scores = bm25.get_scores(query)
        rankings.append(
            tuple(
                _top_indices(
                    [
                        (other, float(score))
                        for other, score in enumerate(scores)
                        if other != index and float(score) > 0.0
                    ],
                    fact_ids,
                    pool_size,
                )
            )
        )
    return tuple(rankings)


def build_evidence_candidate_run_id(
    facts: Sequence[LegalFact],
    profile: EmbeddingProfile,
    synthesis_settings: SynthesisSettings,
    *,
    embedding_base_url: str,
    embedding_model: str,
) -> str:
    """Hash every input that can change candidate-pair selection."""
    return sha256_json(
        {
            "candidate_selection": {
                "version": EVIDENCE_CANDIDATE_SELECTION_VERSION,
                "fact_retrieval_text": FACT_RETRIEVAL_TEXT_VERSION,
                "retrieval": asdict(synthesis_settings.evidence_retrieval),
                "lexical_tokenizer": VIETNAMESE_LEXICAL_TOKENIZER_VERSION,
            },
            "embedding": {
                "base_url": embedding_base_url,
                "model": embedding_model,
                "batch_size": synthesis_settings.embedding_batch_size,
                "profile": asdict(profile),
            },
            "facts": sorted(fact.fact_id for fact in facts),
        }
    )


def build_evidence_relation_run_id(
    facts: Sequence[LegalFact],
    profile: EmbeddingProfile,
    synthesis_settings: SynthesisSettings,
    *,
    embedding_base_url: str,
    embedding_model: str,
    llm_base_url: str,
    llm_model: str,
) -> str:
    """Hash every input that can change relation confirmation or its candidate set."""
    return sha256_json(
        {
            "candidate_run_id": build_evidence_candidate_run_id(
                facts,
                profile,
                synthesis_settings,
                embedding_base_url=embedding_base_url,
                embedding_model=embedding_model,
            ),
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "prompt_version": EVIDENCE_RELATION_PROMPT_VERSION,
            "relation_generation": asdict(synthesis_settings.evidence_relation),
            "batch_size": synthesis_settings.evidence_relation_batch_size,
        }
    )


def find_candidate_fact_pairs(
    facts: Sequence[LegalFact],
    encoder: _FactEncoder,
    word_segmenter: VietnameseWordSegmenter,
    settings: EvidenceRetrievalSettings,
) -> tuple[tuple[str, str], ...]:
    """Return candidate fact pairs from statement-only dense and BM25L retrieval with RRF."""
    if len(facts) < 2:
        return ()

    # Keep the optional implementation detail local so importing the CLI remains lightweight.
    from rank_bm25 import BM25L

    fact_texts = [_fact_embedding_text(fact) for fact in facts]
    vectors = encoder.encode(fact_texts)
    lexical_corpus = _lexical_corpus(fact_texts, word_segmenter)
    fact_ids = [fact.fact_id for fact in facts]
    bm25 = BM25L(
        lexical_corpus,
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        delta=settings.bm25_delta,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            _dense_rankings,
            vectors,
            fact_ids,
            settings.retrieval_pool_size,
        )
        lexical_future = executor.submit(
            _lexical_rankings,
            bm25,
            lexical_corpus,
            fact_ids,
            settings.retrieval_pool_size,
        )
        dense_rankings = dense_future.result()
        lexical_rankings = lexical_future.result()

    pairs: set[tuple[str, str]] = set()
    for index, fact in enumerate(facts):
        fused = _reciprocal_rank_fusion(
            (dense_rankings[index], lexical_rankings[index]),
            settings.rrf_k,
        )

        best = sorted(fused, key=lambda other_index: (-fused[other_index], fact_ids[other_index]))
        for other_index in best[: settings.max_candidates_per_fact]:
            pairs.add(tuple(sorted((fact.fact_id, facts[other_index].fact_id))))
    return tuple(sorted(pairs))


class EvidenceRelationClient(Protocol):
    """Boundary implemented by an evidence-relation-confirmation language-model client."""

    def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> Sequence[EvidenceRelation | None]:
        """Return one confirmed relation or None per pair, in the same order as the input pairs."""


class BamlEvidenceRelationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings, *, run_id: str) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.evidence_relation.max_output_tokens,
            temperature=synthesis_settings.evidence_relation.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.evidence_relation.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes
        self._model = settings.llm_model
        self._run_id = run_id

    def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> Sequence[EvidenceRelation | None]:
        """Run one bounded BAML call over a batch of fact pairs and translate its typed results, in order."""
        items = [
            EvidenceRelationBatchItem(
                pair_id=f"{fact_a.fact_id}::{fact_b.fact_id}",
                fact_a=_summarize(fact_a),
                fact_b=_summarize(fact_b),
            )
            for fact_a, fact_b in pairs
        ]
        request = b.request.ConfirmEvidenceRelationBatch(items, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        results = b.parse.ConfirmEvidenceRelationBatch(raw_output, baml_options={"client_registry": self._registry})
        requested_ids = [item.pair_id for item in items]
        result_ids = [result.pair_id for result in results]
        if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
            msg = "Evidence relation batch response must contain exactly one result for every requested pair"
            raise ValueError(msg)
        results_by_pair_id = {result.pair_id: result for result in results}
        return [
            _build_relation(results_by_pair_id[item.pair_id].relation, fact_a, fact_b, self._model, self._run_id)
            for item, (fact_a, fact_b) in zip(items, pairs, strict=True)
        ]


def _summarize(fact: LegalFact) -> LegalFactSummary:
    """Return the BAML-facing summary of a fact used for relation confirmation."""
    return LegalFactSummary(
        fact_id=fact.fact_id,
        statement=fact.statement,
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
    )


def _build_relation(
    result: BamlConfirmedRelation,
    fact_a: LegalFact,
    fact_b: LegalFact,
    model: str,
    run_id: str,
) -> EvidenceRelation | None:
    """Translate one BAML confirmation result into a persisted relation, or None."""
    if not result.is_related:
        return None
    if result.bundle_kind is None or result.bridge is None:
        msg = "A related evidence pair must declare its bundle kind and bridge"
        raise ValueError(msg)
    fact_ids = tuple(sorted((fact_a.fact_id, fact_b.fact_id)))
    bundle_kind = _BUNDLE_KIND_BY_BAML_ENUM[result.bundle_kind.value]
    identity = {
        "fact_ids": list(fact_ids),
        "bundle_kind": bundle_kind,
        "bridge": result.bridge,
        "run_id": run_id,
    }
    return EvidenceRelation(
        relation_id=sha256_json(identity),
        fact_ids=fact_ids,
        bundle_kind=bundle_kind,
        bridge=result.bridge,
        confirmed_by_model=model,
        run_id=run_id,
        prompt_version=EVIDENCE_RELATION_PROMPT_VERSION,
    )


def _batch_task_key(pairs: Sequence[tuple[str, str]]) -> str:
    """Build a deterministic retry key from the ordered fact-id pairs in one batch."""
    pair_ids = "\0".join(f"{fact_id_a}:{fact_id_b}" for fact_id_a, fact_id_b in pairs)
    return f"relation-batch:{hashlib.sha256(pair_ids.encode('utf-8')).hexdigest()}"


def _confirm_batch(
    client: EvidenceRelationClient,
    facts_by_id: Mapping[str, LegalFact],
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[tuple[str, str], EvidenceRelation | None], ...]:
    """Confirm one batch and pair each result back to its candidate pair, in order."""
    relations = client.confirm_batch(
        [(facts_by_id[fact_id_a], facts_by_id[fact_id_b]) for fact_id_a, fact_id_b in pairs]
    )
    if len(relations) != len(pairs):
        msg = f"Evidence relation batch response must contain exactly one result per pair, got {len(relations)} for {len(pairs)}"
        raise ValueError(msg)
    return tuple(zip(pairs, relations, strict=True))


def confirm_evidence_relations(
    facts: Sequence[LegalFact],
    candidate_pairs: Sequence[tuple[str, str]],
    client: EvidenceRelationClient,
    *,
    batch_size: int,
    max_concurrency: int,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_confirmed: Callable[[Sequence[tuple[tuple[str, str], EvidenceRelation | None]]], None] | None = None,
) -> tuple[tuple[EvidenceRelation, ...], tuple[LlmTaskFailure, ...]]:
    """Confirm candidate pairs concurrently and checkpoint each completed batch.

    Batches are submitted to a bounded worker pool but their results are read and persisted
    on this thread, so on_batch_confirmed never runs concurrently with itself.
    """
    facts_by_id = {fact.fact_id: fact for fact in facts}
    batches = [candidate_pairs[start : start + batch_size] for start in range(0, len(candidate_pairs), batch_size)]
    relations: list[EvidenceRelation] = []
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                execute_llm_task,
                _batch_task_key(batch),
                partial(_confirm_batch, client, facts_by_id, batch),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_results = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=_batch_task_key(batch), error=str(error)))
                continue
            for _pair, relation in batch_results:
                if relation is not None:
                    relations.append(relation)
            if on_batch_confirmed is not None:
                on_batch_confirmed(batch_results)
    return tuple(relations), tuple(failures)


def evidence_relation_sort_key(relation: EvidenceRelation) -> tuple[object, ...]:
    """Return a semantic ordering key independent of run-specific relation identity."""
    return (
        relation.fact_ids,
        relation.bundle_kind,
        relation.bridge,
    )


def sample_bundles(
    facts: Sequence[LegalFact],
    relations: Sequence[EvidenceRelation],
    *,
    overlap_threshold: float,
) -> tuple[EvidenceBundle, ...]:
    """Sample single-hop and relation-derived bundles, bounding overlap between them.

    Every legal fact contributes exactly one single-hop bundle; those are
    excluded from the overlap check since, by construction, every other
    bundle overlaps with them.
    """
    single_hop_bundles = tuple(
        EvidenceBundle(
            bundle_id=sha256_json({"fact_ids": [fact.fact_id]}),
            kind="single_hop",
            fact_ids=(fact.fact_id,),
            bridge=None,
        )
        for fact in sorted(facts, key=lambda item: item.fact_id)
    )

    selected: list[EvidenceBundle] = list(single_hop_bundles)
    for relation in sorted(relations, key=evidence_relation_sort_key):
        candidate = EvidenceBundle(
            bundle_id=sha256_json(
                {
                    "bridge": relation.bridge,
                    "fact_ids": list(relation.fact_ids),
                    "kind": relation.bundle_kind,
                }
            ),
            kind=relation.bundle_kind,
            fact_ids=relation.fact_ids,
            bridge=relation.bridge,
        )
        if not _exceeds_overlap(candidate, selected, threshold=overlap_threshold):
            selected.append(candidate)
    return tuple(selected)


def _exceeds_overlap(candidate: EvidenceBundle, selected: Sequence[EvidenceBundle], *, threshold: float) -> bool:
    """Return whether candidate overlaps an already-selected multi-fact bundle beyond threshold."""
    candidate_facts = set(candidate.fact_ids)
    for other in selected:
        if other.kind == "single_hop":
            continue
        overlap_ratio = len(candidate_facts & set(other.fact_ids)) / len(candidate_facts)
        if overlap_ratio > threshold:
            return True
    return False
