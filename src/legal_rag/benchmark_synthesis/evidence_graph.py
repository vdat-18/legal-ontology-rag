"""Confirm candidate evidence relations and sample bundles for scenario generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol
from unicodedata import normalize

import numpy as np
from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ConfirmedRelation as BamlConfirmedRelation
from legal_rag.baml_client.types import LegalFactSummary, RelationEvidence
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

EVIDENCE_RELATION_SCHEMA_VERSION = "1.1.0"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "1.0.0"
EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION = "1.1.0"
EVIDENCE_RELATION_PROMPT_VERSION = "1.1.0"
EVIDENCE_CANDIDATE_SELECTION_VERSION = "2.0.0"
EVIDENCE_CANDIDATE_SET_SCHEMA_VERSION = "1.0.0"
VIETNAMESE_LEXICAL_TOKENIZER_VERSION = "underthesea_hybrid_v1"

RelationType = Literal[
    "conduct_to_penalty",
    "conduct_to_subject",
    "conduct_to_points_deduction",
    "conduct_to_remedial_measure",
    "rule_to_exception",
    "rule_to_authority",
    "amended_by",
    "procedure_sequence",
]

BundleKind = Literal["single_hop", "synthesis", "multi_hop", "temporal"]

_RELATION_TYPE_BY_BAML_ENUM: dict[str, RelationType] = {
    "ConductToPenalty": "conduct_to_penalty",
    "ConductToSubject": "conduct_to_subject",
    "ConductToPointsDeduction": "conduct_to_points_deduction",
    "ConductToRemedialMeasure": "conduct_to_remedial_measure",
    "RuleToException": "rule_to_exception",
    "RuleToAuthority": "rule_to_authority",
    "AmendedBy": "amended_by",
    "ProcedureSequence": "procedure_sequence",
}


@dataclass(frozen=True, slots=True)
class EvidenceRelation:
    """One confirmed relation between two approved facts."""

    relation_id: str
    relation_type: RelationType
    from_fact_id: str
    to_fact_id: str
    bridge: tuple[str, ...]
    confirmed_by_model: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-relation record."""
        return {
            "schema_version": EVIDENCE_RELATION_SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "from_fact_id": self.from_fact_id,
            "to_fact_id": self.to_fact_id,
            "bridge": list(self.bridge),
            "confirmed_by_model": self.confirmed_by_model,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """One group of facts sampled as evidence for a scenario."""

    bundle_id: str
    kind: BundleKind
    fact_ids: tuple[str, ...]
    bridge: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-bundle record."""
        return {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "kind": self.kind,
            "fact_ids": list(self.fact_ids),
            "bridge": list(self.bridge),
        }


@dataclass(frozen=True, slots=True)
class EvidenceRelationProgressRecord:
    """One candidate fact pair evidence-relation confirmation has finished checking, related or not."""

    fact_id_a: str
    fact_id_b: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence-relation-progress record."""
        return {
            "schema_version": EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
            "fact_id_a": self.fact_id_a,
            "fact_id_b": self.fact_id_b,
            "run_id": self.run_id,
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
    """Return a canonical text representation of a fact for similarity search."""
    return f"{fact.conduct}. {fact.subject}. {fact.provision_path}."


def _lexical_units(text: str) -> list[str]:
    """Split normalized text on Unicode punctuation and separators while preserving letters and digits."""
    return "".join(character if character.isalnum() else " " for character in text).split()


def _vietnamese_legal_tokens(text: str) -> list[str]:
    """Return namespaced syllables, segmented words, and word bigrams for Vietnamese legal BM25."""
    from underthesea import word_tokenize

    normalized_text = normalize("NFC", text).casefold()
    syllable_tokens = [f"s:{unit}" for unit in _lexical_units(normalized_text)]

    word_tokens: list[str] = []
    bigram_tokens: list[str] = []
    previous_word: str | None = None
    for segmented_token in word_tokenize(normalized_text):
        units = _lexical_units(segmented_token)
        if not units:
            previous_word = None
            continue
        word = "_".join(units)
        word_tokens.append(f"w:{word}")
        if previous_word is not None:
            bigram_tokens.append(f"b:{previous_word}|{word}")
        previous_word = word
    return [*syllable_tokens, *word_tokens, *bigram_tokens]


def _fact_lexical_tokens(fact: LegalFact) -> list[str]:
    """Return hybrid Vietnamese BM25 tokens without crossing fact-field boundaries."""
    return _vietnamese_legal_tokens(_fact_embedding_text(fact))


class _FactEncoder(Protocol):
    """Embedding behavior required by evidence candidate retrieval."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized vector per input text."""


def _top_indices(scored: Sequence[tuple[int, float]], fact_ids: Sequence[str], limit: int) -> list[int]:
    """Return indices ordered by descending score then fact ID, capped at limit."""
    return [index for index, _ in sorted(scored, key=lambda item: (-item[1], fact_ids[item[0]]))[:limit]]


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
                "max_candidates_per_fact": synthesis_settings.relation_max_candidates_per_fact,
                "min_similarity": synthesis_settings.relation_min_similarity,
                "pool_multiplier": synthesis_settings.relation_pool_multiplier,
                "rrf_k": synthesis_settings.relation_rrf_k,
                "lexical_tokenizer": VIETNAMESE_LEXICAL_TOKENIZER_VERSION,
            },
            "embedding": {
                "base_url": embedding_base_url,
                "model": embedding_model,
                "batch_size": synthesis_settings.embedding_batch_size,
                "profile": asdict(profile),
            },
            "facts": sorted((fact.fact_id, fact.fact_content_sha256) for fact in facts),
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
        }
    )


def find_candidate_fact_pairs(
    facts: Sequence[LegalFact],
    encoder: _FactEncoder,
    *,
    max_candidates_per_fact: int,
    min_similarity: float,
    pool_multiplier: int,
    rrf_k: int,
) -> tuple[tuple[str, str], ...]:
    """Return unique fact_id pairs worth confirming, fusing dense and lexical retrieval with reciprocal rank fusion.

    Dense retrieval finds facts phrased differently about the same thing; lexical
    retrieval finds near-verbatim restatements, which is how an amending
    provision quotes the provision it amends. Each leg filters on its own scale —
    fused RRF scores have no absolute meaning — and RRF merges the survivors.
    """
    if len(facts) < 2:
        return ()

    # Keep the optional implementation detail local so importing the CLI remains lightweight.
    from rank_bm25 import BM25Okapi

    vectors = encoder.encode([_fact_embedding_text(fact) for fact in facts])
    corpus = [_fact_lexical_tokens(fact) for fact in facts]
    fact_ids = [fact.fact_id for fact in facts]
    bm25 = BM25Okapi(corpus)
    pool_size = max_candidates_per_fact * pool_multiplier

    pairs: set[tuple[str, str]] = set()
    for index, fact in enumerate(facts):
        dense_scores = np.dot(vectors, vectors[index])
        dense_ranking = _top_indices(
            [
                (other, float(dense_scores[other]))
                for other in range(len(facts))
                if other != index and float(dense_scores[other]) >= min_similarity
            ],
            fact_ids,
            pool_size,
        )
        lexical_scores = bm25.get_scores(corpus[index])
        lexical_ranking = _top_indices(
            [
                (other, float(lexical_scores[other]))
                for other in range(len(facts))
                if other != index and float(lexical_scores[other]) > 0.0
            ],
            fact_ids,
            pool_size,
        )

        fused: dict[int, float] = {}
        for ranking in (dense_ranking, lexical_ranking):
            for rank, other_index in enumerate(ranking, start=1):
                fused[other_index] = fused.get(other_index, 0.0) + 1.0 / (rrf_k + rank)

        best = sorted(fused, key=lambda other_index: (-fused[other_index], fact_ids[other_index]))
        for other_index in best[:max_candidates_per_fact]:
            pairs.add(tuple(sorted((fact.fact_id, facts[other_index].fact_id))))
    return tuple(sorted(pairs))


class EvidenceRelationClient(Protocol):
    """Boundary implemented by an evidence-relation-confirmation language-model client."""

    def confirm(self, fact_a: LegalFact, fact_b: LegalFact) -> EvidenceRelation | None:
        """Return a confirmed relation, or None when the facts are not directly related."""


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
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes
        self._model = settings.llm_model
        self._run_id = run_id

    def confirm(self, fact_a: LegalFact, fact_b: LegalFact) -> EvidenceRelation | None:
        """Run one bounded BAML call and translate its typed result."""
        request = b.request.ConfirmEvidenceRelation(
            _summarize(fact_a), _summarize(fact_b), baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        result = b.parse.ConfirmEvidenceRelation(raw_output, baml_options={"client_registry": self._registry})
        return _build_relation(result, fact_a, fact_b, self._model, self._run_id)


def _summarize(fact: LegalFact) -> LegalFactSummary:
    """Return the BAML-facing summary of a fact used for relation confirmation."""
    return LegalFactSummary(
        fact_id=fact.fact_id,
        conduct=fact.conduct,
        subject=fact.subject,
        conditions=list(fact.conditions),
        monetary_penalty=fact.monetary_penalty,
        license_points_deducted=fact.license_points_deducted,
        additional_penalties=list(fact.additional_penalties),
        remedial_measures=list(fact.remedial_measures),
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
        provision_path=fact.provision_path,
        evidence=[
            RelationEvidence(chunk_id=evidence.chunk_id, quoted_span=evidence.quoted_span) for evidence in fact.evidence
        ],
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
        if result.relation_type is not None or result.from_fact_id is not None or result.to_fact_id is not None:
            msg = "An unrelated evidence pair must not declare a relation type or direction"
            raise ValueError(msg)
        if result.bridge:
            msg = "An unrelated evidence pair must not declare a bridge"
            raise ValueError(msg)
        return None
    if result.relation_type is None or result.from_fact_id is None or result.to_fact_id is None:
        msg = "A related evidence pair must declare its relation type and direction"
        raise ValueError(msg)
    fact_ids = {fact_a.fact_id, fact_b.fact_id}
    oriented_ids = {result.from_fact_id, result.to_fact_id}
    if len(oriented_ids) != 2 or oriented_ids != fact_ids:
        msg = "Evidence relation direction must contain each input fact_id exactly once"
        raise ValueError(msg)
    if not result.bridge:
        msg = "A related evidence pair must declare at least one bridge"
        raise ValueError(msg)
    relation_type = _RELATION_TYPE_BY_BAML_ENUM[result.relation_type.value]
    identity = {
        "from_fact_id": result.from_fact_id,
        "to_fact_id": result.to_fact_id,
        "relation_type": relation_type,
        "bridge": list(result.bridge),
        "run_id": run_id,
    }
    return EvidenceRelation(
        relation_id=sha256_json(identity),
        relation_type=relation_type,
        from_fact_id=result.from_fact_id,
        to_fact_id=result.to_fact_id,
        bridge=tuple(result.bridge),
        confirmed_by_model=model,
        run_id=run_id,
    )


def confirm_evidence_relations(
    facts: Sequence[LegalFact],
    candidate_pairs: Sequence[tuple[str, str]],
    client: EvidenceRelationClient,
    *,
    max_retries: int,
    retry_delay_seconds: float,
    on_pair_confirmed: Callable[[tuple[str, str], EvidenceRelation | None], None] | None = None,
) -> tuple[EvidenceRelation, ...]:
    """Confirm every candidate pair, retrying a failed call before skipping it; call on_pair_confirmed after each pair."""
    facts_by_id = {fact.fact_id: fact for fact in facts}
    relations: list[EvidenceRelation] = []
    failures: list[LlmTaskFailure] = []
    for fact_id_a, fact_id_b in candidate_pairs:
        task_key = f"{fact_id_a}:{fact_id_b}"
        try:
            relation = execute_llm_task(
                task_key,
                lambda fact_id_a=fact_id_a, fact_id_b=fact_id_b: client.confirm(
                    facts_by_id[fact_id_a], facts_by_id[fact_id_b]
                ),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        except LlmTaskExhaustedError as error:
            failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
        else:
            if relation is not None:
                relations.append(relation)
            if on_pair_confirmed is not None:
                on_pair_confirmed((fact_id_a, fact_id_b), relation)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("evidence relation confirmation", failures)
        raise incomplete_error
    return tuple(relations)


def classify_bundle_kind(relation: EvidenceRelation, facts_by_id: Mapping[str, LegalFact]) -> BundleKind:
    """Classify a relation's bundle kind: temporal, multi-hop, or synthesis.

    A relation whose two facts come from different documents is multi-hop
    unless it is an amendment relation, which is always temporal regardless
    of document — this is the structural downgrade rule from the pipeline
    design, expressed directly rather than as a post-hoc correction.
    """
    if relation.relation_type == "amended_by":
        return "temporal"
    from_document = facts_by_id[relation.from_fact_id].document_number
    to_document = facts_by_id[relation.to_fact_id].document_number
    return "multi_hop" if from_document != to_document else "synthesis"


def evidence_relation_sort_key(relation: EvidenceRelation) -> tuple[object, ...]:
    """Return a semantic ordering key independent of run-specific relation identity."""
    return (
        tuple(sorted((relation.from_fact_id, relation.to_fact_id))),
        relation.relation_type,
        relation.from_fact_id,
        relation.to_fact_id,
        relation.bridge,
    )


def sample_bundles(
    facts: Sequence[LegalFact],
    relations: Sequence[EvidenceRelation],
    *,
    overlap_threshold: float,
) -> tuple[EvidenceBundle, ...]:
    """Sample single-hop and relation-derived bundles, bounding overlap between them.

    Every approved fact contributes exactly one single-hop bundle; those are
    excluded from the overlap check since, by construction, every other
    bundle overlaps with them.
    """
    facts_by_id = {fact.fact_id: fact for fact in facts}
    single_hop_bundles = tuple(
        EvidenceBundle(
            bundle_id=sha256_json({"fact_ids": [fact.fact_id]}),
            kind="single_hop",
            fact_ids=(fact.fact_id,),
            bridge=(),
        )
        for fact in sorted(facts, key=lambda item: item.fact_id)
    )

    selected: list[EvidenceBundle] = list(single_hop_bundles)
    for relation in sorted(relations, key=evidence_relation_sort_key):
        fact_ids = tuple(sorted((relation.from_fact_id, relation.to_fact_id)))
        kind = classify_bundle_kind(relation, facts_by_id)
        candidate = EvidenceBundle(
            bundle_id=sha256_json(
                {
                    "bridge": list(relation.bridge),
                    "fact_ids": list(fact_ids),
                    "kind": kind,
                    "relation_type": relation.relation_type,
                }
            ),
            kind=kind,
            fact_ids=fact_ids,
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
