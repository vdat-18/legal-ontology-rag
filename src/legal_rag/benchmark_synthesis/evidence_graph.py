"""Confirm candidate evidence relations and sample bundles for scenario generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ConfirmedRelation as BamlConfirmedRelation
from legal_rag.baml_client.types import LegalFactSummary
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import EvaluationSettings, Settings

EVIDENCE_RELATION_SCHEMA_VERSION = "1.0.0"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "1.0.0"

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

DEFAULT_MAX_CANDIDATES_PER_FACT = 5
DEFAULT_MIN_CANDIDATE_SIMILARITY = 0.5


@dataclass(frozen=True, slots=True)
class EvidenceRelation:
    """One confirmed relation between two approved facts."""

    relation_id: str
    relation_type: RelationType
    from_fact_id: str
    to_fact_id: str
    bridge: tuple[str, ...]
    confirmed_by_model: str

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


def _fact_embedding_text(fact: LegalFact) -> str:
    """Return a canonical text representation of a fact for similarity search."""
    return f"{fact.conduct}. {fact.subject}. {fact.provision_path}."


def find_candidate_fact_pairs(
    facts: Sequence[LegalFact],
    encoder: EmbeddingEncoder,
    *,
    max_candidates_per_fact: int = DEFAULT_MAX_CANDIDATES_PER_FACT,
    min_similarity: float = DEFAULT_MIN_CANDIDATE_SIMILARITY,
) -> tuple[tuple[str, str], ...]:
    """Return unique fact_id pairs worth confirming as a relation, by embedding similarity."""
    vectors = encoder.encode([_fact_embedding_text(fact) for fact in facts])
    pairs: set[tuple[str, str]] = set()
    for index, fact in enumerate(facts):
        scored = [
            (float(np.dot(vectors[index], vectors[other_index])), other_fact)
            for other_index, other_fact in enumerate(facts)
            if other_index != index
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, other_fact in scored[:max_candidates_per_fact]:
            if score < min_similarity:
                continue
            pairs.add(tuple(sorted((fact.fact_id, other_fact.fact_id))))
    return tuple(sorted(pairs))


class EvidenceRelationClient(Protocol):
    """Boundary implemented by an evidence-relation-confirmation language-model client."""

    def confirm(self, fact_a: LegalFact, fact_b: LegalFact) -> EvidenceRelation | None:
        """Return a confirmed relation, or None when the facts are not directly related."""


class BamlEvidenceRelationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, evaluation_settings: EvaluationSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=evaluation_settings.fact_extraction_max_output_tokens,
            temperature=evaluation_settings.fact_extraction_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = evaluation_settings.max_response_bytes
        self._model = settings.llm_model

    def confirm(self, fact_a: LegalFact, fact_b: LegalFact) -> EvidenceRelation | None:
        """Run one bounded BAML call and translate its typed result."""
        request = b.request.ConfirmEvidenceRelation(
            _summarize(fact_a), _summarize(fact_b), baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        result = b.parse.ConfirmEvidenceRelation(raw_output, baml_options={"client_registry": self._registry})
        return _build_relation(result, fact_a.fact_id, fact_b.fact_id, self._model)


def _summarize(fact: LegalFact) -> LegalFactSummary:
    """Return the BAML-facing summary of a fact used for relation confirmation."""
    return LegalFactSummary(
        fact_id=fact.fact_id,
        conduct=fact.conduct,
        subject=fact.subject,
        monetary_penalty=fact.monetary_penalty,
        license_points_deducted=fact.license_points_deducted,
        additional_penalties=list(fact.additional_penalties),
        remedial_measures=list(fact.remedial_measures),
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
        provision_path=fact.provision_path,
    )


def _build_relation(
    result: BamlConfirmedRelation,
    from_fact_id: str,
    to_fact_id: str,
    model: str,
) -> EvidenceRelation | None:
    """Translate one BAML confirmation result into a persisted relation, or None."""
    if not result.is_related or result.relation_type is None:
        return None
    relation_type = _RELATION_TYPE_BY_BAML_ENUM[result.relation_type.value]
    identity = {"from_fact_id": from_fact_id, "to_fact_id": to_fact_id, "relation_type": relation_type}
    return EvidenceRelation(
        relation_id=sha256_json(identity),
        relation_type=relation_type,
        from_fact_id=from_fact_id,
        to_fact_id=to_fact_id,
        bridge=tuple(result.bridge),
        confirmed_by_model=model,
    )


def confirm_evidence_relations(
    facts: Sequence[LegalFact],
    candidate_pairs: Sequence[tuple[str, str]],
    client: EvidenceRelationClient,
) -> tuple[EvidenceRelation, ...]:
    """Confirm every candidate pair, keeping only genuinely related facts."""
    facts_by_id = {fact.fact_id: fact for fact in facts}
    relations: list[EvidenceRelation] = []
    for fact_id_a, fact_id_b in candidate_pairs:
        relation = client.confirm(facts_by_id[fact_id_a], facts_by_id[fact_id_b])
        if relation is not None:
            relations.append(relation)
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
        for fact in facts
    )

    selected: list[EvidenceBundle] = list(single_hop_bundles)
    for relation in relations:
        fact_ids = tuple(sorted((relation.from_fact_id, relation.to_fact_id)))
        candidate = EvidenceBundle(
            bundle_id=sha256_json({"fact_ids": list(fact_ids)}),
            kind=classify_bundle_kind(relation, facts_by_id),
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
