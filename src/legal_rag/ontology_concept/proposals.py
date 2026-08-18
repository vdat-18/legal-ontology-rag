"""Generate typed concept-relation proposals for mandatory human review."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ConceptRelationProposalEntry,
    ConceptResolutionPairInput,
    ConceptResolutionPairResult,
    OntologyConceptKind,
    OntologyConceptOccurrenceInput,
    OntologyEvidenceEntry,
)
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.models import (
    ConceptCandidatePair,
    ConceptOccurrence,
    ConceptRelationProposal,
)
from legal_rag.ontology_population.extraction import bind_population_evidence
from legal_rag.ontology_population.models import population_identity
from legal_rag.settings import Settings

KIND_VALUES = {
    "RegulatedConduct": OntologyConceptKind.RegulatedConduct,
    "LegalSubjectRole": OntologyConceptKind.LegalSubjectRole,
    "VehicleCategory": OntologyConceptKind.VehicleCategory,
}


class ConceptProposalClient(Protocol):
    """Boundary implemented by a concept-resolution language-model client."""

    @property
    def model(self) -> str:
        """Return the model identifier recorded on proposals."""

    def propose_batch(self, pairs: Sequence[ConceptResolutionPairInput]) -> Sequence[ConceptResolutionPairResult]:
        """Return typed concept relation proposals for candidate pairs."""


class BamlConceptProposalClient:
    """Typed BAML concept client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, concept_settings: ConceptResolutionSettings) -> None:
        """Create a bounded runtime BAML client."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=concept_settings.max_output_tokens,
            temperature=concept_settings.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = concept_settings.max_response_bytes
        self._model = settings.llm_model

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def propose_batch(self, pairs: Sequence[ConceptResolutionPairInput]) -> Sequence[ConceptResolutionPairResult]:
        """Run one bounded BAML call over concept pairs."""
        request = b.request.ProposeOntologyConceptRelations(
            list(pairs), baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ProposeOntologyConceptRelations(raw_output, baml_options={"client_registry": self._registry})


def build_concept_proposal_run_id(
    candidate_run_id: str,
    concept_settings: ConceptResolutionSettings,
    settings: Settings,
) -> str:
    """Fingerprint every input that can change semantic concept proposals."""
    return population_identity(
        {
            "candidate_run_id": candidate_run_id,
            "prompt_version": concept_settings.prompt_version,
            "batch_size": concept_settings.batch_size,
            "max_output_tokens": concept_settings.max_output_tokens,
            "temperature": concept_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
        }
    )


def propose_concept_relations(
    pairs: Sequence[ConceptCandidatePair],
    occurrences: Sequence[ConceptOccurrence],
    chunk_text_by_id: Mapping[str, str],
    client: ConceptProposalClient,
    settings: ConceptResolutionSettings,
    *,
    run_id: str,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[ConceptRelationProposal, ...]]]], None] | None = None,
) -> tuple[ConceptRelationProposal, ...]:
    """Generate exact-evidenced concept proposals in resumable batches."""
    occurrences_by_id = {item.occurrence_id: item for item in occurrences}
    proposals = []
    for start in range(0, len(pairs), settings.batch_size):
        batch = pairs[start : start + settings.batch_size]
        inputs = tuple(_pair_input(pair, occurrences_by_id) for pair in batch)
        batch_results = execute_llm_task(
            _batch_task_key(inputs, run_id),
            partial(
                _build_batch_proposals,
                client,
                inputs,
                batch,
                occurrences_by_id,
                chunk_text_by_id,
                settings,
                run_id,
            ),
            max_retries=settings.max_retries,
            retry_delay_seconds=settings.retry_delay_seconds,
        )
        for _, candidate_proposals in batch_results:
            proposals.extend(candidate_proposals)
        if on_batch_processed is not None:
            on_batch_processed(batch_results)
    return tuple(proposals)


def _build_batch_proposals(
    client: ConceptProposalClient,
    inputs: Sequence[ConceptResolutionPairInput],
    pairs: Sequence[ConceptCandidatePair],
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    chunk_text_by_id: Mapping[str, str],
    settings: ConceptResolutionSettings,
    run_id: str,
) -> tuple[tuple[str, tuple[ConceptRelationProposal, ...]], ...]:
    """Translate one typed response and enforce pair and evidence membership."""
    results = client.propose_batch(inputs)
    requested_ids = [item.candidate_id for item in inputs]
    result_ids = [item.candidate_id for item in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Concept response must contain exactly one result per candidate pair"
        raise ValueError(msg)
    results_by_id = {item.candidate_id: item for item in results}
    pairs_by_id = {item.candidate_id: item for item in pairs}
    translated = []
    for candidate_id in requested_ids:
        raw = results_by_id[candidate_id].proposal
        if raw is None:
            translated.append((candidate_id, ()))
            continue
        pair = pairs_by_id[candidate_id]
        left = occurrences_by_id[pair.left_occurrence_id]
        right = occurrences_by_id[pair.right_occurrence_id]
        left_evidence = _bind_occurrence_evidence(raw, left, chunk_text_by_id, side="left")
        right_evidence = _bind_occurrence_evidence(raw, right, chunk_text_by_id, side="right")
        relation = raw.relation.value
        if relation == "SameConcept":
            if raw.preferred_label_vi not in {left.label_vi, right.label_vi}:
                msg = "SameConcept preferred label must be one supplied occurrence label"
                raise ValueError(msg)
        elif raw.preferred_label_vi:
            msg = "Only SameConcept proposals may select a canonical preferred label"
            raise ValueError(msg)
        proposal_id = population_identity(
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "left_occurrence_id": left.occurrence_id,
                "right_occurrence_id": right.occurrence_id,
                "relation": relation,
                "preferred_label_vi": raw.preferred_label_vi,
                "left_evidence": [item.to_dict() for item in left_evidence],
                "right_evidence": [item.to_dict() for item in right_evidence],
            }
        )
        translated.append(
            (
                candidate_id,
                (
                    ConceptRelationProposal(
                        proposal_id=proposal_id,
                        candidate_id=candidate_id,
                        left_occurrence_id=left.occurrence_id,
                        right_occurrence_id=right.occurrence_id,
                        relation=relation,
                        preferred_label_vi=raw.preferred_label_vi,
                        left_evidence=left_evidence,
                        right_evidence=right_evidence,
                        rationale_vi=raw.rationale_vi,
                        proposed_by_model=client.model,
                        prompt_version=settings.prompt_version,
                        run_id=run_id,
                    ),
                ),
            )
        )
    return tuple(translated)


def _bind_occurrence_evidence(
    raw: ConceptRelationProposalEntry,
    occurrence: ConceptOccurrence,
    chunk_text_by_id: Mapping[str, str],
    *,
    side: str,
) -> tuple:
    """Bind one proposal side to exact evidence owned by that occurrence."""
    raw_evidence = raw.left_evidence if side == "left" else raw.right_evidence
    evidence = bind_population_evidence(raw_evidence, chunk_text_by_id)
    if not evidence or not set(evidence) <= set(occurrence.evidence):
        msg = f"Concept {side} evidence must belong to its supplied occurrence"
        raise ValueError(msg)
    return evidence


def _pair_input(
    pair: ConceptCandidatePair,
    occurrences_by_id: Mapping[str, ConceptOccurrence],
) -> ConceptResolutionPairInput:
    """Build one complete typed BAML pair input."""
    return ConceptResolutionPairInput(
        candidate_id=pair.candidate_id,
        left=_occurrence_input(occurrences_by_id[pair.left_occurrence_id]),
        right=_occurrence_input(occurrences_by_id[pair.right_occurrence_id]),
    )


def _occurrence_input(occurrence: ConceptOccurrence) -> OntologyConceptOccurrenceInput:
    """Translate one occurrence to its typed BAML boundary."""
    return OntologyConceptOccurrenceInput(
        occurrence_id=occurrence.occurrence_id,
        kind=KIND_VALUES[occurrence.kind],
        label_vi=occurrence.label_vi,
        population_id=occurrence.population_id,
        document_number=occurrence.document_number,
        provision_path=occurrence.provision_path,
        evidence=[
            OntologyEvidenceEntry(chunk_id=item.chunk_id, quoted_span=item.exact_text) for item in occurrence.evidence
        ],
    )


def _batch_task_key(inputs: Sequence[ConceptResolutionPairInput], run_id: str) -> str:
    """Return a deterministic retry key for one concept proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-concept:{digest}"
