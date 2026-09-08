"""Generate typed concept-relation proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ConceptRelationProposalEntry,
    ConceptResolutionInventoryInput,
    ConceptResolutionInventoryResult,
    OntologyConceptKind,
    OntologyConceptOccurrenceInput,
    OntologyEvidenceSource,
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.models import (
    ConceptCandidateInventory,
    ConceptOccurrence,
    ConceptRelationProposal,
)
from legal_rag.ontology_population.evidence import population_evidence_sources
from legal_rag.ontology_population.extraction import bind_population_evidence
from legal_rag.ontology_population.models import population_identity
from legal_rag.ontology_proposal_execution import execute_proposal_batches
from legal_rag.settings import Settings

ONTOLOGY_CONCEPT_PROMPT_VERSION = "0.5.0"

KIND_VALUES = {
    "RegulatedConduct": OntologyConceptKind.RegulatedConduct,
    "LegalSubjectRole": OntologyConceptKind.LegalSubjectRole,
    "VehicleCategory": OntologyConceptKind.VehicleCategory,
}


class ConceptProposalClient(Protocol):
    """Boundary implemented by a concept-resolution language-model client."""

    def propose_batch(
        self,
        inventories: Sequence[ConceptResolutionInventoryInput],
    ) -> Sequence[ConceptResolutionInventoryResult]:
        """Return typed concept relations for controlled-vocabulary inventories."""


class BamlConceptProposalClient:
    """Typed BAML concept client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, concept_settings: ConceptResolutionSettings) -> None:
        """Create a bounded runtime BAML client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=concept_settings.max_output_tokens,
            temperature=concept_settings.temperature,
            max_response_bytes=concept_settings.max_response_bytes,
        )

    def propose_batch(
        self,
        inventories: Sequence[ConceptResolutionInventoryInput],
    ) -> Sequence[ConceptResolutionInventoryResult]:
        """Run one bounded BAML call over concept inventories."""
        request = b.request.ProposeOntologyConceptRelations(list(inventories), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.ProposeOntologyConceptRelations(raw_output, baml_options=self._caller.options)


def build_concept_proposal_run_id(
    candidate_run_id: str,
    concept_settings: ConceptResolutionSettings,
    settings: Settings,
) -> str:
    """Fingerprint every input that can change semantic concept proposals."""
    return population_identity(
        {
            "candidate_run_id": candidate_run_id,
            "prompt_version": ONTOLOGY_CONCEPT_PROMPT_VERSION,
            "batch_size": concept_settings.batch_size,
            "max_output_tokens": concept_settings.max_output_tokens,
            "temperature": concept_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_extra_request_body": dict(settings.llm_extra_request_body),
        }
    )


def build_concept_proposal_rerun_id(base_run_id: str, candidate_ids: Sequence[str]) -> str:
    """Fingerprint one reviewed subset replay independently of the base run."""
    return population_identity({"base_run_id": base_run_id, "rerun_candidate_ids": sorted(candidate_ids)})


def propose_concept_relations(
    inventories: Sequence[ConceptCandidateInventory],
    occurrences: Sequence[ConceptOccurrence],
    client: ConceptProposalClient,
    settings: ConceptResolutionSettings,
    *,
    run_id: str,
    max_concurrency: int = 1,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[ConceptRelationProposal, ...]]]], None] | None = None,
) -> tuple[ConceptRelationProposal, ...]:
    """Generate exact-evidenced concept proposals in concurrent inventory batches."""
    occurrences_by_id = {item.occurrence_id: item for item in occurrences}
    tasks = []
    for start in range(0, len(inventories), settings.batch_size):
        batch = inventories[start : start + settings.batch_size]
        inputs = tuple(_inventory_input(inventory, occurrences_by_id) for inventory in batch)
        tasks.append(
            (
                _batch_task_key(inputs, run_id),
                partial(
                    _build_batch_proposals,
                    client,
                    inputs,
                    occurrences_by_id,
                    run_id,
                ),
            )
        )
    batch_results = execute_proposal_batches(
        tasks,
        stage="ontology concept resolution",
        max_concurrency=max_concurrency,
        max_retries=settings.max_retries,
        retry_delay_seconds=settings.retry_delay_seconds,
        on_batch_processed=on_batch_processed,
    )
    return tuple(
        proposal
        for batch_result in batch_results
        for _, candidate_proposals in batch_result
        for proposal in candidate_proposals
    )


def _build_batch_proposals(
    client: ConceptProposalClient,
    inputs: Sequence[ConceptResolutionInventoryInput],
    occurrences_by_id: Mapping[str, ConceptOccurrence],
    run_id: str,
) -> tuple[tuple[str, tuple[ConceptRelationProposal, ...]], ...]:
    """Translate one typed response and bind inventory references."""
    results = client.propose_batch(inputs)
    requested_ids = [item.candidate_id for item in inputs]
    result_ids = [item.candidate_id for item in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Concept response must contain exactly one result per candidate inventory"
        raise ValueError(msg)
    results_by_id = {item.candidate_id: item for item in results}
    inputs_by_id = {item.candidate_id: item for item in inputs}
    translated = []
    for candidate_id in requested_ids:
        inventory_input = inputs_by_id[candidate_id]
        inventory_occurrences = {item.occurrence_id: item for item in inventory_input.occurrences}
        seen_pairs = set()
        proposals = []
        for raw in results_by_id[candidate_id].proposals:
            if (
                raw.left_occurrence_id not in inventory_occurrences
                or raw.right_occurrence_id not in inventory_occurrences
            ):
                msg = "Concept proposal occurrences must belong to the requested inventory"
                raise ValueError(msg)
            pair = tuple(sorted((raw.left_occurrence_id, raw.right_occurrence_id)))
            if pair[0] == pair[1] or pair in seen_pairs:
                msg = "Concept inventory response must contain unique pairs of distinct occurrences"
                raise ValueError(msg)
            seen_pairs.add(pair)
            left = occurrences_by_id[raw.left_occurrence_id]
            right = occurrences_by_id[raw.right_occurrence_id]
            if raw.relation.value == "Distinct":
                msg = "Concept inventory responses must omit unrelated or distinct pairs"
                raise ValueError(msg)
            if raw.relation.value == "SameConcept" and raw.preferred_label_vi not in {
                left.label_vi,
                right.label_vi,
            }:
                msg = "Same-concept preferred label must be copied from one source occurrence"
                raise ValueError(msg)
            left_evidence = _bind_occurrence_evidence(
                raw,
                inventory_occurrences[raw.left_occurrence_id].evidence,
                side="left",
            )
            right_evidence = _bind_occurrence_evidence(
                raw,
                inventory_occurrences[raw.right_occurrence_id].evidence,
                side="right",
            )
            relation = raw.relation.value
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
            proposals.append(
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
                    run_id=run_id,
                )
            )
        translated.append((candidate_id, tuple(proposals)))
    return tuple(translated)


def _bind_occurrence_evidence(
    raw: ConceptRelationProposalEntry,
    evidence_sources: Sequence[OntologyEvidenceSource],
    *,
    side: str,
) -> tuple:
    """Bind one proposal side to supplied evidence references."""
    raw_evidence = raw.left_evidence if side == "left" else raw.right_evidence
    return bind_population_evidence(raw_evidence, evidence_sources)


def _inventory_input(
    inventory: ConceptCandidateInventory,
    occurrences_by_id: Mapping[str, ConceptOccurrence],
) -> ConceptResolutionInventoryInput:
    """Build one complete typed BAML controlled-vocabulary inventory."""
    return ConceptResolutionInventoryInput(
        candidate_id=inventory.candidate_id,
        kind=KIND_VALUES[inventory.kind],
        occurrences=[_occurrence_input(occurrences_by_id[item]) for item in inventory.occurrence_ids],
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
        evidence=list(population_evidence_sources(occurrence.evidence)),
    )


def _batch_task_key(inputs: Sequence[ConceptResolutionInventoryInput], run_id: str) -> str:
    """Return a deterministic retry key for one concept proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-concept:{digest}"
