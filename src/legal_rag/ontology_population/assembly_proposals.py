"""Generate typed cross-chunk rule-assembly proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    AssemblyConditionSummary,
    AssemblyConsequenceSummary,
    OntologyConditionKind,
    OntologyConsequenceKind,
    OntologyCurrency,
    OntologyEvidenceSource,
    OntologyPopulationSummary,
    OntologySubjectKind,
    RuleAssemblyProposal,
    RuleAssemblyWindowInput,
    RuleAssemblyWindowResult,
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_models import AssemblyCandidateWindow, AssemblyProposal
from legal_rag.ontology_population.evidence import evidence_source
from legal_rag.ontology_population.extraction import build_population_record
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity
from legal_rag.ontology_proposal_execution import execute_proposal_batches
from legal_rag.settings import Settings

ONTOLOGY_ASSEMBLY_PROMPT_VERSION = "0.6.0"


class AssemblyProposalClient(Protocol):
    """Boundary implemented by a rule-assembly language-model client."""

    def propose_batch(self, windows: Sequence[RuleAssemblyWindowInput]) -> Sequence[RuleAssemblyWindowResult]:
        """Return typed assembly proposals for candidate windows."""


class BamlAssemblyProposalClient:
    """Typed BAML assembly client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, assembly_settings: AssemblySettings) -> None:
        """Create a bounded runtime BAML client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=assembly_settings.max_output_tokens,
            temperature=assembly_settings.temperature,
            max_response_bytes=assembly_settings.max_response_bytes,
        )

    def propose_batch(self, windows: Sequence[RuleAssemblyWindowInput]) -> Sequence[RuleAssemblyWindowResult]:
        """Run one bounded BAML call over candidate windows."""
        request = b.request.ProposeOntologyRuleAssemblies(list(windows), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.ProposeOntologyRuleAssemblies(raw_output, baml_options=self._caller.options)


def build_assembly_proposal_run_id(
    candidate_run_id: str,
    assembly_settings: AssemblySettings,
    settings: Settings,
) -> str:
    """Fingerprint every input that can change semantic assembly proposals."""
    return population_identity(
        {
            "candidate_run_id": candidate_run_id,
            "prompt_version": ONTOLOGY_ASSEMBLY_PROMPT_VERSION,
            "max_output_tokens": assembly_settings.max_output_tokens,
            "temperature": assembly_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_extra_request_body": dict(settings.llm_extra_request_body),
        }
    )


def resolve_assembly_proposal_resume_run_id(
    persisted_run_ids: set[str],
    candidate_run_id: str,
    candidate_count: int,
    assembly_settings: AssemblySettings,
    settings: Settings,
) -> str:
    """Accept legacy checkpoints whose only semantic difference is batch size."""
    current_run_id = build_assembly_proposal_run_id(candidate_run_id, assembly_settings, settings)
    if not persisted_run_ids or persisted_run_ids == {current_run_id}:
        return current_run_id
    if len(persisted_run_ids) != 1:
        return current_run_id
    persisted_run_id = next(iter(persisted_run_ids))
    for batch_size in range(1, candidate_count + 1):
        legacy_run_id = population_identity(
            {
                "candidate_run_id": candidate_run_id,
                "prompt_version": ONTOLOGY_ASSEMBLY_PROMPT_VERSION,
                "batch_size": batch_size,
                "max_output_tokens": assembly_settings.max_output_tokens,
                "temperature": assembly_settings.temperature,
                "llm_base_url": settings.llm_base_url,
                "llm_model": settings.llm_model,
                "llm_extra_request_body": dict(settings.llm_extra_request_body),
            }
        )
        if persisted_run_id == legacy_run_id:
            return persisted_run_id
    return current_run_id


def propose_assemblies(
    windows: Sequence[AssemblyCandidateWindow],
    records: Sequence[OntologyPopulationRecord],
    chunks: Sequence[Chunk],
    metadata: DocumentMetadataRecord,
    client: AssemblyProposalClient,
    assembly_settings: AssemblySettings,
    *,
    run_id: str,
    max_concurrency: int = 1,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[AssemblyProposal, ...]]]], None] | None = None,
) -> tuple[AssemblyProposal, ...]:
    """Generate source-grounded proposals in bounded concurrent batches."""
    records_by_id = {record.population_id: record for record in records}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    tasks = []
    for batch_start in range(0, len(windows), assembly_settings.batch_size):
        batch = windows[batch_start : batch_start + assembly_settings.batch_size]
        inputs = tuple(_window_input(window, records_by_id) for window in batch)
        tasks.append(
            (
                _batch_task_key(inputs, run_id),
                partial(
                    _build_batch_proposals,
                    client,
                    inputs,
                    batch,
                    records_by_id,
                    chunks_by_id,
                    metadata,
                    assembly_settings,
                    run_id,
                ),
            )
        )
    batch_results = execute_proposal_batches(
        tasks,
        stage="ontology rule assembly",
        max_concurrency=max_concurrency,
        max_retries=assembly_settings.max_retries,
        retry_delay_seconds=assembly_settings.retry_delay_seconds,
        on_batch_processed=on_batch_processed,
    )
    return tuple(
        proposal
        for batch_result in batch_results
        for _, candidate_proposals in batch_result
        for proposal in candidate_proposals
    )


def _build_batch_proposals(
    client: AssemblyProposalClient,
    inputs: Sequence[RuleAssemblyWindowInput],
    windows: Sequence[AssemblyCandidateWindow],
    records_by_id: Mapping[str, OntologyPopulationRecord],
    chunks_by_id: Mapping[str, Chunk],
    metadata: DocumentMetadataRecord,
    assembly_settings: AssemblySettings,
    run_id: str,
) -> tuple[tuple[str, tuple[AssemblyProposal, ...]], ...]:
    """Translate one typed model response and bind candidate references."""
    results = client.propose_batch(inputs)
    requested_ids = [item.candidate_id for item in inputs]
    result_ids = [result.candidate_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Assembly response must contain exactly one result per candidate window"
        raise ValueError(msg)
    results_by_id = {result.candidate_id: result for result in results}
    windows_by_id = {window.candidate_id: window for window in windows}
    translated = []
    for candidate_id in requested_ids:
        window = windows_by_id[candidate_id]
        used_members: set[str] = set()
        candidate_proposals = []
        for raw in results_by_id[candidate_id].proposals:
            members = tuple(sorted(raw.member_population_ids))
            if len(members) < 2 or len(members) != len(set(members)) or not set(members) <= set(window.population_ids):
                msg = "Assembly proposal members must be unique candidate-window population ids"
                raise ValueError(msg)
            if used_members & set(members):
                msg = "Assembly proposals within one candidate window must not overlap members"
                raise ValueError(msg)
            used_members.update(members)
            candidate_proposals.append(
                build_assembly_proposal(
                    raw,
                    candidate_id,
                    records_by_id,
                    chunks_by_id,
                    metadata,
                    run_id,
                )
            )
        translated.append((candidate_id, tuple(candidate_proposals)))
    return tuple(translated)


def build_assembly_proposal(
    raw: RuleAssemblyProposal,
    candidate_id: str,
    records_by_id: Mapping[str, OntologyPopulationRecord],
    chunks_by_id: Mapping[str, Chunk],
    metadata: DocumentMetadataRecord,
    run_id: str,
) -> AssemblyProposal:
    """Bind one typed model proposal to current population records and evidence."""
    members = tuple(sorted(raw.member_population_ids))
    anchor = min(
        (records_by_id[member] for member in members),
        key=lambda record: (
            chunks_by_id[record.source_chunk_id].scope_id,
            chunks_by_id[record.source_chunk_id].chunk_index,
            record.population_id,
        ),
    )
    ontology_version, ontology_semantic_sha256 = _member_ontology_identity(members, records_by_id)
    assembled_record = build_population_record(
        raw.assembled_rule,
        anchor.source_chunk_id,
        metadata,
        [
            evidence_source(chunk_id, exact_text)
            for chunk_id, exact_text in sorted(
                set().union(*(_record_evidence(records_by_id[member]) for member in members))
            )
        ],
        ontology_version=ontology_version,
        ontology_semantic_sha256=ontology_semantic_sha256,
        population_run_id=run_id,
    )
    proposal_id = population_identity(
        {
            "run_id": run_id,
            "member_population_ids": members,
            "assembled_population_id": assembled_record.population_id,
        }
    )
    return AssemblyProposal(
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        member_population_ids=members,
        assembled_record=assembled_record,
        rationale_vi=raw.rationale_vi,
        run_id=run_id,
    )


def _member_ontology_identity(
    members: tuple[str, ...],
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> tuple[str, str]:
    """Require one ontology version and semantic closure across assembled member records."""
    identities = {
        (records_by_id[member].ontology_version, records_by_id[member].ontology_semantic_sha256) for member in members
    }
    if len(identities) != 1:
        msg = "Assembly proposal members must use one ontology version and semantic hash"
        raise ValueError(msg)
    ontology_version, ontology_semantic_sha256 = identities.pop()
    if not ontology_version or not ontology_semantic_sha256:
        msg = "Assembly proposal members require current ontology provenance"
        raise ValueError(msg)
    return ontology_version, ontology_semantic_sha256


def _window_input(
    window: AssemblyCandidateWindow,
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> RuleAssemblyWindowInput:
    """Build one complete typed BAML candidate window."""
    return RuleAssemblyWindowInput(
        candidate_id=window.candidate_id,
        records=[summarize_population_record(records_by_id[population_id]) for population_id in window.population_ids],
    )


def summarize_population_record(record: OntologyPopulationRecord) -> OntologyPopulationSummary:
    """Translate a population record without dropping semantic or evidence fields."""
    return OntologyPopulationSummary(
        population_id=record.population_id,
        source_chunk_id=record.source_chunk_id,
        conduct_label_vi=record.conduct_label_vi,
        subject_kind=OntologySubjectKind(record.subject_kind) if record.subject_kind else None,
        role_labels_vi=list(record.role_labels_vi),
        vehicle_category_labels_vi=list(record.vehicle_category_labels_vi),
        condition_root_id=record.condition_root_id,
        conditions=[
            AssemblyConditionSummary(
                condition_id=condition.condition_id,
                kind=OntologyConditionKind(condition.kind),
                text_vi=condition.text_vi,
                operand_ids=list(condition.operand_ids),
                evidence=[_evidence_source(item.chunk_id, item.exact_text) for item in condition.evidence],
            )
            for condition in record.conditions
        ],
        consequences=[
            AssemblyConsequenceSummary(
                consequence_id=consequence.consequence_id,
                kind=OntologyConsequenceKind(consequence.kind),
                minimum_amount=str(consequence.minimum_amount) if consequence.minimum_amount is not None else None,
                maximum_amount=str(consequence.maximum_amount) if consequence.maximum_amount is not None else None,
                currency=OntologyCurrency(consequence.currency_code) if consequence.currency_code else None,
                point_value=consequence.point_value,
                consequence_label_vi=consequence.consequence_label_vi,
                minimum_duration_months=consequence.minimum_duration_months,
                maximum_duration_months=consequence.maximum_duration_months,
                target_label_vi=consequence.target_label_vi,
                responsible_role_label_vi=consequence.responsible_role_label_vi,
                evidence=[_evidence_source(item.chunk_id, item.exact_text) for item in consequence.evidence],
            )
            for consequence in record.consequences
        ],
        valid_from=record.valid_from.isoformat() if record.valid_from else None,
        valid_to=record.valid_to.isoformat() if record.valid_to else None,
        document_number=record.document_number,
        provision_path=record.provision_path,
        evidence=[_evidence_source(item.chunk_id, item.exact_text) for item in record.evidence],
    )


def _evidence_source(chunk_id: str, exact_text: str) -> OntologyEvidenceSource:
    """Build one BAML evidence source."""
    return evidence_source(chunk_id, exact_text)


def _record_evidence(record: OntologyPopulationRecord) -> set[tuple[str, str]]:
    """Collect every exact evidence span available on one population record."""
    evidence = {(item.chunk_id, item.exact_text) for item in record.evidence}
    for condition in record.conditions:
        evidence.update((item.chunk_id, item.exact_text) for item in condition.evidence)
    for consequence in record.consequences:
        evidence.update((item.chunk_id, item.exact_text) for item in consequence.evidence)
    return evidence


def _batch_task_key(inputs: Sequence[RuleAssemblyWindowInput], run_id: str) -> str:
    """Return a deterministic retry key for one proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-assembly:{digest}"
