"""Generate typed cross-document legal-change proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    CrossDocumentChangeWindowInput,
    CrossDocumentChangeWindowResult,
    LegalChangeConditionSummary,
    LegalChangeConsequenceSummary,
    LegalChangeRuleSummary,
    OntologyConditionKind,
    OntologyConsequenceKind,
    OntologyCurrency,
    OntologySubjectKind,
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_change.models import ChangeCandidateWindow, LegalChangeProposal
from legal_rag.ontology_population.evidence import fragment_evidence_sources
from legal_rag.ontology_population.extraction import bind_population_evidence
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity
from legal_rag.ontology_proposal_execution import execute_proposal_batches
from legal_rag.settings import Settings

ONTOLOGY_CHANGE_PROMPT_VERSION = "1.0.0"


class ChangeProposalClient(Protocol):
    """Boundary implemented by a legal-change language-model client."""

    def propose_batch(
        self, windows: Sequence[CrossDocumentChangeWindowInput]
    ) -> Sequence[CrossDocumentChangeWindowResult]:
        """Return typed legal-change proposals for candidate windows."""


class BamlChangeProposalClient:
    """Typed BAML legal-change client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, change_settings: ChangeResolutionSettings) -> None:
        """Create a bounded runtime BAML client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=change_settings.max_output_tokens,
            temperature=change_settings.temperature,
            max_response_bytes=change_settings.max_response_bytes,
        )

    def propose_batch(
        self, windows: Sequence[CrossDocumentChangeWindowInput]
    ) -> Sequence[CrossDocumentChangeWindowResult]:
        """Run one bounded BAML call over cross-document windows."""
        request = b.request.ProposeCrossDocumentLegalChanges(list(windows), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.ProposeCrossDocumentLegalChanges(raw_output, baml_options=self._caller.options)


def build_change_proposal_run_id(
    candidate_run_id: str,
    change_settings: ChangeResolutionSettings,
    settings: Settings,
) -> str:
    """Fingerprint every input that can change semantic legal-change proposals."""
    return population_identity(
        {
            "candidate_run_id": candidate_run_id,
            "prompt_version": ONTOLOGY_CHANGE_PROMPT_VERSION,
            "batch_size": change_settings.batch_size,
            "max_output_tokens": change_settings.max_output_tokens,
            "temperature": change_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_extra_request_body": dict(settings.llm_extra_request_body),
        }
    )


def propose_legal_changes(
    windows: Sequence[ChangeCandidateWindow],
    corpus: ChangeCorpus,
    client: ChangeProposalClient,
    change_settings: ChangeResolutionSettings,
    *,
    run_id: str,
    max_concurrency: int = 1,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[LegalChangeProposal, ...]]]], None] | None = None,
) -> tuple[LegalChangeProposal, ...]:
    """Generate exact-evidenced legal-change proposals in concurrent batches."""
    documents_by_id = {document.document_id: document for document in corpus.documents}
    records_by_id = {record.population_id: record for document in corpus.documents for record in document.records}
    tasks = []
    for batch_start in range(0, len(windows), change_settings.batch_size):
        batch = windows[batch_start : batch_start + change_settings.batch_size]
        inputs = tuple(_window_input(window, documents_by_id, records_by_id) for window in batch)
        tasks.append(
            (
                _batch_task_key(inputs, run_id),
                partial(
                    _build_batch_proposals,
                    client,
                    inputs,
                    batch,
                    documents_by_id,
                    change_settings,
                    run_id,
                ),
            )
        )
    batch_results = execute_proposal_batches(
        tasks,
        stage="ontology legal-change resolution",
        max_concurrency=max_concurrency,
        max_retries=change_settings.max_retries,
        retry_delay_seconds=change_settings.retry_delay_seconds,
        on_batch_processed=on_batch_processed,
    )
    return tuple(
        proposal
        for batch_result in batch_results
        for _, candidate_proposals in batch_result
        for proposal in candidate_proposals
    )


def _build_batch_proposals(
    client: ChangeProposalClient,
    inputs: Sequence[CrossDocumentChangeWindowInput],
    windows: Sequence[ChangeCandidateWindow],
    documents_by_id: Mapping[str, ChangeCorpusDocument],
    change_settings: ChangeResolutionSettings,
    run_id: str,
) -> tuple[tuple[str, tuple[LegalChangeProposal, ...]], ...]:
    """Translate one typed model response and bind window references."""
    results = client.propose_batch(inputs)
    requested_ids = [item.candidate_id for item in inputs]
    result_ids = [result.candidate_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Legal-change response must contain exactly one result per candidate window"
        raise ValueError(msg)
    results_by_id = {result.candidate_id: result for result in results}
    inputs_by_id = {item.candidate_id: item for item in inputs}
    windows_by_id = {window.candidate_id: window for window in windows}
    translated = []
    for candidate_id in requested_ids:
        window = windows_by_id[candidate_id]
        later_document = documents_by_id[window.later_document_id]
        evidence_sources = inputs_by_id[candidate_id].change_evidence_sources
        candidate_proposals = []
        for raw in results_by_id[candidate_id].proposals:
            _validate_members(raw.earlier_population_id, raw.later_population_id, window)
            evidence = bind_population_evidence(raw.evidence, evidence_sources)
            effective_on = date.fromisoformat(raw.effective_on)
            proposal_id = population_identity(
                {
                    "run_id": run_id,
                    "earlier_population_id": raw.earlier_population_id,
                    "later_population_id": raw.later_population_id,
                    "change_type": raw.change_type.value,
                    "effective_on": effective_on.isoformat(),
                    "change_document_number": later_document.metadata.document_number,
                    "change_provision_path": raw.change_provision_path,
                    "evidence": [item.to_dict() for item in evidence],
                }
            )
            candidate_proposals.append(
                LegalChangeProposal(
                    proposal_id=proposal_id,
                    candidate_id=candidate_id,
                    earlier_population_id=raw.earlier_population_id,
                    later_population_id=raw.later_population_id,
                    change_type=raw.change_type.value,
                    effective_on=effective_on,
                    change_document_number=later_document.metadata.document_number,
                    change_provision_path=raw.change_provision_path,
                    evidence=evidence,
                    rationale_vi=raw.rationale_vi,
                    run_id=run_id,
                )
            )
        translated.append((candidate_id, tuple(candidate_proposals)))
    return tuple(translated)


def _validate_members(
    earlier_id: str,
    later_id: str | None,
    window: ChangeCandidateWindow,
) -> None:
    """Require proposal members to belong to the current window."""
    if earlier_id not in window.earlier_population_ids:
        msg = "Legal-change earlier rule must be a member of the candidate window"
        raise ValueError(msg)
    if later_id is not None and later_id not in window.later_population_ids:
        msg = "Legal-change later rule must be a member of the candidate window"
        raise ValueError(msg)


def _window_input(
    window: ChangeCandidateWindow,
    documents_by_id: Mapping[str, ChangeCorpusDocument],
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> CrossDocumentChangeWindowInput:
    """Build one complete typed BAML cross-document window."""
    earlier_document = documents_by_id[window.earlier_document_id]
    later_document = documents_by_id[window.later_document_id]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in later_document.chunks}
    return CrossDocumentChangeWindowInput(
        candidate_id=window.candidate_id,
        earlier_document_number=earlier_document.metadata.document_number,
        later_document_number=later_document.metadata.document_number,
        later_document_valid_from=(
            later_document.metadata.valid_from.isoformat() if later_document.metadata.valid_from else None
        ),
        earlier_records=[
            _summarize_change_record(records_by_id[population_id]) for population_id in window.earlier_population_ids
        ],
        later_records=[
            _summarize_change_record(records_by_id[population_id]) for population_id in window.later_population_ids
        ],
        change_evidence_sources=[
            source
            for chunk_id in window.change_source_chunk_ids
            for source in fragment_evidence_sources(chunk_id, chunks_by_id[chunk_id].text)
        ],
    )


def _summarize_change_record(record: OntologyPopulationRecord) -> LegalChangeRuleSummary:
    """Keep rule semantics while leaving source evidence in the dedicated change context."""
    return LegalChangeRuleSummary(
        population_id=record.population_id,
        conduct_label_vi=record.conduct_label_vi,
        subject_kind=OntologySubjectKind(record.subject_kind) if record.subject_kind else None,
        role_labels_vi=list(record.role_labels_vi),
        vehicle_category_labels_vi=list(record.vehicle_category_labels_vi),
        condition_root_id=record.condition_root_id,
        conditions=[
            LegalChangeConditionSummary(
                condition_id=condition.condition_id,
                kind=OntologyConditionKind(condition.kind),
                text_vi=condition.text_vi,
                operand_ids=list(condition.operand_ids),
            )
            for condition in record.conditions
        ],
        consequences=[
            LegalChangeConsequenceSummary(
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
            )
            for consequence in record.consequences
        ],
        valid_from=record.valid_from.isoformat() if record.valid_from else None,
        valid_to=record.valid_to.isoformat() if record.valid_to else None,
        document_number=record.document_number,
        provision_path=record.provision_path,
    )


def _batch_task_key(inputs: Sequence[CrossDocumentChangeWindowInput], run_id: str) -> str:
    """Return a deterministic retry key for one legal-change proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-change:{digest}"
