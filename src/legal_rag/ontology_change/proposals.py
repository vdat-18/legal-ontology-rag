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
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_change.models import ChangeCandidateWindow, LegalChangeProposal
from legal_rag.ontology_population.assembly_proposals import summarize_population_record
from legal_rag.ontology_population.evidence import fragment_evidence_sources
from legal_rag.ontology_population.extraction import bind_population_evidence
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity
from legal_rag.settings import Settings

ONTOLOGY_CHANGE_PROMPT_VERSION = "0.2.0"


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
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[LegalChangeProposal, ...]]]], None] | None = None,
) -> tuple[LegalChangeProposal, ...]:
    """Generate exact-evidenced legal-change proposals in resumable batches."""
    documents_by_id = {document.document_id: document for document in corpus.documents}
    records_by_id = {record.population_id: record for document in corpus.documents for record in document.records}
    proposals = []
    for batch_start in range(0, len(windows), change_settings.batch_size):
        batch = windows[batch_start : batch_start + change_settings.batch_size]
        inputs = tuple(_window_input(window, documents_by_id, records_by_id) for window in batch)
        batch_results = execute_llm_task(
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
            max_retries=change_settings.max_retries,
            retry_delay_seconds=change_settings.retry_delay_seconds,
        )
        for _, candidate_proposals in batch_results:
            proposals.extend(candidate_proposals)
        if on_batch_processed is not None:
            on_batch_processed(batch_results)
    return tuple(proposals)


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
        used_earlier: set[str] = set()
        used_later: set[str] = set()
        candidate_proposals = []
        for raw in results_by_id[candidate_id].proposals:
            _validate_members(raw.earlier_population_id, raw.later_population_id, window, used_earlier, used_later)
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
    used_earlier: set[str],
    used_later: set[str],
) -> None:
    """Require proposal members to be unique members of the current window."""
    if earlier_id not in window.earlier_population_ids or earlier_id in used_earlier:
        msg = "Legal-change earlier rule must be a unique member of the candidate window"
        raise ValueError(msg)
    if later_id is not None and (later_id not in window.later_population_ids or later_id in used_later):
        msg = "Legal-change later rule must be a unique member of the candidate window"
        raise ValueError(msg)
    used_earlier.add(earlier_id)
    if later_id is not None:
        used_later.add(later_id)


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
        earlier_records=[
            summarize_population_record(records_by_id[population_id]) for population_id in window.earlier_population_ids
        ],
        later_records=[
            summarize_population_record(records_by_id[population_id]) for population_id in window.later_population_ids
        ],
        change_evidence_sources=[
            source
            for chunk_id in window.change_source_chunk_ids
            for source in fragment_evidence_sources(chunk_id, chunks_by_id[chunk_id].text)
        ],
    )


def _batch_task_key(inputs: Sequence[CrossDocumentChangeWindowInput], run_id: str) -> str:
    """Return a deterministic retry key for one legal-change proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-change:{digest}"
