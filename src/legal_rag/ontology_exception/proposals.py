"""Generate typed legal-exception proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ExceptionCitationProposalEntry,
    ExceptionResolutionWindowInput,
    ExceptionResolutionWindowResult,
    OntologyEvidenceSource,
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import (
    ExceptionCandidateWindow,
    ExceptionCitationResolution,
    LegalExceptionProposal,
)
from legal_rag.ontology_population.assembly_proposals import summarize_population_record
from legal_rag.ontology_population.evidence import fragment_evidence_sources
from legal_rag.ontology_population.extraction import bind_population_evidence
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity
from legal_rag.settings import Settings

ONTOLOGY_EXCEPTION_PROMPT_VERSION = "0.2.0"


class ExceptionProposalClient(Protocol):
    """Boundary implemented by a legal-exception language-model client."""

    def propose_batch(
        self, windows: Sequence[ExceptionResolutionWindowInput]
    ) -> Sequence[ExceptionResolutionWindowResult]:
        """Return typed exception proposals for candidate windows."""


class BamlExceptionProposalClient:
    """Typed BAML exception client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, exception_settings: ExceptionResolutionSettings) -> None:
        """Create a bounded runtime BAML client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=exception_settings.max_output_tokens,
            temperature=exception_settings.temperature,
            max_response_bytes=exception_settings.max_response_bytes,
        )

    def propose_batch(
        self, windows: Sequence[ExceptionResolutionWindowInput]
    ) -> Sequence[ExceptionResolutionWindowResult]:
        """Run one bounded BAML call over exception windows."""
        request = b.request.ProposeLegalExceptionResolutions(list(windows), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.ProposeLegalExceptionResolutions(raw_output, baml_options=self._caller.options)


def build_exception_proposal_run_id(
    candidate_run_id: str,
    exception_settings: ExceptionResolutionSettings,
    settings: Settings,
) -> str:
    """Fingerprint every input that can change semantic exception proposals."""
    return population_identity(
        {
            "candidate_run_id": candidate_run_id,
            "prompt_version": ONTOLOGY_EXCEPTION_PROMPT_VERSION,
            "batch_size": exception_settings.batch_size,
            "max_output_tokens": exception_settings.max_output_tokens,
            "temperature": exception_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_extra_request_body": dict(settings.llm_extra_request_body),
        }
    )


def propose_legal_exceptions(
    windows: Sequence[ExceptionCandidateWindow],
    corpus: ChangeCorpus,
    client: ExceptionProposalClient,
    exception_settings: ExceptionResolutionSettings,
    *,
    run_id: str,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[LegalExceptionProposal, ...]]]], None] | None = None,
) -> tuple[LegalExceptionProposal, ...]:
    """Generate exact-evidenced exception proposals in resumable batches."""
    documents_by_id = {document.document_id: document for document in corpus.documents}
    records_by_id = {record.population_id: record for document in corpus.documents for record in document.records}
    proposals = []
    for batch_start in range(0, len(windows), exception_settings.batch_size):
        batch = windows[batch_start : batch_start + exception_settings.batch_size]
        inputs = tuple(_window_input(window, documents_by_id, records_by_id) for window in batch)
        batch_results = execute_llm_task(
            _batch_task_key(inputs, run_id),
            partial(
                _build_batch_proposals,
                client,
                inputs,
                batch,
                documents_by_id,
                exception_settings,
                run_id,
            ),
            max_retries=exception_settings.max_retries,
            retry_delay_seconds=exception_settings.retry_delay_seconds,
        )
        for _, candidate_proposals in batch_results:
            proposals.extend(candidate_proposals)
        if on_batch_processed is not None:
            on_batch_processed(batch_results)
    return tuple(proposals)


def _build_batch_proposals(
    client: ExceptionProposalClient,
    inputs: Sequence[ExceptionResolutionWindowInput],
    windows: Sequence[ExceptionCandidateWindow],
    documents_by_id: Mapping[str, ChangeCorpusDocument],
    exception_settings: ExceptionResolutionSettings,
    run_id: str,
) -> tuple[tuple[str, tuple[LegalExceptionProposal, ...]], ...]:
    """Translate one typed response and bind source and target references."""
    results = client.propose_batch(inputs)
    requested_ids = [item.candidate_id for item in inputs]
    result_ids = [result.candidate_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Exception response must contain exactly one result per candidate window"
        raise ValueError(msg)
    results_by_id = {result.candidate_id: result for result in results}
    inputs_by_id = {item.candidate_id: item for item in inputs}
    windows_by_id = {window.candidate_id: window for window in windows}
    translated = []
    for candidate_id in requested_ids:
        window = windows_by_id[candidate_id]
        source_document = documents_by_id[window.source_document_id]
        evidence_sources = inputs_by_id[candidate_id].source_evidence_sources
        candidate_proposals = []
        for raw in results_by_id[candidate_id].proposals:
            if raw.general_population_id not in window.general_population_ids:
                msg = "Exception general rule must belong to the candidate window"
                raise ValueError(msg)
            evidence = bind_population_evidence(raw.evidence, evidence_sources)
            citations = _translate_citations(raw.citations, evidence_sources, window)
            proposal_id = population_identity(
                {
                    "run_id": run_id,
                    "general_population_id": raw.general_population_id,
                    "source_document_number": source_document.metadata.document_number,
                    "exception_provision_path": raw.exception_provision_path,
                    "evidence": [item.to_dict() for item in evidence],
                    "citations": [citation.to_dict() for citation in citations],
                }
            )
            candidate_proposals.append(
                LegalExceptionProposal(
                    proposal_id=proposal_id,
                    candidate_id=candidate_id,
                    general_population_id=raw.general_population_id,
                    source_document_number=source_document.metadata.document_number,
                    exception_provision_path=raw.exception_provision_path,
                    evidence=evidence,
                    citations=citations,
                    rationale_vi=raw.rationale_vi,
                    run_id=run_id,
                )
            )
        translated.append((candidate_id, tuple(candidate_proposals)))
    return tuple(translated)


def _translate_citations(
    raw_citations: Sequence[ExceptionCitationProposalEntry],
    evidence_sources: Sequence[OntologyEvidenceSource],
    window: ExceptionCandidateWindow,
) -> tuple[ExceptionCitationResolution, ...]:
    """Bind citation wording and target rule references."""
    citations = []
    for raw in raw_citations:
        bound = bind_population_evidence((raw.citation,), evidence_sources)
        target_ids = tuple(sorted(set(raw.target_population_ids)))
        if not set(target_ids) <= set(window.target_population_ids):
            msg = "Exception citation targets must belong to the candidate window"
            raise ValueError(msg)
        citations.append(
            ExceptionCitationResolution(
                citation_text=bound[0].exact_text,
                source_chunk_id=bound[0].chunk_id,
                target_population_ids=target_ids,
            )
        )
    return tuple(citations)


def _window_input(
    window: ExceptionCandidateWindow,
    documents_by_id: Mapping[str, ChangeCorpusDocument],
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> ExceptionResolutionWindowInput:
    """Build one complete typed BAML exception window."""
    source_document = documents_by_id[window.source_document_id]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in source_document.chunks}
    return ExceptionResolutionWindowInput(
        candidate_id=window.candidate_id,
        source_document_number=source_document.metadata.document_number,
        general_records=[
            summarize_population_record(records_by_id[population_id]) for population_id in window.general_population_ids
        ],
        target_records=[
            summarize_population_record(records_by_id[population_id]) for population_id in window.target_population_ids
        ],
        source_evidence_sources=[
            source
            for chunk_id in window.source_chunk_ids
            for source in fragment_evidence_sources(chunk_id, chunks_by_id[chunk_id].text)
        ],
    )


def _batch_task_key(inputs: Sequence[ExceptionResolutionWindowInput], run_id: str) -> str:
    """Return a deterministic retry key for one exception proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-exception:{digest}"
