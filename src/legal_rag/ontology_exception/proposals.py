"""Generate typed legal-exception proposals for mandatory human review."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ChangeSourceChunkInput,
    ExceptionCitationProposalEntry,
    ExceptionResolutionWindowInput,
    ExceptionResolutionWindowResult,
)
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import (
    ExceptionCandidateWindow,
    ExceptionCitationResolution,
    LegalExceptionProposal,
)
from legal_rag.ontology_population.assembly_proposals import summarize_population_record
from legal_rag.ontology_population.extraction import bind_population_evidence
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity
from legal_rag.settings import Settings


class ExceptionProposalClient(Protocol):
    """Boundary implemented by a legal-exception language-model client."""

    @property
    def model(self) -> str:
        """Return the model identifier recorded on proposals."""

    def propose_batch(
        self, windows: Sequence[ExceptionResolutionWindowInput]
    ) -> Sequence[ExceptionResolutionWindowResult]:
        """Return typed exception proposals for candidate windows."""


class BamlExceptionProposalClient:
    """Typed BAML exception client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, exception_settings: ExceptionResolutionSettings) -> None:
        """Create a bounded runtime BAML client."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=exception_settings.max_output_tokens,
            temperature=exception_settings.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = exception_settings.max_response_bytes
        self._model = settings.llm_model

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def propose_batch(
        self, windows: Sequence[ExceptionResolutionWindowInput]
    ) -> Sequence[ExceptionResolutionWindowResult]:
        """Run one bounded BAML call over exception windows."""
        request = b.request.ProposeLegalExceptionResolutions(
            list(windows), baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ProposeLegalExceptionResolutions(raw_output, baml_options={"client_registry": self._registry})


def build_exception_proposal_run_id(
    candidate_run_id: str,
    exception_settings: ExceptionResolutionSettings,
    settings: Settings,
) -> str:
    """Fingerprint every input that can change semantic exception proposals."""
    return population_identity(
        {
            "candidate_run_id": candidate_run_id,
            "prompt_version": exception_settings.prompt_version,
            "batch_size": exception_settings.batch_size,
            "max_output_tokens": exception_settings.max_output_tokens,
            "temperature": exception_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
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
    """Translate one typed response and enforce source and target membership."""
    results = client.propose_batch(inputs)
    requested_ids = [item.candidate_id for item in inputs]
    result_ids = [result.candidate_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Exception response must contain exactly one result per candidate window"
        raise ValueError(msg)
    results_by_id = {result.candidate_id: result for result in results}
    windows_by_id = {window.candidate_id: window for window in windows}
    translated = []
    for candidate_id in requested_ids:
        window = windows_by_id[candidate_id]
        source_document = documents_by_id[window.source_document_id]
        source_text_by_id = {
            chunk.chunk_id: chunk.text
            for chunk in source_document.chunks
            if chunk.chunk_id in set(window.source_chunk_ids)
        }
        used_exception_keys: set[tuple[object, ...]] = set()
        candidate_proposals = []
        for raw in results_by_id[candidate_id].proposals:
            if raw.general_population_id not in window.general_population_ids:
                msg = "Exception general rule must belong to the candidate window"
                raise ValueError(msg)
            evidence = bind_population_evidence(raw.evidence, source_text_by_id)
            if not evidence or not raw.citations:
                msg = "Exception proposals require exact evidence and at least one resolved citation"
                raise ValueError(msg)
            citations = _translate_citations(raw.citations, source_text_by_id, window)
            exception_key = (
                raw.general_population_id,
                raw.exception_provision_path,
                tuple((item.chunk_id, item.exact_text) for item in evidence),
            )
            if exception_key in used_exception_keys:
                msg = "One candidate result must not repeat the same exception statement and general rule"
                raise ValueError(msg)
            used_exception_keys.add(exception_key)
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
                    proposed_by_model=client.model,
                    prompt_version=exception_settings.prompt_version,
                    run_id=run_id,
                )
            )
        translated.append((candidate_id, tuple(candidate_proposals)))
    return tuple(translated)


def _translate_citations(
    raw_citations: Sequence[ExceptionCitationProposalEntry],
    source_text_by_id: Mapping[str, str],
    window: ExceptionCandidateWindow,
) -> tuple[ExceptionCitationResolution, ...]:
    """Bind citation wording exactly and validate every target rule candidate."""
    citations = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_citations:
        bound = bind_population_evidence((raw.citation,), source_text_by_id)
        target_ids = tuple(sorted(raw.target_population_ids))
        if not target_ids or len(target_ids) != len(set(target_ids)):
            msg = "Each grouped exception citation requires unique target rule versions"
            raise ValueError(msg)
        if not set(target_ids) <= set(window.target_population_ids):
            msg = "Exception citation targets must belong to the candidate window"
            raise ValueError(msg)
        citation_key = (bound[0].chunk_id, bound[0].exact_text)
        if citation_key in seen:
            msg = "One exception proposal must not repeat a grouped citation"
            raise ValueError(msg)
        seen.add(citation_key)
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
        source_chunks=[
            ChangeSourceChunkInput(
                chunk_id=chunk_id,
                text=chunks_by_id[chunk_id].text,
                source_name=chunks_by_id[chunk_id].source_name,
            )
            for chunk_id in window.source_chunk_ids
        ],
    )


def _batch_task_key(inputs: Sequence[ExceptionResolutionWindowInput], run_id: str) -> str:
    """Return a deterministic retry key for one exception proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-exception:{digest}"
