"""Discover legal exceptions and resolve their cited rule versions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ExceptionDiscoveryWindowInput,
    ExceptionDiscoveryWindowResult,
    ExceptionSourceContext,
    ExceptionTargetCandidateInput,
    ExceptionTargetResolutionWindowInput,
    ExceptionTargetResolutionWindowResult,
    LegalExceptionDiscoveryEntry,
    OntologyEvidenceSource,
    OntologyPopulationSummary,
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import (
    ExceptionCandidateWindow,
    ExceptionCitationResolution,
    LegalExceptionProposal,
)
from legal_rag.ontology_population.assembly_proposals import summarize_population_record
from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationEvidence, population_identity
from legal_rag.ontology_proposal_execution import execute_proposal_batches
from legal_rag.settings import Settings

ONTOLOGY_EXCEPTION_PROMPT_VERSION = "0.14.0"


@dataclass(frozen=True, slots=True)
class _DiscoveredCitation:
    citation_text: str
    source_chunk_id: str
    target_document_number: str
    target_provision_paths: tuple[str, ...]
    target_records: tuple[OntologyPopulationRecord, ...]


@dataclass(frozen=True, slots=True)
class _DiscoveredException:
    general_population_id: str
    exception_provision_path: str
    evidence: tuple[PopulationEvidence, ...]
    citations: tuple[_DiscoveredCitation, ...]
    rationale_vi: str


class ExceptionProposalClient(Protocol):
    """Boundary implemented by a two-phase legal-exception model client."""

    def discover_batch(
        self, windows: Sequence[ExceptionDiscoveryWindowInput]
    ) -> Sequence[ExceptionDiscoveryWindowResult]:
        """Discover source-supported exceptions and normalized citation locators."""

    def resolve_batch(
        self, windows: Sequence[ExceptionTargetResolutionWindowInput]
    ) -> Sequence[ExceptionTargetResolutionWindowResult]:
        """Select cited rule versions from structurally retrieved targets."""


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

    def discover_batch(
        self, windows: Sequence[ExceptionDiscoveryWindowInput]
    ) -> Sequence[ExceptionDiscoveryWindowResult]:
        """Run source-local exception discovery."""
        request = b.request.DiscoverLegalExceptions(list(windows), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.DiscoverLegalExceptions(raw_output, baml_options=self._caller.options)

    def resolve_batch(
        self, windows: Sequence[ExceptionTargetResolutionWindowInput]
    ) -> Sequence[ExceptionTargetResolutionWindowResult]:
        """Resolve discovered citations against structurally retrieved targets."""
        request = b.request.ResolveLegalExceptionTargets(list(windows), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.ResolveLegalExceptionTargets(raw_output, baml_options=self._caller.options)


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


def build_exception_proposal_rerun_id(base_run_id: str, candidate_ids: Sequence[str]) -> str:
    """Fingerprint one explicit candidate replay without invalidating retained proposals."""
    return population_identity(
        {
            "stage": "exception-proposal-selective-rerun",
            "base_run_id": base_run_id,
            "candidate_ids": sorted(candidate_ids),
        }
    )


def propose_legal_exceptions(
    windows: Sequence[ExceptionCandidateWindow],
    corpus: ChangeCorpus,
    client: ExceptionProposalClient,
    exception_settings: ExceptionResolutionSettings,
    *,
    run_id: str,
    max_concurrency: int = 1,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[LegalExceptionProposal, ...]]]], None] | None = None,
) -> tuple[LegalExceptionProposal, ...]:
    """Discover and resolve exact-evidenced exceptions in concurrent batches."""
    documents_by_id = {document.document_id: document for document in corpus.documents}
    records_by_id = {record.population_id: record for document in corpus.documents for record in document.records}
    tasks = []
    for batch_start in range(0, len(windows), exception_settings.batch_size):
        batch = windows[batch_start : batch_start + exception_settings.batch_size]
        inputs = tuple(_discovery_input(window, corpus, documents_by_id, records_by_id) for window in batch)
        tasks.append(
            (
                _batch_task_key(inputs, run_id),
                partial(
                    _build_batch_proposals,
                    client,
                    inputs,
                    batch,
                    corpus,
                    documents_by_id,
                    records_by_id,
                    run_id,
                ),
            )
        )
    batch_results = execute_proposal_batches(
        tasks,
        stage="ontology legal-exception resolution",
        max_concurrency=max_concurrency,
        max_retries=exception_settings.max_retries,
        retry_delay_seconds=exception_settings.retry_delay_seconds,
        on_batch_processed=on_batch_processed,
    )
    return tuple(
        proposal
        for batch_result in batch_results
        for _, candidate_proposals in batch_result
        for proposal in candidate_proposals
    )


def _build_batch_proposals(
    client: ExceptionProposalClient,
    inputs: Sequence[ExceptionDiscoveryWindowInput],
    windows: Sequence[ExceptionCandidateWindow],
    corpus: ChangeCorpus,
    documents_by_id: Mapping[str, ChangeCorpusDocument],
    records_by_id: Mapping[str, OntologyPopulationRecord],
    run_id: str,
) -> tuple[tuple[str, tuple[LegalExceptionProposal, ...]], ...]:
    """Run both semantic phases and bind their typed results."""
    raw_results = client.discover_batch(inputs)
    requested_ids = tuple(item.candidate_id for item in inputs)
    _require_exact_result_ids(requested_ids, tuple(result.candidate_id for result in raw_results), "discovery")
    results_by_id = {result.candidate_id: result for result in raw_results}
    inputs_by_id = {item.candidate_id: item for item in inputs}
    windows_by_id = {window.candidate_id: window for window in windows}
    discoveries_by_id = {
        candidate_id: _bind_discoveries(
            results_by_id[candidate_id].discoveries,
            inputs_by_id[candidate_id],
            windows_by_id[candidate_id],
            corpus,
        )
        for candidate_id in requested_ids
    }
    resolution_inputs = tuple(
        resolution_input
        for candidate_id in requested_ids
        if (resolution_input := _resolution_input(candidate_id, discoveries_by_id[candidate_id], records_by_id))
        is not None
    )
    resolved_targets = _resolve_targets(client, resolution_inputs)
    translated = []
    for candidate_id in requested_ids:
        source_document = documents_by_id[windows_by_id[candidate_id].source_document_id]
        proposals = tuple(
            proposal
            for proposal_index, discovery in enumerate(discoveries_by_id[candidate_id])
            if (
                proposal := _proposal(
                    candidate_id,
                    proposal_index,
                    discovery,
                    source_document.metadata.document_number,
                    resolved_targets,
                    run_id,
                )
            )
            is not None
        )
        translated.append((candidate_id, proposals))
    return tuple(translated)


def _bind_discoveries(
    raw_discoveries: Sequence[LegalExceptionDiscoveryEntry],
    discovery_input: ExceptionDiscoveryWindowInput,
    window: ExceptionCandidateWindow,
    corpus: ChangeCorpus,
) -> tuple[_DiscoveredException, ...]:
    """Bind exact source evidence and structurally retrieve cited records."""
    discoveries = []
    for raw in raw_discoveries:
        if raw.general_population_id not in window.general_population_ids:
            msg = "Exception general rule must belong to the discovery window"
            raise ValueError(msg)
        general_record = next(
            record for record in discovery_input.general_records if record.population_id == raw.general_population_id
        )
        evidence_sources = _general_evidence_sources(general_record)
        sources_by_id = {source.evidence_id: source for source in evidence_sources}
        evidence = _bind_whole_evidence(raw.evidence_source_ids, sources_by_id)
        citations = []
        for citation in raw.citations:
            source = sources_by_id.get(citation.source_evidence_id)
            if source is None:
                msg = "Exception citation evidence must belong to the selected general rule"
                raise ValueError(msg)
            if not citation.citation_text.strip():
                msg = "Exception citation text must not be empty"
                raise ValueError(msg)
            target_paths = tuple(dict.fromkeys(citation.target_provision_paths))
            citations.append(
                _DiscoveredCitation(
                    citation_text=citation.citation_text,
                    source_chunk_id=source.chunk_id,
                    target_document_number=citation.target_document_number,
                    target_provision_paths=target_paths,
                    target_records=_records_at_locators(corpus, citation.target_document_number, target_paths),
                )
            )
        discoveries.append(
            _DiscoveredException(
                general_population_id=raw.general_population_id,
                exception_provision_path=general_record.provision_path,
                evidence=evidence,
                citations=tuple(citations),
                rationale_vi=raw.rationale_vi,
            )
        )
    return tuple(discoveries)


def _general_evidence_sources(
    general_record: OntologyPopulationSummary,
) -> tuple[OntologyEvidenceSource, ...]:
    """Collect evidence supplied for one selected general rule."""
    sources_by_id: dict[str, OntologyEvidenceSource] = {}
    for source in _summary_evidence_sources(general_record):
        existing = sources_by_id.get(source.evidence_id)
        if existing is not None and (existing.chunk_id, existing.text) != (source.chunk_id, source.text):
            msg = f"Evidence source ID collision: {source.evidence_id}"
            raise ValueError(msg)
        sources_by_id[source.evidence_id] = source
    return tuple(sources_by_id.values())


def _bind_whole_evidence(
    evidence_source_ids: Sequence[str],
    sources_by_id: Mapping[str, OntologyEvidenceSource],
) -> tuple[PopulationEvidence, ...]:
    """Bind complete canonical evidence sources without model-calculated offsets."""
    if not evidence_source_ids:
        msg = "A legal exception must select at least one evidence source"
        raise ValueError(msg)
    if len(evidence_source_ids) != len(set(evidence_source_ids)):
        msg = "Exception evidence source IDs must be unique"
        raise ValueError(msg)
    unknown_ids = set(evidence_source_ids) - set(sources_by_id)
    if unknown_ids:
        msg = "Exception evidence must belong to the selected general rule"
        raise ValueError(msg)
    return tuple(
        PopulationEvidence(chunk_id=sources_by_id[source_id].chunk_id, exact_text=sources_by_id[source_id].text)
        for source_id in evidence_source_ids
    )


def _summary_evidence_sources(record: OntologyPopulationSummary) -> tuple[OntologyEvidenceSource, ...]:
    """Return top-level and nested evidence exposed by one canonical rule summary."""
    return (
        *record.evidence,
        *(source for condition in record.conditions for source in condition.evidence),
        *(source for consequence in record.consequences for source in consequence.evidence),
    )


def _resolution_input(
    candidate_id: str,
    discoveries: Sequence[_DiscoveredException],
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> ExceptionTargetResolutionWindowInput | None:
    """Build one target-selection input, omitting citations with no structural match."""
    candidates = [
        ExceptionTargetCandidateInput(
            proposal_index=proposal_index,
            citation_index=citation_index,
            general_record=summarize_population_record(records_by_id[discovery.general_population_id]),
            exception_provision_path=discovery.exception_provision_path,
            citation_text=citation.citation_text,
            target_document_number=citation.target_document_number,
            target_provision_paths=list(citation.target_provision_paths),
            target_records=[summarize_population_record(record) for record in citation.target_records],
        )
        for proposal_index, discovery in enumerate(discoveries)
        for citation_index, citation in enumerate(discovery.citations)
        if citation.target_records
    ]
    if not candidates:
        return None
    return ExceptionTargetResolutionWindowInput(candidate_id=candidate_id, candidates=candidates)


def _resolve_targets(
    client: ExceptionProposalClient,
    inputs: Sequence[ExceptionTargetResolutionWindowInput],
) -> dict[tuple[str, int, int], tuple[str, ...]]:
    """Resolve target IDs and enforce membership in structural candidates."""
    if not inputs:
        return {}
    results = client.resolve_batch(inputs)
    requested_ids = tuple(item.candidate_id for item in inputs)
    _require_exact_result_ids(requested_ids, tuple(result.candidate_id for result in results), "target resolution")
    inputs_by_id = {item.candidate_id: item for item in inputs}
    resolved = {}
    for result in results:
        candidates = {
            (candidate.proposal_index, candidate.citation_index): candidate
            for candidate in inputs_by_id[result.candidate_id].candidates
        }
        result_keys = [(item.proposal_index, item.citation_index) for item in result.resolutions]
        if len(result_keys) != len(set(result_keys)) or set(result_keys) != set(candidates):
            msg = "Target response must contain exactly one resolution per discovered citation"
            raise ValueError(msg)
        for item in result.resolutions:
            key = (item.proposal_index, item.citation_index)
            allowed_ids = {record.population_id for record in candidates[key].target_records}
            target_ids = tuple(sorted(set(item.target_population_ids)))
            if not set(target_ids) <= allowed_ids:
                msg = "Exception citation targets must belong to the structural locator result"
                raise ValueError(msg)
            resolved[(result.candidate_id, *key)] = target_ids
    return resolved


def _proposal(
    candidate_id: str,
    proposal_index: int,
    discovery: _DiscoveredException,
    source_document_number: str,
    resolved_targets: Mapping[tuple[str, int, int], tuple[str, ...]],
    run_id: str,
) -> LegalExceptionProposal | None:
    """Build one stable final proposal from a bound discovery."""
    citations = tuple(
        ExceptionCitationResolution(
            citation_text=citation.citation_text,
            source_chunk_id=citation.source_chunk_id,
            target_population_ids=resolved_targets.get((candidate_id, proposal_index, citation_index), ()),
        )
        for citation_index, citation in enumerate(discovery.citations)
        if not citation.target_records or resolved_targets.get((candidate_id, proposal_index, citation_index), ())
    )
    if discovery.citations and not citations:
        return None
    proposal_id = population_identity(
        {
            "run_id": run_id,
            "general_population_id": discovery.general_population_id,
            "source_document_number": source_document_number,
            "exception_provision_path": discovery.exception_provision_path,
            "evidence": [item.to_dict() for item in discovery.evidence],
            "citations": [citation.to_dict() for citation in citations],
        }
    )
    return LegalExceptionProposal(
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        general_population_id=discovery.general_population_id,
        source_document_number=source_document_number,
        exception_provision_path=discovery.exception_provision_path,
        evidence=discovery.evidence,
        citations=citations,
        rationale_vi=discovery.rationale_vi,
        run_id=run_id,
    )


def _records_at_locators(
    corpus: ChangeCorpus,
    document_number: str,
    provision_paths: Sequence[str],
) -> tuple[OntologyPopulationRecord, ...]:
    """Retrieve records by exact document identity and hierarchical locator components."""
    normalized_document_number = _normalized_text(document_number)
    normalized_paths = tuple(_locator_parts(path) for path in provision_paths if _locator_parts(path))
    records = (
        record
        for document in corpus.documents
        if _normalized_text(document.metadata.document_number) == normalized_document_number
        for record in document.records
        if any(_contains_locator(_locator_parts(record.provision_path), path) for path in normalized_paths)
    )
    return tuple(sorted(records, key=lambda record: record.population_id))


def _contains_locator(canonical: tuple[str, ...], target: tuple[str, ...]) -> bool:
    """Return whether an exact target locator occurs in the canonical hierarchy."""
    return any(canonical[start : start + len(target)] == target for start in range(len(canonical) - len(target) + 1))


def _locator_parts(value: str) -> tuple[str, ...]:
    """Normalize explicit hierarchy separators without interpreting legal text."""
    return tuple(_normalized_locator_part(part) for part in value.split(">") if part.strip())


def _normalized_locator_part(value: str) -> str:
    """Normalize punctuation around one explicit locator component."""
    return _normalized_text(value).strip("()[]{}.,;:")


def _normalized_text(value: str) -> str:
    """Normalize insignificant whitespace and case for exact structural lookup."""
    return " ".join(value.split()).casefold()


def _discovery_input(
    window: ExceptionCandidateWindow,
    corpus: ChangeCorpus,
    documents_by_id: Mapping[str, ChangeCorpusDocument],
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> ExceptionDiscoveryWindowInput:
    """Build one complete source-local discovery input without target-corpus records."""
    source_document = documents_by_id[window.source_document_id]
    general_records = [
        summarize_population_record(records_by_id[population_id]) for population_id in window.general_population_ids
    ]
    evidence_chunk_ids = {source.chunk_id for record in general_records for source in _summary_evidence_sources(record)}
    return ExceptionDiscoveryWindowInput(
        candidate_id=window.candidate_id,
        source_document_number=source_document.metadata.document_number,
        comparison_document_numbers=[
            documents_by_id[document_id].metadata.document_number
            for document_id in (source_document.comparison_document_ids or ())
        ],
        corpus_document_numbers=sorted({document.metadata.document_number for document in corpus.documents}),
        source_contexts=[
            ExceptionSourceContext(chunk_id=chunk.chunk_id, text=chunk.text)
            for chunk in source_document.chunks
            if chunk.chunk_id in evidence_chunk_ids
        ],
        general_records=general_records,
    )


def _require_exact_result_ids(requested_ids: Sequence[str], result_ids: Sequence[str], phase: str) -> None:
    """Require one result for every requested window in a semantic phase."""
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = f"Exception {phase} response must contain exactly one result per candidate window"
        raise ValueError(msg)


def _batch_task_key(inputs: Sequence[ExceptionDiscoveryWindowInput], run_id: str) -> str:
    """Return a deterministic retry key for one exception proposal batch."""
    candidate_ids = "\0".join(item.candidate_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{candidate_ids}".encode()).hexdigest()
    return f"ontology-exception:{digest}"
