"""Tests for two-phase, exact-evidenced exception proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    ExceptionCitationDiscoveryEntry,
    ExceptionDiscoveryWindowInput,
    ExceptionDiscoveryWindowResult,
    ExceptionTargetResolutionEntry,
    ExceptionTargetResolutionWindowInput,
    ExceptionTargetResolutionWindowResult,
    LegalExceptionDiscoveryEntry,
    OntologyEvidenceReference,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.candidates import build_exception_candidates
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import LegalExceptionProposal
from legal_rag.ontology_exception.proposals import build_exception_proposal_rerun_id, propose_legal_exceptions
from legal_rag.ontology_population.models import PopulationEvidence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ExceptionClient:
    """Return configurable discovery and resolution results without an external model."""

    model = "test-model"

    def __init__(
        self,
        exception_text: str,
        *,
        general_id: str = "general-population",
        target_id: str = "target-population",
        citation_text: str = "Điều 3",
        target_path: str = "Điều 3",
        empty: bool = False,
        use_general_evidence: bool = False,
    ) -> None:
        self._exception_text = exception_text
        self._general_id = general_id
        self._target_id = target_id
        self._citation_text = citation_text
        self._target_path = target_path
        self._empty = empty
        self._use_general_evidence = use_general_evidence
        self.discovery_inputs: tuple[ExceptionDiscoveryWindowInput, ...] = ()
        self.resolution_inputs: tuple[ExceptionTargetResolutionWindowInput, ...] = ()

    def discover_batch(
        self, windows: Sequence[ExceptionDiscoveryWindowInput]
    ) -> Sequence[ExceptionDiscoveryWindowResult]:
        """Capture source-local windows and return typed discoveries."""
        self.discovery_inputs = tuple(windows)
        results = []
        for window in windows:
            evidence_sources = (
                tuple(source for record in window.general_records for source in record.evidence)
                if self._use_general_evidence
                else tuple(window.source_evidence_sources)
            )
            exception_source = next((item for item in evidence_sources if self._exception_text in item.text), None)
            if self._empty:
                results.append(ExceptionDiscoveryWindowResult(candidate_id=window.candidate_id, discoveries=[]))
                continue
            citation_source = next(
                (item for item in evidence_sources if self._citation_text in item.text),
                None,
            )
            exception_reference = OntologyEvidenceReference(
                evidence_id=exception_source.evidence_id if exception_source else "ev_unknown"
            )
            if citation_source:
                citation_start = citation_source.text.index(self._citation_text)
                citation_reference = OntologyEvidenceReference(
                    evidence_id=citation_source.evidence_id,
                    start_char=citation_start,
                    end_char=citation_start + len(self._citation_text),
                )
            else:
                citation_reference = OntologyEvidenceReference(evidence_id="ev_unknown")
            results.append(
                ExceptionDiscoveryWindowResult(
                    candidate_id=window.candidate_id,
                    discoveries=[
                        LegalExceptionDiscoveryEntry(
                            general_population_id=self._general_id,
                            evidence=[exception_reference],
                            citations=[
                                ExceptionCitationDiscoveryEntry(
                                    citation=citation_reference,
                                    target_document_number="168/2024/NĐ-CP",
                                    target_provision_paths=[self._target_path],
                                )
                            ],
                            rationale_vi="Nguồn trực tiếp loại trừ trường hợp thuộc Điều 3.",
                        )
                    ],
                )
            )
        return tuple(results)

    def resolve_batch(
        self, windows: Sequence[ExceptionTargetResolutionWindowInput]
    ) -> Sequence[ExceptionTargetResolutionWindowResult]:
        """Capture locator-filtered records and select a configurable target."""
        self.resolution_inputs = tuple(windows)
        return tuple(
            ExceptionTargetResolutionWindowResult(
                candidate_id=window.candidate_id,
                resolutions=[
                    ExceptionTargetResolutionEntry(
                        proposal_index=candidate.proposal_index,
                        citation_index=candidate.citation_index,
                        target_population_ids=[self._target_id],
                    )
                    for candidate in window.candidates
                ],
            )
            for window in windows
        )


def _propose(corpus: ChangeCorpus, client: _ExceptionClient) -> tuple[LegalExceptionProposal, ...]:
    """Run one proposal pass with retries disabled."""
    settings = replace(
        ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_exception_candidates(corpus, settings)
    return propose_legal_exceptions(candidates.windows, corpus, client, settings, run_id="exception-run")


def test_proposal_discovers_locally_then_resolves_only_structural_targets(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Discovery excludes the target corpus and resolution sees only locator matches."""
    client = _ExceptionClient(exception_text)
    first = _propose(exception_corpus, client)
    second = _propose(exception_corpus, client)

    assert first == second
    assert first[0].evidence[0].exact_text == exception_text
    assert first[0].citations[0].citation_text == "Điều 3"
    assert first[0].citations[0].target_population_ids == ("target-population",)
    assert first[0].exception_provision_path == "Điều 6 khoản 1 điểm a"
    assert not hasattr(client.discovery_inputs[0], "target_records")


def test_resolution_input_contains_only_the_cited_locator(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Structural lookup excludes unrelated rules before semantic target selection."""
    client = _ExceptionClient(exception_text)

    _propose(exception_corpus, client)

    assert client.discovery_inputs[0].general_records[0].conduct_label_vi == "Không chấp hành biển báo"
    target_records = client.resolution_inputs[0].candidates[0].target_records
    assert [record.population_id for record in target_records] == ["target-population"]


@pytest.mark.parametrize(
    ("general_id", "target_id"),
    [("unknown", "target-population"), ("general-population", "unknown")],
)
def test_proposal_rejects_ids_outside_discovery_or_structural_lookup(
    general_id: str,
    target_id: str,
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """The model cannot choose arbitrary general or target rule identities."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(
            exception_corpus,
            _ExceptionClient(exception_text, general_id=general_id, target_id=target_id),
        )


def test_proposal_rejects_evidence_owned_by_another_general_rule(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """A nearby rule cannot borrow exception evidence from another general rule."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(
            exception_corpus,
            _ExceptionClient(exception_text, general_id="target-population"),
        )


@pytest.mark.parametrize(
    ("exception_value", "citation_value"),
    [("Nội dung ngoại lệ do mô hình diễn giải.", "Điều 3"), ("Trừ trường hợp quy định tại Điều 3.", "Điều ba")],
)
def test_proposal_rejects_paraphrased_exception_or_citation(
    exception_value: str,
    citation_value: str,
    exception_corpus: ChangeCorpus,
) -> None:
    """Neither exception evidence nor citation wording may be paraphrased."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(
            exception_corpus,
            _ExceptionClient(exception_value, citation_text=citation_value),
        )


def test_unresolved_locator_preserves_citation_without_hallucinated_target(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """A citation remains useful when its normalized locator is absent from the corpus."""
    client = _ExceptionClient(exception_text, target_path="Điều 999")

    proposals = _propose(exception_corpus, client)

    assert proposals[0].citations[0].target_population_ids == ()
    assert client.resolution_inputs == ()


def test_structural_locator_lookup_ignores_boundary_punctuation(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Equivalent locator components match despite source punctuation around a point label."""
    document = exception_corpus.documents[0]
    general, target = document.records
    corpus = ChangeCorpus(
        documents=(replace(document, records=(general, replace(target, provision_path="Điều 3 > Khoản 8 > Điểm d)"))),)
    )

    proposals = _propose(
        corpus,
        _ExceptionClient(exception_text, target_path="Điều 3 > Khoản 8 > Điểm d"),
    )

    assert proposals[0].citations[0].target_population_ids == ("target-population",)


def test_general_record_evidence_outside_source_window_remains_bindable(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Evidence exposed by a general rule remains valid when its chunk is outside the source window."""
    document = exception_corpus.documents[0]
    general, target = document.records
    external_evidence = PopulationEvidence(chunk_id="outside-source-window", exact_text=exception_text)
    corpus = ChangeCorpus(
        documents=(replace(document, records=(replace(general, evidence=(external_evidence,)), target)),)
    )

    proposals = _propose(corpus, _ExceptionClient(exception_text, use_general_evidence=True))

    assert proposals[0].evidence == (external_evidence,)
    assert proposals[0].citations[0].source_chunk_id == "outside-source-window"


def test_empty_exception_result_is_valid_and_checkpointed(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """A window without a confirmed exception completes with zero proposals."""
    processed: list[object] = []
    settings = replace(
        ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_exception_candidates(exception_corpus, settings)
    proposals = propose_legal_exceptions(
        candidates.windows,
        exception_corpus,
        _ExceptionClient(exception_text, empty=True),
        settings,
        run_id="exception-run",
        on_batch_processed=processed.extend,
    )

    assert proposals == ()
    assert processed == [(candidates.windows[0].candidate_id, ())]


def test_exception_rerun_id_is_stable_under_candidate_order() -> None:
    """A selective replay identity depends on the selected set, not JSON ordering."""
    assert build_exception_proposal_rerun_id("base-run", ("candidate-b", "candidate-a")) == (
        build_exception_proposal_rerun_id("base-run", ("candidate-a", "candidate-b"))
    )
