"""Tests for typed, exact-evidenced exception proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    ExceptionCitationProposalEntry,
    ExceptionResolutionWindowInput,
    ExceptionResolutionWindowResult,
    LegalExceptionProposalEntry,
    OntologyEvidenceEntry,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.candidates import build_exception_candidates
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.models import LegalExceptionProposal
from legal_rag.ontology_exception.proposals import propose_legal_exceptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ExceptionClient:
    """Return configurable typed exception results without an external model."""

    model = "test-model"

    def __init__(
        self,
        exception_text: str,
        *,
        general_id: str = "general-population",
        target_id: str = "target-population",
        citation_text: str = "Điều 3",
        empty: bool = False,
    ) -> None:
        self._exception_text = exception_text
        self._general_id = general_id
        self._target_id = target_id
        self._citation_text = citation_text
        self._empty = empty
        self.inputs: tuple[ExceptionResolutionWindowInput, ...] = ()

    def propose_batch(
        self, windows: Sequence[ExceptionResolutionWindowInput]
    ) -> Sequence[ExceptionResolutionWindowResult]:
        """Capture complete windows and return one typed result per candidate."""
        self.inputs = tuple(windows)
        return tuple(
            ExceptionResolutionWindowResult(
                candidate_id=window.candidate_id,
                proposals=[]
                if self._empty
                else [
                    LegalExceptionProposalEntry(
                        general_population_id=self._general_id,
                        exception_provision_path="Điều 6 khoản 1 điểm a",
                        evidence=[
                            OntologyEvidenceEntry(
                                chunk_id="general-chunk",
                                quoted_span=self._exception_text,
                            )
                        ],
                        citations=[
                            ExceptionCitationProposalEntry(
                                citation=OntologyEvidenceEntry(
                                    chunk_id="general-chunk",
                                    quoted_span=self._citation_text,
                                ),
                                target_population_ids=[self._target_id],
                            )
                        ],
                        rationale_vi="Source trực tiếp loại trừ trường hợp thuộc Điều 3.",
                    )
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


def test_proposal_preserves_rules_and_binds_exception_and_citation_exactly(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """The model sees complete rule versions and all accepted source wording is exact-bound."""
    client = _ExceptionClient(exception_text)
    first = _propose(exception_corpus, client)
    second = _propose(exception_corpus, client)

    assert first == second
    assert first[0].evidence[0].exact_text == exception_text
    assert first[0].citations[0].citation_text == "Điều 3"
    assert first[0].citations[0].target_population_ids == ("target-population",)
    assert client.inputs[0].general_records[0].conduct_label_vi == "Không chấp hành biển báo"
    assert client.inputs[0].target_records[1].provision_path == "Điều 3"


@pytest.mark.parametrize(
    ("general_id", "target_id"),
    [("unknown", "target-population"), ("general-population", "unknown")],
)
def test_proposal_rejects_rules_outside_the_candidate_window(
    general_id: str,
    target_id: str,
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """The model cannot choose arbitrary general or redirect rule identities."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(
            exception_corpus,
            _ExceptionClient(exception_text, general_id=general_id, target_id=target_id),
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
