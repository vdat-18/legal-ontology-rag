"""Tests for typed, exact-evidenced legal-change proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    CrossDocumentChangeWindowInput,
    CrossDocumentChangeWindowResult,
    LegalChangeProposalEntry,
    OntologyEvidenceReference,
    OntologyLegalChangeType,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.ontology_change.candidates import build_change_candidates
from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.proposals import propose_legal_changes

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ChangeClient:
    """Return configurable typed legal-change results without an external model."""

    model = "test-model"

    def __init__(
        self,
        evidence_text: str,
        *,
        earlier_id: str = "earlier-population",
        later_id: str | None = "later-population",
        empty: bool = False,
    ) -> None:
        self._evidence_text = evidence_text
        self._earlier_id = earlier_id
        self._later_id = later_id
        self._empty = empty
        self.inputs: tuple[CrossDocumentChangeWindowInput, ...] = ()

    def propose_batch(
        self, windows: Sequence[CrossDocumentChangeWindowInput]
    ) -> Sequence[CrossDocumentChangeWindowResult]:
        """Capture complete windows and return one result per candidate."""
        self.inputs = tuple(windows)
        results = []
        for window in windows:
            source = next(
                (item for item in window.change_evidence_sources if item.text == self._evidence_text),
                None,
            )
            evidence_id = source.evidence_id if source else "ev_unknown"
            results.append(
                CrossDocumentChangeWindowResult(
                    candidate_id=window.candidate_id,
                    proposals=[]
                    if self._empty
                    else [
                        LegalChangeProposalEntry(
                            earlier_population_id=self._earlier_id,
                            later_population_id=self._later_id,
                            change_type=OntologyLegalChangeType.Amendment,
                            effective_on="2026-08-15",
                            change_provision_path="Điều 3",
                            evidence=[OntologyEvidenceReference(evidence_id=evidence_id)],
                            rationale_vi="Văn bản sau trực tiếp sửa đổi đúng điều khoản của rule trước.",
                        )
                    ],
                )
            )
        return tuple(results)


def _propose(corpus: ChangeCorpus, client: _ChangeClient) -> tuple:
    """Run one proposal pass with retries disabled."""
    settings = replace(
        ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_change_candidates(corpus, settings)
    return propose_legal_changes(candidates.windows, corpus, client, settings, run_id="change-run")


def test_proposal_preserves_rule_semantics_and_binds_exact_change_evidence(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """The LLM receives compact rule semantics and output is rebound to authoritative text."""
    client = _ChangeClient(change_evidence_text)
    first = _propose(change_corpus, client)
    second = _propose(change_corpus, client)

    assert first == second
    assert first[0].effective_on.isoformat() == "2026-08-15"
    assert first[0].evidence[0].exact_text == change_evidence_text
    assert client.inputs[0].earlier_records[0].consequences[0].minimum_amount == "3000000"
    assert client.inputs[0].later_records[0].consequences[0].maximum_amount == "6000000"
    assert "evidence" not in client.inputs[0].earlier_records[0].model_dump()
    assert "evidence" not in client.inputs[0].earlier_records[0].consequences[0].model_dump()
    assert change_evidence_text in {item.text for item in client.inputs[0].change_evidence_sources}


@pytest.mark.parametrize(
    ("earlier_id", "later_id"),
    [("unknown", "later-population"), ("earlier-population", "unknown")],
)
def test_proposal_rejects_members_outside_the_candidate_window(
    earlier_id: str,
    later_id: str,
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """The model cannot link arbitrary rule identities outside its bounded input."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(
            change_corpus,
            _ChangeClient(change_evidence_text, earlier_id=earlier_id, later_id=later_id),
        )


def test_proposal_rejects_paraphrased_change_evidence(change_corpus: ChangeCorpus) -> None:
    """A model paraphrase cannot support a canonical legal-change event."""
    with pytest.raises(LlmTaskExhaustedError) as error:
        _propose(change_corpus, _ChangeClient("Nội dung sửa đổi do mô hình diễn giải."))

    assert "Unknown evidence reference" in str(error.value.__cause__)


def test_empty_change_result_is_valid_and_checkpointed(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """No explicit change evidence is a valid zero-proposal result that still completes."""
    processed: list[object] = []
    settings = replace(
        ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_change_candidates(change_corpus, settings)
    proposals = propose_legal_changes(
        candidates.windows,
        change_corpus,
        _ChangeClient(change_evidence_text, empty=True),
        settings,
        run_id="change-run",
        on_batch_processed=processed.extend,
    )

    assert proposals == ()
    assert processed == [(candidates.windows[0].candidate_id, ())]
