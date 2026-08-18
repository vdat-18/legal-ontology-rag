"""End-to-end tests for the legal-exception operator workflow."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag import (
    ontology_exception_candidates_cli,
    ontology_exception_finalize_cli,
    ontology_exception_proposals_cli,
)
from legal_rag.baml_client.types import (
    ExceptionCitationProposalEntry,
    ExceptionResolutionWindowInput,
    ExceptionResolutionWindowResult,
    LegalExceptionProposalEntry,
    OntologyEvidenceEntry,
)
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.writer import write_legal_change_events
from legal_rag.ontology_exception import proposals as exception_proposals
from legal_rag.ontology_exception.models import ExceptionReview
from legal_rag.ontology_exception.writer import (
    read_exception_proposals,
    read_exception_reviews,
    read_legal_exception_records,
    write_exception_reviews,
)

from .conftest import write_exception_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliExceptionClient:
    """Return one typed exception proposal and track resumable LLM calls."""

    calls = 0
    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept the production client's constructor arguments."""

    def propose_batch(
        self, windows: Sequence[ExceptionResolutionWindowInput]
    ) -> Sequence[ExceptionResolutionWindowResult]:
        """Return one exact-evidenced exception for each fixture window."""
        type(self).calls += 1
        return tuple(
            ExceptionResolutionWindowResult(
                candidate_id=window.candidate_id,
                proposals=[
                    LegalExceptionProposalEntry(
                        general_population_id="general-population",
                        exception_provision_path="Điều 6 khoản 1 điểm a",
                        evidence=[
                            OntologyEvidenceEntry(
                                chunk_id="general-chunk",
                                quoted_span="Trừ trường hợp quy định tại Điều 3.",
                            )
                        ],
                        citations=[
                            ExceptionCitationProposalEntry(
                                citation=OntologyEvidenceEntry(
                                    chunk_id="general-chunk",
                                    quoted_span="Điều 3",
                                ),
                                target_population_ids=["target-population"],
                            )
                        ],
                        rationale_vi="Source trực tiếp loại trừ và dẫn tới Điều 3.",
                    )
                ],
            )
            for window in windows
        )


def test_cli_runs_candidate_proposal_review_and_offline_exception_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """The three commands resume model work and write only reviewed SHACL-valid outputs."""
    manifest_path = tmp_path / "corpus.json"
    canonical_path = write_exception_manifest(manifest_path, exception_corpus)
    change_events_path = tmp_path / "change-events.jsonl"
    candidates_path = tmp_path / "candidates.json"
    proposals_path = tmp_path / "proposals.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    exceptions_path = tmp_path / "exceptions.jsonl"
    graph_path = tmp_path / "canonical.ttl"
    write_legal_change_events(change_events_path, ())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-exception-candidates",
            str(manifest_path),
            str(canonical_path),
            str(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
            str(candidates_path),
        ],
    )
    ontology_exception_candidates_cli.main()

    _CliExceptionClient.calls = 0
    monkeypatch.setattr(exception_proposals, "BamlExceptionProposalClient", _CliExceptionClient)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    proposal_argv = [
        "legal-propose-legal-exceptions",
        str(manifest_path),
        str(canonical_path),
        str(candidates_path),
        str(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        str(proposals_path),
        str(progress_path),
        str(reviews_path),
    ]
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_exception_proposals_cli.main()
    ontology_exception_proposals_cli.main()

    proposal = read_exception_proposals(proposals_path)[0]
    assert _CliExceptionClient.calls == 1
    assert read_exception_reviews(reviews_path)[0].decision == "pending"
    finalize_argv = [
        "legal-finalize-legal-exceptions",
        str(manifest_path),
        str(canonical_path),
        str(change_events_path),
        str(proposals_path),
        str(reviews_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml"),
        str(exceptions_path),
        str(graph_path),
    ]
    monkeypatch.setattr(sys, "argv", finalize_argv)
    with pytest.raises(ValueError, match="Pending exception reviews"):
        ontology_exception_finalize_cli.main()
    assert not exceptions_path.exists()
    assert not graph_path.exists()

    write_exception_reviews(
        reviews_path,
        (
            ExceptionReview(
                proposal_id=proposal.proposal_id,
                decision="accepted",
                reviewer="legal-reviewer",
                reviewed_on=date(2026, 8, 18),
                note="Đã đối chiếu source và target.",
            ),
        ),
    )
    monkeypatch.setattr(sys, "argv", finalize_argv)
    ontology_exception_finalize_cli.main()

    assert len(read_legal_exception_records(exceptions_path)) == 1
    assert len(Graph().parse(graph_path, format="turtle")) > 0
