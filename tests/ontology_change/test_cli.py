"""End-to-end tests for the cross-document legal-change operator workflow."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag import (
    ontology_change_candidates_cli,
    ontology_change_finalize_cli,
    ontology_change_proposals_cli,
)
from legal_rag.baml_client.types import (
    CrossDocumentChangeWindowInput,
    CrossDocumentChangeWindowResult,
    LegalChangeProposalEntry,
    OntologyEvidenceEntry,
    OntologyLegalChangeType,
)
from legal_rag.ontology_change import proposals as change_proposals
from legal_rag.ontology_change.corpus import ChangeCorpusDocument
from legal_rag.ontology_change.models import ChangeReview
from legal_rag.ontology_change.writer import (
    read_change_proposals,
    read_change_reviews,
    read_legal_change_events,
    write_change_reviews,
)
from legal_rag.ontology_population.writer import read_population_records

from .conftest import write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliChangeClient:
    """Return one typed amendment and track resumable LLM calls."""

    calls = 0
    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept the production client's constructor arguments."""

    def propose_batch(
        self, windows: Sequence[CrossDocumentChangeWindowInput]
    ) -> Sequence[CrossDocumentChangeWindowResult]:
        """Return one exact-evidenced amendment for each fixture window."""
        type(self).calls += 1
        return tuple(
            CrossDocumentChangeWindowResult(
                candidate_id=window.candidate_id,
                proposals=[
                    LegalChangeProposalEntry(
                        earlier_population_id=window.earlier_records[0].population_id,
                        later_population_id=window.later_records[0].population_id,
                        change_type=OntologyLegalChangeType.Amendment,
                        effective_on="2026-08-15",
                        change_provision_path="Điều 3",
                        evidence=[
                            OntologyEvidenceEntry(
                                chunk_id=window.change_source_chunks[0].chunk_id,
                                quoted_span="Sửa đổi điểm b khoản 8 Điều 13, có hiệu lực từ ngày 15 tháng 8 năm 2026.",
                            )
                        ],
                        rationale_vi="Điều 3 trực tiếp sửa đổi rule trước.",
                    )
                ],
            )
            for window in windows
        )


def test_cli_runs_candidate_proposal_review_and_offline_change_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """The three commands resume model work and write only reviewed SHACL-valid outputs."""
    manifest_path = tmp_path / "corpus.json"
    candidates_path = tmp_path / "candidates.json"
    proposals_path = tmp_path / "proposals.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    canonical_path = tmp_path / "canonical.jsonl"
    events_path = tmp_path / "events.jsonl"
    graph_path = tmp_path / "canonical.ttl"
    write_manifest(manifest_path, change_documents)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-change-candidates",
            str(manifest_path),
            str(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
            str(candidates_path),
        ],
    )
    ontology_change_candidates_cli.main()

    _CliChangeClient.calls = 0
    monkeypatch.setattr(change_proposals, "BamlChangeProposalClient", _CliChangeClient)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    proposal_argv = [
        "legal-propose-legal-changes",
        str(manifest_path),
        str(candidates_path),
        str(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        str(proposals_path),
        str(progress_path),
        str(reviews_path),
    ]
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_change_proposals_cli.main()
    ontology_change_proposals_cli.main()

    proposal = read_change_proposals(proposals_path)[0]
    assert _CliChangeClient.calls == 1
    assert read_change_reviews(reviews_path)[0].decision == "pending"
    finalize_argv = [
        "legal-finalize-legal-changes",
        str(manifest_path),
        str(proposals_path),
        str(reviews_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        str(canonical_path),
        str(events_path),
        str(graph_path),
    ]
    monkeypatch.setattr(sys, "argv", finalize_argv)
    with pytest.raises(ValueError, match="Pending legal-change reviews"):
        ontology_change_finalize_cli.main()
    assert not canonical_path.exists()
    assert not events_path.exists()
    assert not graph_path.exists()

    write_change_reviews(
        reviews_path,
        (
            ChangeReview(
                proposal_id=proposal.proposal_id,
                decision="accepted",
                reviewer="legal-reviewer",
                reviewed_on=date(2026, 8, 18),
                note="Đã đối chiếu source.",
            ),
        ),
    )
    monkeypatch.setattr(sys, "argv", finalize_argv)
    ontology_change_finalize_cli.main()

    assert len(read_population_records(canonical_path)) == 2
    assert len(read_legal_change_events(events_path)) == 1
    assert len(Graph().parse(graph_path, format="turtle")) > 0
