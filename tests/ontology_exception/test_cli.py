"""End-to-end tests for the legal-exception operator workflow."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag import (
    ontology_exception_candidates_cli,
    ontology_exception_finalize_cli,
    ontology_exception_proposals_cli,
)
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
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.writer import write_legal_change_events
from legal_rag.ontology_exception import proposals as exception_proposals
from legal_rag.ontology_exception.writer import (
    read_exception_progress,
    read_exception_proposals,
    read_legal_exception_records,
)

from .conftest import write_exception_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliExceptionClient:
    """Return one typed exception proposal and track resumable LLM calls."""

    calls = 0
    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept the production client's constructor arguments."""

    def discover_batch(
        self, windows: Sequence[ExceptionDiscoveryWindowInput]
    ) -> Sequence[ExceptionDiscoveryWindowResult]:
        """Return one exact-evidenced discovery for each fixture window."""
        type(self).calls += 1
        results = []
        for window in windows:
            source = next(
                item for item in window.source_evidence_sources if item.text == "Trừ trường hợp quy định tại Điều 3."
            )
            citation_text = "Điều 3"
            citation_start = source.text.index(citation_text)
            results.append(
                ExceptionDiscoveryWindowResult(
                    candidate_id=window.candidate_id,
                    discoveries=[
                        LegalExceptionDiscoveryEntry(
                            general_population_id="general-population",
                            evidence=[OntologyEvidenceReference(evidence_id=source.evidence_id)],
                            citations=[
                                ExceptionCitationDiscoveryEntry(
                                    citation=OntologyEvidenceReference(
                                        evidence_id=source.evidence_id,
                                        start_char=citation_start,
                                        end_char=citation_start + len(citation_text),
                                    ),
                                    target_document_number="168/2024/NĐ-CP",
                                    target_provision_paths=["Điều 3"],
                                )
                            ],
                            rationale_vi="Source trực tiếp loại trừ và dẫn tới Điều 3.",
                        )
                    ],
                )
            )
        return tuple(results)

    def resolve_batch(
        self, windows: Sequence[ExceptionTargetResolutionWindowInput]
    ) -> Sequence[ExceptionTargetResolutionWindowResult]:
        """Resolve each structurally retrieved target candidate."""
        return tuple(
            ExceptionTargetResolutionWindowResult(
                candidate_id=window.candidate_id,
                resolutions=[
                    ExceptionTargetResolutionEntry(
                        proposal_index=candidate.proposal_index,
                        citation_index=candidate.citation_index,
                        target_population_ids=["target-population"],
                    )
                    for candidate in window.candidates
                ],
            )
            for window in windows
        )


class _EmptyExceptionClient:
    """Return a completed candidate result with no legal-exception proposal."""

    calls = 0
    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept production constructor arguments."""

    def discover_batch(
        self, windows: Sequence[ExceptionDiscoveryWindowInput]
    ) -> Sequence[ExceptionDiscoveryWindowResult]:
        """Return an explicit no-proposal result for every replayed window."""
        type(self).calls += 1
        return tuple(
            ExceptionDiscoveryWindowResult(candidate_id=window.candidate_id, discoveries=[]) for window in windows
        )

    def resolve_batch(
        self, windows: Sequence[ExceptionTargetResolutionWindowInput]
    ) -> Sequence[ExceptionTargetResolutionWindowResult]:
        """No target resolution is expected for empty discoveries."""
        raise AssertionError(windows)


def test_cli_runs_candidate_proposal_and_offline_exception_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """The three commands resume model work and materialize typed outputs."""
    manifest_path = tmp_path / "corpus.json"
    canonical_path = write_exception_manifest(manifest_path, exception_corpus)
    change_events_path = tmp_path / "change-events.jsonl"
    candidates_path = tmp_path / "candidates.json"
    proposals_path = tmp_path / "proposals.jsonl"
    progress_path = tmp_path / "progress.jsonl"
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
        "--max-concurrency",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_exception_proposals_cli.main()
    ontology_exception_proposals_cli.main()

    assert len(read_exception_proposals(proposals_path)) == 1
    assert _CliExceptionClient.calls == 1
    finalize_argv = [
        "legal-finalize-legal-exceptions",
        str(manifest_path),
        str(canonical_path),
        str(change_events_path),
        str(proposals_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(exceptions_path),
        str(graph_path),
    ]
    monkeypatch.setattr(sys, "argv", finalize_argv)
    ontology_exception_finalize_cli.main()

    assert len(read_legal_exception_records(exceptions_path)) == 1
    assert len(Graph().parse(graph_path, format="turtle")) > 0

    proposal = read_exception_proposals(proposals_path)[0]
    selection_path = tmp_path / "exception-rerun.json"
    selection_path.write_text(
        json.dumps({"schema_version": "0.1.0", "candidate_ids": [proposal.candidate_id]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exception_proposals, "BamlExceptionProposalClient", _EmptyExceptionClient)
    selective_argv = [*proposal_argv, "--rerun-candidate-ids-path", str(selection_path)]
    _EmptyExceptionClient.calls = 0
    monkeypatch.setattr(sys, "argv", selective_argv)
    ontology_exception_proposals_cli.main()
    ontology_exception_proposals_cli.main()

    assert read_exception_proposals(proposals_path) == ()
    assert read_exception_progress(progress_path)[0].run_id != proposal.run_id
    assert _EmptyExceptionClient.calls == 1
