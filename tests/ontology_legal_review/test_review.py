"""Tests for independent legal-review packet validation and rendering."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ontology_validation.legal_review import render_legal_review_packet, run_legal_review_readiness
from legal_rag.ontology_validation.legal_review_config import LegalReviewSettings

CHECKED_AT = datetime(2026, 8, 18, 13, 0, tzinfo=UTC)


def _issues(project_root: Path, settings: LegalReviewSettings) -> set[str]:
    """Return every structural packet issue from one readiness run."""
    report = run_legal_review_readiness(project_root, settings, CHECKED_AT)
    return {issue for check in report.checks for issue in check.issues}


def test_current_packet_is_complete_but_independent_review_remains_pending(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """All questions and evidence are ready without claiming a professional legal decision."""
    report = run_legal_review_readiness(project_root, legal_review_settings, CHECKED_AT)

    assert report.overall_status == "passed"
    assert report.review_status == "pending"
    assert report.accepted_count == 0
    assert report.total_count == 11
    assert all(check.status == "passed" for check in report.checks)


def test_missing_review_decision_is_rejected(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """Every contracted legal question requires one explicit reviewer decision record."""
    settings = replace(legal_review_settings, decisions=legal_review_settings.decisions[:-1])

    assert "missing-review-decision:ILR-011" in _issues(project_root, settings)


def test_decisions_must_target_the_exact_contract_version(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """Decision records cannot be reused for another review contract or ontology version."""
    settings = replace(legal_review_settings, decision_contract_version="0.2.0")

    assert "decision-contract-version-mismatch" in _issues(project_root, settings)


def test_unknown_traceability_id_is_rejected(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """Review questions cannot cite an ID absent from the versioned ontology registries."""
    first = replace(
        legal_review_settings.items[0],
        affected_ids=(*legal_review_settings.items[0].affected_ids, "MD-UNKNOWN-001"),
    )
    settings = replace(legal_review_settings, items=(first, *legal_review_settings.items[1:]))

    assert "unknown-affected-id:MD-UNKNOWN-001" in _issues(project_root, settings)


def test_uncovered_modeling_topic_is_rejected(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """Every legal topic left open by ontology engineering must reach the independent reviewer."""
    first = replace(legal_review_settings.items[0], pending_topics=("legal_identity",))
    settings = replace(legal_review_settings, items=(first, *legal_review_settings.items[1:]))

    assert "uncovered-pending-topic:legal_expression_identity" in _issues(project_root, settings)


def test_non_pending_decision_requires_reviewer_identity_and_evidence(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """An accept label alone cannot become attributable independent legal approval."""
    accepted = replace(legal_review_settings.decisions[0], decision="accept")
    settings = replace(legal_review_settings, decisions=(accepted, *legal_review_settings.decisions[1:]))
    issues = _issues(project_root, settings)

    assert "incomplete-review-decision:ILR-001" in issues
    assert "incomplete-independent-reviewer-record" in issues


def test_renderer_exposes_questions_evidence_and_pending_decisions(
    project_root: Path,
    legal_review_settings: LegalReviewSettings,
) -> None:
    """The generated packet is directly usable by a reviewer without reading implementation code."""
    packet = render_legal_review_packet(project_root, legal_review_settings)

    assert "### ILR-001 - Rule and legal-expression identity" in packet
    assert "### ILR-011 - V0.1 scope, competency questions, and legal limitations" in packet
    assert "Current decision: `pending`" in packet
    assert "ontology/modules/normative_rule.ttl" in packet
