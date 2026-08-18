"""Validate and render the AI-assisted internal legal-review packet."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from legal_rag.ontology_validation.legal_review_config import LegalReviewSettings
from legal_rag.ontology_validation.legal_review_models import LegalReviewCheckResult, LegalReviewReadinessReport
from legal_rag.ontology_validation.loader import load_yaml
from legal_rag.ontology_validation.traceability import collect_ids


def run_legal_review_readiness(
    project_root: Path,
    settings: LegalReviewSettings,
    checked_at: datetime,
) -> LegalReviewReadinessReport:
    """Validate packet completeness without granting project-owner acceptance."""
    checks = (
        _decision_inventory_check(settings),
        _traceability_check(settings),
        _pending_topic_coverage_check(settings),
        _glossary_coverage_check(project_root, settings),
        _evidence_check(settings),
        _human_record_check(settings),
    )
    overall_status = "passed" if all(check.status == "passed" for check in checks) else "failed"
    decisions = tuple(decision.decision for decision in settings.decisions)
    review_status = _review_status(decisions)
    if overall_status == "failed" and review_status == "approved":
        review_status = "pending"
    return LegalReviewReadinessReport(
        checked_at=checked_at,
        artifact_id=settings.artifact_id,
        artifact_version=settings.artifact_version,
        overall_status=overall_status,
        review_status=review_status,
        project_owner_acceptance=settings.project_owner_status,
        accepted_count=decisions.count("accept"),
        total_count=len(settings.items),
        checks=checks,
    )


def collect_legal_review_packet_issues(project_root: Path, settings: LegalReviewSettings) -> tuple[str, ...]:
    """Return stable release-gate issues for legal-review packet structure."""
    report = run_legal_review_readiness(project_root, settings, datetime.min.replace(tzinfo=UTC))
    return tuple(sorted(issue for check in report.checks for issue in check.issues))


def render_legal_review_packet(project_root: Path, settings: LegalReviewSettings) -> str:
    """Render a deterministic Markdown packet for the internal reviewer."""
    decisions = {decision.review_id: decision for decision in settings.decisions}
    lines = [
        f"# {settings.title}",
        "",
        f"Artifact: `{settings.artifact_id}` version `{settings.artifact_version}`",
        "",
        "This packet records AI-assisted internal legal-domain review for a solo project. It is not licensed legal",
        "advice or independent professional certification; project-owner acceptance remains explicit.",
        "",
        "## Reviewer Record",
        "",
        f"- Name: {settings.reviewer.name or '[pending]'}",
        f"- Qualifications: {settings.reviewer.qualifications or '[pending]'}",
        f"- Organization: {settings.reviewer.organization or '[pending]'}",
        f"- Reviewed on: {settings.reviewer.reviewed_on or '[pending]'}",
        f"- Project-owner acceptance: {settings.project_owner_status}",
        f"- Project-owner accepted on: {settings.project_owner_accepted_on or '[pending]'}",
        "",
        "## Review Items",
    ]
    for item in settings.items:
        decision = decisions[item.review_id]
        evidence = [path.resolve().relative_to(project_root.resolve()).as_posix() for path in item.evidence_paths]
        lines.extend(
            (
                "",
                f"### {item.review_id} - {item.title}",
                "",
                f"Question: {item.question}",
                "",
                f"Current decision: `{decision.decision}`",
                "",
                f"Affected IDs: {', '.join(f'`{value}`' for value in item.affected_ids)}",
                "",
                f"Evidence files: {', '.join(f'`{value}`' for value in evidence)}",
                "",
                "Acceptance criteria:",
                "",
                *(f"- {criterion}" for criterion in item.acceptance_criteria),
                "",
                f"Reviewer rationale: {decision.rationale or '[pending]'}",
            )
        )
    return "\n".join(lines) + "\n"


def _decision_inventory_check(settings: LegalReviewSettings) -> LegalReviewCheckResult:
    """Require one decision record for every contracted review item."""
    contracted = {item.review_id for item in settings.items}
    recorded = {decision.review_id for decision in settings.decisions}
    issues = [f"missing-review-decision:{value}" for value in sorted(contracted - recorded)]
    issues.extend(f"unexpected-review-decision:{value}" for value in sorted(recorded - contracted))
    if len(recorded) != len(settings.decisions):
        issues.append("duplicate-review-decision-id")
    if settings.decision_contract_id != settings.artifact_id:
        issues.append("decision-contract-id-mismatch")
    if settings.decision_contract_version != settings.artifact_version:
        issues.append("decision-contract-version-mismatch")
    return _check("decision-inventory", issues)


def _traceability_check(settings: LegalReviewSettings) -> LegalReviewCheckResult:
    """Require every affected ID to resolve in a versioned ontology registry."""
    known_ids = frozenset().union(*(collect_ids(load_yaml(path)) for path in settings.source_registry_paths))
    referenced = {value for item in settings.items for value in item.affected_ids}
    return _check("registry-traceability", [f"unknown-affected-id:{value}" for value in sorted(referenced - known_ids)])


def _pending_topic_coverage_check(settings: LegalReviewSettings) -> LegalReviewCheckResult:
    """Require every modeling-decision legal topic to appear in the review contract."""
    decisions_document = load_yaml(settings.source_registry_paths[-1])
    required = {
        topic
        for decision in decisions_document["modeling_decisions"]
        for topic in (*decision["pending_review"], *decision.get("resolved_review", []))
    }
    covered = {topic for item in settings.items for topic in item.pending_topics}
    issues = [f"uncovered-pending-topic:{value}" for value in sorted(required - covered)]
    issues.extend(f"unknown-pending-topic:{value}" for value in sorted(covered - required))
    return _check("pending-topic-coverage", issues)


def _glossary_coverage_check(project_root: Path, settings: LegalReviewSettings) -> LegalReviewCheckResult:
    """Require the internal review packet to cover every V0.1 glossary term."""
    glossary = load_yaml(project_root / "ontology" / "glossary" / "terms.yaml")
    required = {term["id"] for term in glossary["terms"]}
    covered = {value for item in settings.items for value in item.affected_ids if value.startswith("TERM-")}
    return _check("glossary-coverage", [f"uncovered-glossary-term:{value}" for value in sorted(required - covered)])


def _evidence_check(settings: LegalReviewSettings) -> LegalReviewCheckResult:
    """Require review questions, criteria, and all referenced evidence files."""
    issues = []
    for item in settings.items:
        if not item.question or not item.acceptance_criteria:
            issues.append(f"incomplete-review-item:{item.review_id}")
        issues.extend(
            f"missing-review-evidence:{item.review_id}:{path}" for path in item.evidence_paths if not path.is_file()
        )
    return _check("review-evidence", issues)


def _human_record_check(settings: LegalReviewSettings) -> LegalReviewCheckResult:
    """Require attributable evidence for every non-pending human decision."""
    issues = []
    for decision in settings.decisions:
        if decision.decision not in settings.allowed_decisions:
            issues.append(f"invalid-review-decision:{decision.review_id}:{decision.decision}")
        if decision.decision != "pending" and (not decision.rationale or not decision.evidence_refs):
            issues.append(f"incomplete-review-decision:{decision.review_id}")
        if decision.decision == "change_requested" and not decision.requested_changes:
            issues.append(f"missing-requested-changes:{decision.review_id}")
    if any(decision.decision != "pending" for decision in settings.decisions) and not all(
        (
            settings.reviewer.name,
            settings.reviewer.qualifications,
            settings.reviewer.organization,
            settings.reviewer.reviewed_on,
        )
    ):
        issues.append("incomplete-internal-reviewer-record")
    expected_status = {
        "pending": "pending_internal_legal_review",
        "approved": "approved",
        "rejected": "rejected",
        "changes_requested": "changes_requested",
    }[_review_status(tuple(decision.decision for decision in settings.decisions))]
    if settings.decision_status != expected_status:
        issues.append(f"decision-status-mismatch:{settings.decision_status}:{expected_status}")
    if expected_status == "approved" and settings.project_owner_status != "approved":
        issues.append("missing-project-owner-acceptance")
    if settings.project_owner_status == "approved" and not (
        settings.project_owner_accepted_on and settings.project_owner_evidence
    ):
        issues.append("incomplete-project-owner-acceptance")
    return _check("human-review-record", issues)


def _review_status(decisions: tuple[str, ...]) -> Literal["pending", "approved", "rejected", "changes_requested"]:
    """Return the aggregate human review state from item decisions."""
    if "reject" in decisions:
        return "rejected"
    if "change_requested" in decisions:
        return "changes_requested"
    if decisions and all(decision == "accept" for decision in decisions):
        return "approved"
    return "pending"


def _check(check_id: str, issues: list[str]) -> LegalReviewCheckResult:
    """Build one stable legal-review readiness check."""
    ordered = tuple(sorted(issues))
    return LegalReviewCheckResult(check_id, "failed" if ordered else "passed", ordered)
