"""Typed configuration for independent ontology legal review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

LEGAL_REVIEW_SCHEMA_VERSION = "0.1.0"
LEGAL_REVIEW_REPORT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class LegalReviewItem:
    """One bounded question for the independent legal reviewer."""

    review_id: str
    title: str
    question: str
    pending_topics: tuple[str, ...]
    affected_ids: tuple[str, ...]
    evidence_paths: tuple[Path, ...]
    acceptance_criteria: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LegalReviewDecision:
    """One attributable human decision for a review item."""

    review_id: str
    decision: str
    rationale: str | None
    evidence_refs: tuple[str, ...]
    requested_changes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LegalReviewer:
    """Identity and qualifications recorded by the independent reviewer."""

    name: str | None
    qualifications: str | None
    organization: str | None
    reviewed_on: str | None


@dataclass(frozen=True, slots=True)
class LegalReviewSettings:
    """Versioned legal-review questions, evidence, and decision records."""

    artifact_id: str
    artifact_version: str
    title: str
    allowed_decisions: tuple[str, ...]
    source_registry_paths: tuple[Path, ...]
    items: tuple[LegalReviewItem, ...]
    decision_artifact_id: str
    decision_contract_id: str
    decision_contract_version: str
    decision_status: str
    reviewer: LegalReviewer
    decisions: tuple[LegalReviewDecision, ...]

    @classmethod
    def from_paths(cls, contract_path: Path, decisions_path: Path, project_root: Path) -> LegalReviewSettings:
        """Load the legal-review contract and its human decision registry."""
        contract = load_yaml(contract_path)
        decisions = load_yaml(decisions_path)
        if contract["schema_version"] != LEGAL_REVIEW_SCHEMA_VERSION:
            msg = f"Unsupported legal-review contract: {contract['schema_version']}"
            raise ValueError(msg)
        if decisions["schema_version"] != LEGAL_REVIEW_SCHEMA_VERSION:
            msg = f"Unsupported legal-review decisions: {decisions['schema_version']}"
            raise ValueError(msg)
        reviewer = decisions["reviewer"]
        return cls(
            artifact_id=contract["artifact_id"],
            artifact_version=contract["artifact_version"],
            title=contract["title"],
            allowed_decisions=tuple(contract["review_policy"]["allowed_decisions"]),
            source_registry_paths=tuple(project_root / path for path in contract["source_registries"].values()),
            items=tuple(
                LegalReviewItem(
                    review_id=item["id"],
                    title=item["title"],
                    question=item["question"],
                    pending_topics=tuple(item["pending_topics"]),
                    affected_ids=tuple(item["affected_ids"]),
                    evidence_paths=tuple(project_root / path for path in item["evidence_files"]),
                    acceptance_criteria=tuple(item["acceptance_criteria"]),
                )
                for item in contract["review_items"]
            ),
            decision_artifact_id=decisions["artifact_id"],
            decision_contract_id=decisions["contract_id"],
            decision_contract_version=decisions["contract_version"],
            decision_status=decisions["status"],
            reviewer=LegalReviewer(
                name=reviewer["name"],
                qualifications=reviewer["qualifications"],
                organization=reviewer["organization"],
                reviewed_on=reviewer["reviewed_on"],
            ),
            decisions=tuple(
                LegalReviewDecision(
                    review_id=item["id"],
                    decision=item["decision"],
                    rationale=item["rationale"],
                    evidence_refs=tuple(item["evidence_refs"]),
                    requested_changes=tuple(item["requested_changes"]),
                )
                for item in decisions["decisions"]
            ),
        )
