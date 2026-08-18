"""End-to-end tests for the external ontology reuse review command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from legal_rag import ontology_reuse_review_cli


def test_cli_persists_failed_report_before_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """Operators retain exact gate evidence when the review fails."""
    registry = yaml.safe_load((project_root / "ontology" / "external_reuse_registry.yaml").read_text(encoding="utf-8"))
    registry["resources"][0]["terms"].append(
        {
            "iri": "https://example.test/not-used",
            "local_use": "Negative CLI fixture.",
            "declaration_in_owl_closure": False,
        }
    )
    registry_path = tmp_path / "external_reuse_registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    report_path = tmp_path / "reuse-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-review-ontology-reuse", str(registry_path), str(report_path)],
    )

    with pytest.raises(ValueError, match="review failed"):
        ontology_reuse_review_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "failed"
    assert report["project_owner_approval"] == "pending"
    assert any("listed-term-not-used:https://example.test/not-used" in check["issues"] for check in report["checks"])
