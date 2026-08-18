"""End-to-end test for AI-assisted internal legal review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from legal_rag import ontology_legal_review_cli


def test_cli_persists_packet_and_failed_report_before_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_root: Path,
) -> None:
    """Structural failures retain a packet and exact machine-readable evidence."""
    contract = yaml.safe_load((project_root / "ontology" / "legal_review_contract.yaml").read_text(encoding="utf-8"))
    contract["review_items"][0]["affected_ids"].append("TERM-UNKNOWN-001")
    contract_path = tmp_path / "legal_review_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    packet_path = tmp_path / "packet.md"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-review-ontology-internally",
            str(contract_path),
            str(project_root / "ontology" / "legal_review_decisions.yaml"),
            str(packet_path),
            str(report_path),
        ],
    )

    with pytest.raises(ValueError, match="packet is invalid"):
        ontology_legal_review_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "failed"
    assert report["review_status"] == "pending"
    assert report["project_owner_acceptance"] == "approved"
    assert packet_path.is_file()
    assert any("unknown-affected-id:TERM-UNKNOWN-001" in check["issues"] for check in report["checks"])
