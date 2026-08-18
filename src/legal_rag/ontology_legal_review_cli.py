"""Command-line interface for AI-assisted internal ontology legal review."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the internal legal-review parser."""
    parser = argparse.ArgumentParser(description="Validate and render the AI-assisted internal legal-review packet.")
    parser.add_argument("contract_path", type=Path, help="Versioned legal-review contract YAML")
    parser.add_argument("decisions_path", type=Path, help="Internal reviewer decision registry YAML")
    parser.add_argument("packet_output_path", type=Path, help="Rendered Markdown review packet")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable readiness report")
    return parser


def main() -> None:
    """Persist the internal review packet and report, failing only on structural issues."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_validation.legal_review import render_legal_review_packet, run_legal_review_readiness
    from legal_rag.ontology_validation.legal_review_config import LegalReviewSettings

    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    settings = LegalReviewSettings.from_paths(arguments.contract_path, arguments.decisions_path, project_root)
    report = run_legal_review_readiness(project_root, settings, datetime.now(UTC))
    arguments.packet_output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.packet_output_path.write_text(
        render_legal_review_packet(project_root, settings),
        encoding="utf-8",
        newline="\n",
    )
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "Internal legal-review packet completed with structural status {} and reviewer status {}",
        report.overall_status,
        report.review_status,
    )
    if report.overall_status != "passed":
        msg = f"Internal legal-review packet is invalid; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
