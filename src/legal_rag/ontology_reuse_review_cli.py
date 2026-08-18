"""Command-line interface for external ontology reuse review."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the external-reuse review parser."""
    parser = argparse.ArgumentParser(description="Validate external ontology term reuse, boundaries, and evidence.")
    parser.add_argument("reuse_registry_path", type=Path, help="Versioned external-reuse registry YAML")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable external-reuse review JSON report")
    return parser


def main() -> None:
    """Persist the technical reuse report and fail on any unresolved structural issue."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_validation.reuse_config import ExternalReuseSettings
    from legal_rag.ontology_validation.reuse_review import run_external_reuse_review

    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    settings = ExternalReuseSettings.from_path(arguments.reuse_registry_path, project_root)
    report = run_external_reuse_review(project_root, settings, datetime.now(UTC))
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "External ontology reuse review completed with status {} across {} checks",
        report.overall_status,
        len(report.checks),
    )
    if report.overall_status != "passed":
        msg = f"External ontology reuse review failed; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
