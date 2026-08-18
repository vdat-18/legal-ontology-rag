"""Command-line interface for live ontology publication verification."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the live publication-verification parser."""
    parser = argparse.ArgumentParser(description="Verify deployed ontology files or merged W3ID redirects.")
    parser.add_argument("publication_contract_path", type=Path, help="Versioned namespace-publication contract YAML")
    parser.add_argument("mode", choices=("deployment", "w3id"), help="Direct deployment or W3ID redirect verification")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable live-verification JSON report")
    return parser


def main() -> None:
    """Run bounded live checks, persist the report, and fail on any mismatch."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings
    from legal_rag.ontology_validation.publication_verification import verify_namespace_publication

    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    settings = NamespacePublicationSettings.from_path(arguments.publication_contract_path, project_root)
    report = verify_namespace_publication(settings, arguments.mode, datetime.now(UTC))
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "Ontology publication verification completed with status {} across {} requests",
        report.overall_status,
        len(report.results),
    )
    if report.overall_status != "passed":
        msg = f"Ontology publication verification failed; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
