"""Command-line interface for ontology namespace publication readiness."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the namespace-publication command-line parser."""
    parser = argparse.ArgumentParser(description="Validate W3ID mappings and build the ontology publication site.")
    parser.add_argument("publication_contract_path", type=Path, help="Versioned namespace-publication contract YAML")
    parser.add_argument(
        "site_output_directory", type=Path, help="Fresh output directory for the deployable static site"
    )
    parser.add_argument("report_output_path", type=Path, help="Machine-readable publication-readiness JSON report")
    return parser


def main() -> None:
    """Build the publication site and persist readiness before failing open gates."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_validation.publication import build_namespace_publication
    from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings

    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    settings = NamespacePublicationSettings.from_path(arguments.publication_contract_path, project_root)
    report = build_namespace_publication(
        project_root,
        settings,
        arguments.site_output_directory,
        datetime.now(UTC),
    )
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "Ontology namespace publication completed with status {} across {} checks",
        report.overall_status,
        len(report.checks),
    )
    if report.overall_status != "built":
        msg = f"Ontology namespace publication failed; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
