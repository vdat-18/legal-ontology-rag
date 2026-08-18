"""Command-line interface for ontology release packaging."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology release command-line parser."""
    parser = argparse.ArgumentParser(description="Validate release gates and build a reproducible ontology archive.")
    parser.add_argument("release_contract_path", type=Path, help="Versioned ontology release contract YAML")
    parser.add_argument("semantic_report_path", type=Path, help="Passing semantic-QA JSON report")
    parser.add_argument("output_directory", type=Path, help="Release archive and checksum output directory")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable release-readiness JSON report")
    return parser


def main() -> None:
    """Persist release readiness and fail after recording any open gate."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_validation.release_config import OntologyReleaseSettings
    from legal_rag.ontology_validation.release_packager import build_ontology_release

    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    settings = OntologyReleaseSettings.from_path(arguments.release_contract_path, project_root)
    report = build_ontology_release(
        project_root,
        settings,
        arguments.semantic_report_path,
        arguments.output_directory,
        datetime.now(UTC),
    )
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "Ontology release packaging completed with status {} across {} gates",
        report.overall_status,
        len(report.gates),
    )
    if report.overall_status != "released":
        msg = f"Ontology release is blocked; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
