"""Command-line interface for ontology semantic evolution diffing."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology evolution-diff command-line parser."""
    parser = argparse.ArgumentParser(description="Compare two ontology project roots and reject breaking changes.")
    parser.add_argument("before_project_root", type=Path, help="Project root containing the earlier ontology version")
    parser.add_argument("after_project_root", type=Path, help="Project root containing the candidate ontology version")
    parser.add_argument("evolution_policy_path", type=Path, help="Versioned ontology evolution policy YAML")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable ontology evolution report JSON")
    return parser


def main() -> None:
    """Persist the semantic diff before failing on any breaking change."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_validation.evolution import compare_ontology_projects
    from legal_rag.ontology_validation.evolution_config import OntologyEvolutionSettings

    arguments = _parser().parse_args()
    settings = OntologyEvolutionSettings.from_path(arguments.evolution_policy_path)
    report = compare_ontology_projects(
        arguments.before_project_root,
        arguments.after_project_root,
        settings,
        datetime.now(UTC),
    )
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "Ontology evolution diff completed with status {} and {} breaking issues",
        report.overall_status,
        len(report.issues),
    )
    if report.overall_status != "passed":
        msg = f"Ontology evolution contains breaking changes; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
