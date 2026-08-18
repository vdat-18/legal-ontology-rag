"""Command-line interface for ontology semantic release-hardening checks."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology semantic-QA command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run pinned OWL 2 DL profile, HermiT, ROBOT quality, and traceability release gates."
    )
    parser.add_argument("semantic_contract_path", type=Path, help="Versioned ontology semantic-QA contract YAML")
    parser.add_argument("artifact_output_directory", type=Path, help="ROBOT profile, reasoner, and quality artifacts")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable semantic-QA JSON report")
    return parser


def main() -> None:
    """Run semantic QA, persist its report, and fail when a release gate remains open."""
    from legal_rag.ontology_validation.paths import OntologyPaths
    from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
    from legal_rag.ontology_validation.semantic_qa import run_semantic_qa
    from legal_rag.ontology_validation.semantic_runner import DockerRobotRunner
    from legal_rag.ontology_validation.semantic_writer import write_semantic_qa_report

    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    paths = OntologyPaths(project_root)
    settings = SemanticQaSettings.from_path(arguments.semantic_contract_path, project_root)
    runner = DockerRobotRunner(project_root, settings)
    report = run_semantic_qa(
        paths,
        settings,
        runner,
        arguments.artifact_output_directory,
        datetime.now(UTC),
    )
    write_semantic_qa_report(arguments.report_output_path, report)
    logger.info(
        "Ontology semantic QA completed with status {} across {} checks",
        report.overall_status,
        len(report.checks),
    )
    if report.overall_status != "passed":
        msg = f"Ontology semantic QA failed; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
