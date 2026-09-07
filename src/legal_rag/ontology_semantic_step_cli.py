"""Integrated ontology semantic workflow CLIs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _run_modules(commands: Sequence[tuple[str, Sequence[str | Path]]]) -> None:
    """Run deterministic preparation, one proposal stage, and finalization."""
    for module, arguments in commands:
        logger.info("Running ontology step component {}", module)
        subprocess.run(  # noqa: S603 - module names are fixed by this CLI and no shell is used.
            [sys.executable, "-m", module, *(str(argument) for argument in arguments)],
            check=True,
        )


def _assembly_parser() -> argparse.ArgumentParser:
    """Build the integrated within-document assembly parser."""
    parser = argparse.ArgumentParser(
        description="Build candidates, propose assemblies, resolve global conflicts, and finalize canonical rules."
    )
    parser.add_argument("chunks_path", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("records_path", type=Path)
    parser.add_argument("population_contract_path", type=Path)
    parser.add_argument("assembly_contract_path", type=Path)
    parser.add_argument("candidates_output_path", type=Path)
    parser.add_argument("proposals_output_path", type=Path)
    parser.add_argument("progress_output_path", type=Path)
    parser.add_argument("canonical_records_output_path", type=Path)
    parser.add_argument("graph_output_path", type=Path)
    parser.add_argument("--resolved-proposals-output-path", type=Path)
    parser.add_argument("--resolution-progress-output-path", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY)
    return parser


def assemble_main() -> None:
    """Run the complete within-document assembly step."""
    arguments = _assembly_parser().parse_args()
    resolved_proposals_path = arguments.resolved_proposals_output_path or arguments.proposals_output_path.with_name(
        f"{arguments.proposals_output_path.stem}.resolved{arguments.proposals_output_path.suffix}"
    )
    resolution_progress_path = arguments.resolution_progress_output_path or arguments.progress_output_path.with_name(
        f"{arguments.progress_output_path.stem}.resolution{arguments.progress_output_path.suffix}"
    )
    _run_modules(
        (
            (
                "legal_rag.ontology_assembly_candidates_cli",
                (
                    arguments.chunks_path,
                    arguments.records_path,
                    arguments.assembly_contract_path,
                    arguments.candidates_output_path,
                ),
            ),
            (
                "legal_rag.ontology_assembly_proposals_cli",
                (
                    arguments.chunks_path,
                    arguments.metadata_path,
                    arguments.records_path,
                    arguments.candidates_output_path,
                    arguments.assembly_contract_path,
                    arguments.proposals_output_path,
                    arguments.progress_output_path,
                    "--max-concurrency",
                    str(arguments.max_concurrency),
                ),
            ),
            (
                "legal_rag.ontology_assembly_resolution_cli",
                (
                    arguments.chunks_path,
                    arguments.metadata_path,
                    arguments.records_path,
                    arguments.proposals_output_path,
                    arguments.assembly_contract_path,
                    resolved_proposals_path,
                    resolution_progress_path,
                    "--max-concurrency",
                    str(arguments.max_concurrency),
                ),
            ),
            (
                "legal_rag.ontology_assembly_finalize_cli",
                (
                    arguments.chunks_path,
                    arguments.metadata_path,
                    arguments.records_path,
                    resolved_proposals_path,
                    arguments.population_contract_path,
                    arguments.canonical_records_output_path,
                    arguments.graph_output_path,
                ),
            ),
        )
    )


def _change_parser() -> argparse.ArgumentParser:
    """Build the integrated legal-change parser."""
    parser = argparse.ArgumentParser(
        description="Build candidates, run one legal-change LLM task, and finalize rule versions."
    )
    parser.add_argument("corpus_manifest_path", type=Path)
    parser.add_argument("change_contract_path", type=Path)
    parser.add_argument("population_contract_path", type=Path)
    parser.add_argument("candidates_output_path", type=Path)
    parser.add_argument("proposals_output_path", type=Path)
    parser.add_argument("progress_output_path", type=Path)
    parser.add_argument("canonical_records_output_path", type=Path)
    parser.add_argument("change_events_output_path", type=Path)
    parser.add_argument("graph_output_path", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY)
    return parser


def changes_main() -> None:
    """Run the complete cross-document legal-change step."""
    arguments = _change_parser().parse_args()
    _run_modules(
        (
            (
                "legal_rag.ontology_change_candidates_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.change_contract_path,
                    arguments.candidates_output_path,
                ),
            ),
            (
                "legal_rag.ontology_change_proposals_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.candidates_output_path,
                    arguments.change_contract_path,
                    arguments.proposals_output_path,
                    arguments.progress_output_path,
                    "--max-concurrency",
                    str(arguments.max_concurrency),
                ),
            ),
            (
                "legal_rag.ontology_change_finalize_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.proposals_output_path,
                    arguments.population_contract_path,
                    arguments.canonical_records_output_path,
                    arguments.change_events_output_path,
                    arguments.graph_output_path,
                ),
            ),
        )
    )


def _exception_parser() -> argparse.ArgumentParser:
    """Build the integrated legal-exception parser."""
    parser = argparse.ArgumentParser(
        description="Build candidates, discover and resolve exceptions with LLMs, then finalize them."
    )
    parser.add_argument("corpus_manifest_path", type=Path)
    parser.add_argument("canonical_records_path", type=Path)
    parser.add_argument("change_events_path", type=Path)
    parser.add_argument("exception_contract_path", type=Path)
    parser.add_argument("population_contract_path", type=Path)
    parser.add_argument("candidates_output_path", type=Path)
    parser.add_argument("proposals_output_path", type=Path)
    parser.add_argument("progress_output_path", type=Path)
    parser.add_argument("exception_records_output_path", type=Path)
    parser.add_argument("graph_output_path", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY)
    return parser


def exceptions_main() -> None:
    """Run the complete legal-exception and citation step."""
    arguments = _exception_parser().parse_args()
    _run_modules(
        (
            (
                "legal_rag.ontology_exception_candidates_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.canonical_records_path,
                    arguments.exception_contract_path,
                    arguments.candidates_output_path,
                ),
            ),
            (
                "legal_rag.ontology_exception_proposals_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.canonical_records_path,
                    arguments.candidates_output_path,
                    arguments.exception_contract_path,
                    arguments.proposals_output_path,
                    arguments.progress_output_path,
                    "--max-concurrency",
                    str(arguments.max_concurrency),
                ),
            ),
            (
                "legal_rag.ontology_exception_finalize_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.canonical_records_path,
                    arguments.change_events_path,
                    arguments.proposals_output_path,
                    arguments.population_contract_path,
                    arguments.exception_records_output_path,
                    arguments.graph_output_path,
                ),
            ),
        )
    )


def _concept_parser() -> argparse.ArgumentParser:
    """Build the integrated concept-resolution parser."""
    parser = argparse.ArgumentParser(
        description="Build candidates, run one concept LLM task, and materialize the final RDF graph."
    )
    parser.add_argument("corpus_manifest_path", type=Path)
    parser.add_argument("canonical_records_path", type=Path)
    parser.add_argument("change_events_path", type=Path)
    parser.add_argument("exception_records_path", type=Path)
    parser.add_argument("population_contract_path", type=Path)
    parser.add_argument("concept_contract_path", type=Path)
    parser.add_argument("candidates_output_path", type=Path)
    parser.add_argument("proposals_output_path", type=Path)
    parser.add_argument("progress_output_path", type=Path)
    parser.add_argument("concepts_output_path", type=Path)
    parser.add_argument("relations_output_path", type=Path)
    parser.add_argument("graph_output_path", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY)
    return parser


def concepts_main() -> None:
    """Run the complete reusable-concept resolution step."""
    arguments = _concept_parser().parse_args()
    _run_modules(
        (
            (
                "legal_rag.ontology_concept_candidates_cli",
                (
                    arguments.canonical_records_path,
                    arguments.concept_contract_path,
                    arguments.candidates_output_path,
                ),
            ),
            (
                "legal_rag.ontology_concept_proposals_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.candidates_output_path,
                    arguments.concept_contract_path,
                    arguments.proposals_output_path,
                    arguments.progress_output_path,
                    "--max-concurrency",
                    str(arguments.max_concurrency),
                ),
            ),
            (
                "legal_rag.ontology_concept_finalize_cli",
                (
                    arguments.corpus_manifest_path,
                    arguments.canonical_records_path,
                    arguments.change_events_path,
                    arguments.exception_records_path,
                    arguments.candidates_output_path,
                    arguments.proposals_output_path,
                    arguments.population_contract_path,
                    arguments.concept_contract_path,
                    arguments.concepts_output_path,
                    arguments.relations_output_path,
                    arguments.graph_output_path,
                ),
            ),
        )
    )
