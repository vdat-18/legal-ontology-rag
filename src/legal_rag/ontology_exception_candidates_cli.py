"""Command-line interface for structural legal-exception candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the exception-candidate command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build source-local general-rule windows for downstream LLM exception discovery."
    )
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("canonical_records_path", type=Path, help="Canonical multi-document population JSONL path")
    parser.add_argument("exception_contract_path", type=Path, help="Versioned exception-resolution contract YAML")
    parser.add_argument("output_path", type=Path, help="Exception candidate-set JSON destination")
    return parser


def main() -> None:
    """Build and persist source-local legal-exception candidate windows offline."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest, replace_change_corpus_records
    from legal_rag.ontology_exception.candidates import build_exception_candidates
    from legal_rag.ontology_exception.config import ExceptionResolutionSettings
    from legal_rag.ontology_exception.writer import write_exception_candidates
    from legal_rag.ontology_population.writer import read_population_records

    arguments = _parser().parse_args()
    manifest_corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    records = read_population_records(arguments.canonical_records_path)
    corpus = replace_change_corpus_records(manifest_corpus, records)
    settings = ExceptionResolutionSettings.from_path(arguments.exception_contract_path)
    candidates = build_exception_candidates(corpus, settings)
    write_exception_candidates(arguments.output_path, candidates)
    logger.info(
        "Wrote {} source-local exception candidate windows with run id {} to {}",
        len(candidates.windows),
        candidates.run_id,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
