"""Command-line interface for structural cross-document change candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the legal-change candidate command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build exhaustive bounded cross-document windows for downstream LLM confirmation."
    )
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("change_contract_path", type=Path, help="Versioned change-resolution contract YAML path")
    parser.add_argument("output_path", type=Path, help="Legal-change candidate-set JSON destination")
    return parser


def main() -> None:
    """Build and persist structural legal-change candidate windows offline."""
    from legal_rag.ontology_change.candidates import build_change_candidates
    from legal_rag.ontology_change.config import ChangeResolutionSettings
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest
    from legal_rag.ontology_change.writer import write_change_candidates

    arguments = _parser().parse_args()
    corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    settings = ChangeResolutionSettings.from_path(arguments.change_contract_path)
    candidates = build_change_candidates(corpus, settings)
    write_change_candidates(arguments.output_path, candidates)
    logger.info(
        "Wrote {} structural legal-change candidate windows with run id {} to {}",
        len(candidates.windows),
        candidates.run_id,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
