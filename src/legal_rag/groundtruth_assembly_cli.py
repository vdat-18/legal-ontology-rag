"""Command-line interface for validating and assembling ground-truth facts."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the corpus assembly command-line interface."""
    parser = argparse.ArgumentParser(description="Validate document synthesis artifacts and assemble legal facts.")
    parser.add_argument("processed_dir", type=Path, help="Directory containing per-document synthesis artifacts")
    parser.add_argument("corpus_config", type=Path, help="Ground-truth corpus selection JSON path")
    parser.add_argument("snapshot", type=Path, help="Frozen corpus snapshot JSON path")
    parser.add_argument("facts_output_path", type=Path, help="Destination corpus-level legal-facts JSONL path")
    parser.add_argument("manifest_output_path", type=Path, help="Destination run manifest JSON path")
    parser.add_argument("--config", type=Path, action="append", default=[], help="Versioned synthesis config path")
    return parser


def main() -> None:
    """Validate every selected document and write one reproducible corpus-level run."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.assembly import assemble_groundtruth_corpus
    from legal_rag.benchmark_synthesis.writer import write_legal_facts
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    result = assemble_groundtruth_corpus(
        arguments.processed_dir,
        arguments.corpus_config,
        arguments.snapshot,
        arguments.config,
        llm_model=settings.llm_model,
    )
    write_legal_facts(arguments.facts_output_path, result.facts)
    write_json_atomic(arguments.manifest_output_path, result.manifest)
    logger.info(
        "Assembled {} validated facts into run {}, wrote {} and {}",
        len(result.facts),
        result.manifest["run_id"],
        arguments.facts_output_path,
        arguments.manifest_output_path,
    )


if __name__ == "__main__":
    main()
