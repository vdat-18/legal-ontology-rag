"""Command-line interface for the target-only conduct LLM step."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the target-conduct extraction parser."""
    parser = argparse.ArgumentParser(
        description="Extract and checkpoint target-only conduct inventories without population binding."
    )
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("contract_path", type=Path, help="Versioned ontology population contract YAML path")
    parser.add_argument("inventories_output_path", type=Path, help="Target-conduct inventory JSONL destination")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target-limit", type=int, help="Maximum accepted chunks selected for a pilot run")
    target_group.add_argument("--target-ids-path", type=Path, help="Reviewed pilot target-selection JSON path")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum concurrent conduct-extraction batches")
    parser.add_argument("--preflight-report-path", type=Path, help="Optional machine-readable readiness report")
    return parser


def main() -> None:
    """Run one resumable LLM step that inventories target conducts."""
    from legal_rag.ontology_population.cli_support import prepare_population_step
    from legal_rag.ontology_population.conduct_models import TargetConductInventoryRecord
    from legal_rag.ontology_population.conduct_writer import (
        read_target_conduct_inventories,
        validate_target_conduct_inventory,
        write_target_conduct_inventories,
    )
    from legal_rag.ontology_population.extraction import (
        BamlOntologyPopulationClient,
        build_target_conduct_run_id,
        extract_target_conducts,
    )
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    prepared = prepare_population_step(
        chunks_path=arguments.chunks_path,
        screening_path=arguments.screening_path,
        metadata_path=arguments.metadata_path,
        contract_path=arguments.contract_path,
        target_limit=arguments.target_limit,
        target_ids_path=arguments.target_ids_path,
        preflight_report_path=arguments.preflight_report_path,
    )
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    provider_settings = Settings.from_environment()
    run_id = build_target_conduct_run_id(prepared.pilot_run_id, provider_settings)
    existing = (
        read_target_conduct_inventories(arguments.inventories_output_path)
        if arguments.inventories_output_path.exists()
        else ()
    )
    target_ids = tuple(chunk.chunk_id for chunk in prepared.targets)
    validate_target_conduct_inventory(existing, target_ids, expected_run_id=run_id, require_complete=False)
    records_by_id = {record.target_chunk_id: record for record in existing}
    chunk_order = {chunk.chunk_id: chunk.chunk_index for chunk in prepared.bundle.chunks}
    remaining = tuple(chunk for chunk in prepared.targets if chunk.chunk_id not in records_by_id)

    def ordered_records() -> tuple[TargetConductInventoryRecord, ...]:
        """Return stable source-ordered inventories."""
        return tuple(sorted(records_by_id.values(), key=lambda record: chunk_order[record.target_chunk_id]))

    def persist_batch(batch_results: Sequence[TargetConductInventoryRecord]) -> None:
        """Checkpoint every completed target, including empty inventories."""
        records_by_id.update({record.target_chunk_id: record for record in batch_results})
        write_target_conduct_inventories(arguments.inventories_output_path, ordered_records())

    logger.info(
        "Loaded {} selected chunks; {} already inventoried, {} remaining; max concurrency {}",
        len(prepared.targets),
        len(prepared.targets) - len(remaining),
        len(remaining),
        arguments.max_concurrency,
    )
    if remaining:
        client = BamlOntologyPopulationClient(provider_settings, prepared.settings)
        extract_target_conducts(
            remaining,
            client,
            prepared.settings,
            run_id=run_id,
            max_concurrency=arguments.max_concurrency,
            on_batch_processed=persist_batch,
        )
    records = ordered_records()
    write_target_conduct_inventories(arguments.inventories_output_path, records)
    logger.info(
        "Completed {} target chunks and wrote {} conducts to {}",
        len(records),
        sum(len(record.conducts) for record in records),
        arguments.inventories_output_path,
    )


if __name__ == "__main__":
    main()
