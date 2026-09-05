"""Command-line interface for the population-binding LLM step."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _parser() -> argparse.ArgumentParser:
    """Build the separate population-binding parser."""
    parser = argparse.ArgumentParser(
        description="Bind checkpointed target conducts to reading-window context and materialize RDF."
    )
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("contract_path", type=Path, help="Versioned ontology population contract YAML path")
    parser.add_argument("inventories_path", type=Path, help="Completed target-conduct inventory JSONL path")
    parser.add_argument("records_output_path", type=Path, help="Population-record JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Population-progress JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Turtle instance-graph destination")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target-limit", type=int, help="Maximum accepted chunks selected for a pilot run")
    target_group.add_argument("--target-ids-path", type=Path, help="Reviewed pilot target-selection JSON path")
    parser.add_argument(
        "--rerun-target-ids-path",
        type=Path,
        help="Reviewed target-selection JSON containing only chunks that must be bound again",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY,
        help="Maximum concurrent population-binding batches",
    )
    parser.add_argument("--batch-size", type=int, help="Override target chunks per population-binding batch")
    parser.add_argument("--preflight-report-path", type=Path, help="Optional machine-readable readiness report")
    return parser


def main() -> None:
    """Bind one completed conduct checkpoint and materialize population RDF."""
    from legal_rag.ontology_population.cli_support import prepare_population_step
    from legal_rag.ontology_population.conduct_writer import (
        read_target_conduct_inventories,
        target_conduct_inventory_identity,
        validate_target_conduct_inventory,
    )
    from legal_rag.ontology_population.extraction import (
        BamlOntologyPopulationClient,
        PersistedConductPopulationClient,
        build_population_binding_rerun_id,
        build_population_binding_run_id,
        extract_population_from_document,
    )
    from legal_rag.ontology_population.materializer import (
        materialize_population_graph,
        write_population_graph,
    )
    from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
    from legal_rag.ontology_population.preflight import (
        load_population_target_selection,
        select_population_rerun_targets,
        validate_population_resume_state,
    )
    from legal_rag.ontology_population.writer import (
        read_population_progress,
        read_population_records,
        write_population_progress,
        write_population_records,
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
    population_settings = (
        replace(prepared.settings, batch_size=arguments.batch_size)
        if arguments.batch_size is not None
        else prepared.settings
    )
    if population_settings.batch_size < 1:
        msg = "Population binding batch size must be at least one"
        raise ValueError(msg)
    inventories = read_target_conduct_inventories(arguments.inventories_path)
    target_ids = tuple(chunk.chunk_id for chunk in prepared.targets)
    validate_target_conduct_inventory(
        inventories,
        target_ids,
        expected_run_id=None,
        require_complete=True,
    )
    base_run_id = build_population_binding_run_id(target_conduct_inventory_identity(inventories), provider_settings)
    rerun_targets = None
    if arguments.rerun_target_ids_path is not None:
        rerun_selection = load_population_target_selection(arguments.rerun_target_ids_path)
        rerun_targets = select_population_rerun_targets(prepared.bundle, prepared.targets, rerun_selection)
    run_id = (
        build_population_binding_rerun_id(base_run_id, tuple(chunk.chunk_id for chunk in rerun_targets))
        if rerun_targets is not None
        else base_run_id
    )
    existing_records = (
        read_population_records(arguments.records_output_path) if arguments.records_output_path.exists() else ()
    )
    records_by_id = {record.population_id: record for record in existing_records}
    chunks = prepared.bundle.chunks
    chunk_order = {chunk.chunk_id: chunk.chunk_index for chunk in chunks}
    existing_progress = (
        read_population_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    validate_population_resume_state(
        prepared.targets,
        existing_records,
        existing_progress,
        expected_run_id=None if rerun_targets is not None else run_id,
        expected_ontology_semantic_sha256=prepared.ontology_semantic_sha256,
    )
    progress_by_id = {record.chunk_id: record for record in existing_progress}
    extraction_targets = prepared.targets if rerun_targets is None else rerun_targets
    remaining = tuple(
        chunk
        for chunk in extraction_targets
        if chunk.chunk_id not in progress_by_id or progress_by_id[chunk.chunk_id].run_id != run_id
    )

    def ordered_records() -> tuple[OntologyPopulationRecord, ...]:
        """Return stable source-ordered population records."""
        return tuple(
            sorted(
                records_by_id.values(),
                key=lambda record: (chunk_order[record.source_chunk_id], record.population_id),
            )
        )

    def persist_batch(batch_results: Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]) -> None:
        """Persist records before marking their target chunks complete."""
        for chunk_id, target_records in batch_results:
            stale_ids = [
                population_id for population_id, record in records_by_id.items() if record.source_chunk_id == chunk_id
            ]
            for population_id in stale_ids:
                del records_by_id[population_id]
            records_by_id.update({record.population_id: record for record in target_records})
        write_population_records(arguments.records_output_path, ordered_records())
        for chunk_id, _ in batch_results:
            progress_by_id[chunk_id] = PopulationProgressRecord(chunk_id=chunk_id, run_id=run_id)
        write_population_progress(
            arguments.progress_output_path,
            tuple(progress_by_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in progress_by_id),
        )

    logger.info(
        "Loaded {} completed conduct inventories; {} binding targets already current, {} remaining; "
        "batch size {}; max concurrency {}",
        len(inventories),
        len(extraction_targets) - len(remaining),
        len(remaining),
        population_settings.batch_size,
        arguments.max_concurrency,
    )
    if remaining:
        baml_client = BamlOntologyPopulationClient(provider_settings, population_settings)
        client = PersistedConductPopulationClient(baml_client, inventories)
        extract_population_from_document(
            remaining,
            chunks,
            client,
            prepared.bundle.metadata,
            population_settings,
            ontology_semantic_sha256=prepared.ontology_semantic_sha256,
            population_run_id=run_id,
            max_concurrency=arguments.max_concurrency,
            on_batch_processed=persist_batch,
        )
    records = ordered_records()
    write_population_records(arguments.records_output_path, records)
    write_population_progress(
        arguments.progress_output_path,
        tuple(progress_by_id[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in progress_by_id),
    )
    graph = materialize_population_graph(
        chunks,
        prepared.bundle.metadata,
        records,
        prepared.settings.instance_namespace,
    )
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Wrote {} population records and {} RDF triples to {}",
        len(records),
        len(graph),
        arguments.graph_output_path,
    )


if __name__ == "__main__":
    main()
