"""Command-line interface for source-grounded ontology population."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _parser() -> argparse.ArgumentParser:
    """Build the ontology-population command-line parser."""
    parser = argparse.ArgumentParser(
        description="Extract typed ontology population records from chunks and materialize an RDF graph."
    )
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("contract_path", type=Path, help="Versioned ontology population contract YAML path")
    parser.add_argument("records_output_path", type=Path, help="Population-record JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Population-progress JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Turtle instance-graph destination")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target-limit", type=int, help="Maximum accepted chunks selected for a pilot run")
    target_group.add_argument("--target-ids-path", type=Path, help="Reviewed pilot target-selection JSON path")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY,
        help="Maximum concurrent ontology-population batches",
    )
    parser.add_argument("--preflight-report-path", type=Path, help="Optional machine-readable readiness report")
    return parser


def main() -> None:
    """Extract, persist, and materialize ontology instances."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.extraction import (
        BamlOntologyPopulationClient,
        build_population_run_id,
        extract_population_from_document,
    )
    from legal_rag.ontology_population.materializer import (
        materialize_population_graph,
        write_population_graph,
    )
    from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
    from legal_rag.ontology_population.preflight import (
        load_population_inputs,
        load_population_target_selection,
        population_ontology_semantic_sha256,
        run_population_preflight,
        select_population_targets,
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
    project_root = Path(__file__).resolve().parents[2]
    ontology_semantic_sha256 = population_ontology_semantic_sha256(project_root)
    population_settings = PopulationSettings.from_path(arguments.contract_path)
    bundle = load_population_inputs(arguments.chunks_path, arguments.screening_path, arguments.metadata_path)
    target_selection = (
        load_population_target_selection(arguments.target_ids_path) if arguments.target_ids_path is not None else None
    )
    artifact_paths = [
        ("chunks", arguments.chunks_path),
        ("screening", arguments.screening_path),
        ("metadata", arguments.metadata_path),
        ("population_contract", arguments.contract_path),
    ]
    if arguments.target_ids_path is not None:
        artifact_paths.append(("target_selection", arguments.target_ids_path))
    preflight = run_population_preflight(
        bundle,
        population_settings,
        tuple(artifact_paths),
        datetime.now(UTC),
        ontology_semantic_sha256=ontology_semantic_sha256,
        target_limit=arguments.target_limit,
        target_selection=target_selection,
    )
    if arguments.preflight_report_path is not None:
        write_json_atomic(arguments.preflight_report_path, preflight.to_dict())
    if preflight.overall_status != "ready":
        report_hint = f"; inspect {arguments.preflight_report_path}" if arguments.preflight_report_path else ""
        msg = f"Ontology population inputs are blocked{report_hint}"
        raise ValueError(msg)

    chunks = bundle.chunks
    metadata = bundle.metadata
    accepted_chunks = select_population_targets(bundle, arguments.target_limit, target_selection)
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    settings = Settings.from_environment()
    run_id = build_population_run_id(preflight.pilot_run_id, settings)
    existing_records = (
        read_population_records(arguments.records_output_path) if arguments.records_output_path.exists() else ()
    )
    records_by_id = {record.population_id: record for record in existing_records}
    chunk_order = {chunk.chunk_id: chunk.chunk_index for chunk in chunks}
    existing_progress = (
        read_population_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    validate_population_resume_state(
        accepted_chunks,
        existing_records,
        existing_progress,
        expected_run_id=run_id,
        expected_ontology_semantic_sha256=ontology_semantic_sha256,
    )
    processed_ids = {record.chunk_id for record in existing_progress}
    remaining_targets = tuple(chunk for chunk in accepted_chunks if chunk.chunk_id not in processed_ids)

    logger.info(
        "Loaded {} accepted chunks from {}; {} already processed, {} remaining; max concurrency {}",
        len(accepted_chunks),
        arguments.chunks_path,
        len(accepted_chunks) - len(remaining_targets),
        len(remaining_targets),
        arguments.max_concurrency,
    )

    def ordered_records() -> tuple[OntologyPopulationRecord, ...]:
        """Return stable source-ordered records regardless of completion order."""
        return tuple(
            sorted(
                records_by_id.values(), key=lambda record: (chunk_order[record.source_chunk_id], record.population_id)
            )
        )

    def persist_batch(batch_results: Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]) -> None:
        """Persist records before progress so interrupted writes replay safely."""
        for chunk_id, target_records in batch_results:
            stale_ids = [
                population_id for population_id, record in records_by_id.items() if record.source_chunk_id == chunk_id
            ]
            for population_id in stale_ids:
                del records_by_id[population_id]
            records_by_id.update({record.population_id: record for record in target_records})
        write_population_records(arguments.records_output_path, ordered_records())
        processed_ids.update(chunk_id for chunk_id, _ in batch_results)
        write_population_progress(
            arguments.progress_output_path,
            tuple(
                PopulationProgressRecord(chunk_id=chunk.chunk_id, run_id=run_id)
                for chunk in chunks
                if chunk.chunk_id in processed_ids
            ),
        )

    if remaining_targets:
        client = BamlOntologyPopulationClient(settings, population_settings)
        extract_population_from_document(
            remaining_targets,
            chunks,
            client,
            metadata,
            population_settings,
            ontology_semantic_sha256=ontology_semantic_sha256,
            population_run_id=run_id,
            max_concurrency=arguments.max_concurrency,
            on_batch_processed=persist_batch,
        )

    records = ordered_records()
    write_population_records(arguments.records_output_path, records)
    write_population_progress(
        arguments.progress_output_path,
        tuple(
            PopulationProgressRecord(chunk_id=chunk.chunk_id, run_id=run_id)
            for chunk in chunks
            if chunk.chunk_id in processed_ids
        ),
    )
    graph = materialize_population_graph(chunks, metadata, records, population_settings.instance_namespace)
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Wrote {} population records and {} RDF triples to {}",
        len(records),
        len(graph),
        arguments.graph_output_path,
    )


if __name__ == "__main__":
    main()
