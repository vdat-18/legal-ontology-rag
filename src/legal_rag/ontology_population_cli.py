"""Command-line interface for source-grounded ontology population."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology-population command-line parser."""
    parser = argparse.ArgumentParser(
        description="Extract ontology population records from chunks and write a SHACL-validated RDF graph."
    )
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("contract_path", type=Path, help="Versioned ontology population contract YAML path")
    parser.add_argument("records_output_path", type=Path, help="Reviewable population-record JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Population-progress JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Validated Turtle instance-graph destination")
    parser.add_argument("--target-limit", type=int, help="Maximum accepted chunks selected for a pilot run")
    parser.add_argument("--preflight-report-path", type=Path, help="Optional machine-readable readiness report")
    return parser


def main() -> None:
    """Extract, persist, materialize, and validate ontology instances."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.extraction import (
        BamlOntologyPopulationClient,
        extract_population_from_document,
    )
    from legal_rag.ontology_population.materializer import (
        materialize_population_graph,
        validate_population_graph,
        write_population_graph,
    )
    from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
    from legal_rag.ontology_population.preflight import (
        load_population_inputs,
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
    from legal_rag.ontology_validation.paths import OntologyPaths
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    population_settings = PopulationSettings.from_path(arguments.contract_path)
    bundle = load_population_inputs(arguments.chunks_path, arguments.screening_path, arguments.metadata_path)
    preflight = run_population_preflight(
        bundle,
        population_settings,
        (
            ("chunks", arguments.chunks_path),
            ("screening", arguments.screening_path),
            ("metadata", arguments.metadata_path),
            ("population_contract", arguments.contract_path),
        ),
        datetime.now(UTC),
        target_limit=arguments.target_limit,
    )
    if arguments.preflight_report_path is not None:
        write_json_atomic(arguments.preflight_report_path, preflight.to_dict())
    if preflight.overall_status != "ready":
        report_hint = f"; inspect {arguments.preflight_report_path}" if arguments.preflight_report_path else ""
        msg = f"Ontology population inputs are blocked{report_hint}"
        raise ValueError(msg)

    chunks = bundle.chunks
    metadata = bundle.metadata
    accepted_chunks = select_population_targets(bundle, arguments.target_limit)
    existing_records = (
        read_population_records(arguments.records_output_path) if arguments.records_output_path.exists() else ()
    )
    records_by_id = {record.population_id: record for record in existing_records}
    existing_progress = (
        read_population_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    validate_population_resume_state(accepted_chunks, existing_records, existing_progress)
    processed_ids = {record.chunk_id for record in existing_progress}
    remaining_targets = tuple(chunk for chunk in accepted_chunks if chunk.chunk_id not in processed_ids)

    logger.info(
        "Loaded {} accepted chunks from {}; {} already processed, {} remaining",
        len(accepted_chunks),
        arguments.chunks_path,
        len(accepted_chunks) - len(remaining_targets),
        len(remaining_targets),
    )

    def persist_batch(batch_results: Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]) -> None:
        """Persist records before progress so interrupted writes replay safely."""
        for _, target_records in batch_results:
            records_by_id.update({record.population_id: record for record in target_records})
        write_population_records(arguments.records_output_path, tuple(records_by_id.values()))
        processed_ids.update(chunk_id for chunk_id, _ in batch_results)
        write_population_progress(
            arguments.progress_output_path,
            tuple(
                PopulationProgressRecord(chunk_id=chunk.chunk_id) for chunk in chunks if chunk.chunk_id in processed_ids
            ),
        )

    if remaining_targets:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
        settings = Settings.from_environment()
        client = BamlOntologyPopulationClient(settings, population_settings)
        extract_population_from_document(
            remaining_targets,
            chunks,
            client,
            metadata,
            population_settings,
            on_batch_processed=persist_batch,
        )

    records = tuple(records_by_id.values())
    write_population_records(arguments.records_output_path, records)
    write_population_progress(
        arguments.progress_output_path,
        tuple(PopulationProgressRecord(chunk_id=chunk.chunk_id) for chunk in chunks if chunk.chunk_id in processed_ids),
    )
    graph = materialize_population_graph(chunks, metadata, records, population_settings.instance_namespace)
    project_root = Path(__file__).resolve().parents[2]
    validate_population_graph(graph, OntologyPaths(project_root))
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Wrote {} population records and {} validated RDF triples to {}",
        len(records),
        len(graph),
        arguments.graph_output_path,
    )


if __name__ == "__main__":
    main()
