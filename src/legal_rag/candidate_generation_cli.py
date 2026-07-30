"""Command-line interface for single-turn candidate generation and both gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the candidate-generation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate single-turn candidates from scenarios and run gate 1 and gate 2."
    )
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("scenarios_path", type=Path, help="Scenarios JSONL path")
    parser.add_argument("chunks_dir", type=Path, help="Directory containing one or more *.chunks.jsonl files")
    parser.add_argument("candidates_output_path", type=Path, help="Destination single-turn candidates JSONL path")
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="Candidate-stage checkpoint JSONL path; defaults beside candidates_output_path",
    )
    parser.add_argument("--query-taxonomy", type=Path, required=True, help="Query-taxonomy config JSON path")
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument("--evolution", type=Path, required=True, help="Evolution-operation config JSON path")
    parser.add_argument(
        "--candidates-per-scenario", type=int, required=True, help="Number of generation attempts per scenario"
    )
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Generate candidates for every scenario, then run gate 1 and, when it passes, gate 2."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.candidate_generation import BamlCandidateGenerationClient
    from legal_rag.benchmark_synthesis.candidate_pipeline import (
        CandidatePipelineContext,
        process_candidate_task,
    )
    from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.llm_execution import (
        IncompleteLlmTasksError,
        LlmTaskExhaustedError,
        LlmTaskFailure,
    )
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        load_evolution_config,
        load_persona_descriptions,
        load_query_taxonomy,
        load_query_type_descriptions,
    )
    from legal_rag.benchmark_synthesis.verification import BamlVerificationClient
    from legal_rag.benchmark_synthesis.writer import (
        read_candidate_progress,
        read_legal_facts,
        read_scenarios,
        write_candidate_progress,
        write_candidates,
    )
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)
    progress_output_path = arguments.progress_output or arguments.candidates_output_path.with_suffix(".progress.jsonl")

    facts_by_id = {fact.fact_id: fact for fact in read_legal_facts(arguments.facts_path)}
    scenarios = read_scenarios(arguments.scenarios_path)

    chunks_by_id = {}
    for chunks_path in sorted(arguments.chunks_dir.glob("*.chunks.jsonl")):
        for chunk in read_chunks(chunks_path):
            chunks_by_id[chunk.chunk_id] = chunk
    logger.info("Loaded {} chunks from {}", len(chunks_by_id), arguments.chunks_dir)

    query_taxonomy = load_query_taxonomy(arguments.query_taxonomy)
    query_type_descriptions = load_query_type_descriptions(arguments.query_taxonomy)
    persona_descriptions = load_persona_descriptions(arguments.personas)
    evolution_config = load_evolution_config(arguments.evolution)
    evolution_operations = evolution_config.operation_ids[: evolution_config.max_rounds]

    generation_client = BamlCandidateGenerationClient(settings, synthesis_settings)
    verification_client = BamlVerificationClient(settings, synthesis_settings)
    context = CandidatePipelineContext(
        facts_by_id=facts_by_id,
        chunks_by_id=chunks_by_id,
        persona_descriptions=persona_descriptions,
        query_type_descriptions=query_type_descriptions,
        query_type_catalog=query_taxonomy.query_type_ids,
        evolution_operations=evolution_operations,
        generation_client=generation_client,
        verification_client=verification_client,
        generator_model=settings.llm_model,
        verifier_model=settings.llm_model,
        span_match_threshold=synthesis_settings.span_match_threshold,
        max_quality_retries=synthesis_settings.max_quality_retries,
        llm_max_retries=synthesis_settings.llm_max_retries,
    )

    existing_progress = read_candidate_progress(progress_output_path) if progress_output_path.exists() else ()
    progress_by_task_id = {record.task_id: record for record in existing_progress}
    task_order = [
        f"{scenario.scenario_id}:{attempt_index}"
        for scenario in scenarios
        for attempt_index in range(arguments.candidates_per_scenario)
    ]

    def _persist(record: CandidateProgressRecord) -> None:
        """Persist one stage checkpoint and every completed candidate atomically per file."""
        progress_by_task_id[record.task_id] = record
        ordered_progress = tuple(
            progress_by_task_id[task_id] for task_id in task_order if task_id in progress_by_task_id
        )
        write_candidate_progress(progress_output_path, ordered_progress)
        write_candidates(
            arguments.candidates_output_path,
            tuple(item.candidate_record() for item in ordered_progress if item.completed),
        )

    failures: list[LlmTaskFailure] = []
    for scenario in scenarios:
        for attempt_index in range(arguments.candidates_per_scenario):
            task_id = f"{scenario.scenario_id}:{attempt_index}"
            progress = progress_by_task_id.get(task_id)
            if progress is not None and progress.completed:
                continue
            try:
                process_candidate_task(scenario, attempt_index, progress, context, _persist)
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=task_id, error=str(error)))

    records = tuple(
        progress_by_task_id[task_id].candidate_record()
        for task_id in task_order
        if task_id in progress_by_task_id and progress_by_task_id[task_id].completed
    )
    write_candidates(arguments.candidates_output_path, records)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("candidate generation", failures)
        raise incomplete_error

    rejected_binding = sum(1 for record in records if record.review_status == "rejected_by_binding_gate")
    rejected_verification = sum(1 for record in records if record.review_status == "rejected_by_verification")
    verified = sum(1 for record in records if record.review_status == "automatically_verified")
    logger.info(
        "Generated {} candidates from {} scenarios: {} rejected by gate 1, {} rejected by gate 2, "
        "{} automatically verified. Wrote {}",
        len(records),
        len(scenarios),
        rejected_binding,
        rejected_verification,
        verified,
        arguments.candidates_output_path,
    )


if __name__ == "__main__":
    main()
