"""Command-line interface for single-turn candidate generation, trusting LLM output as-is."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the candidate-generation command-line interface."""
    parser = argparse.ArgumentParser(description="Generate single-turn candidates from scenarios.")
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("scenarios_path", type=Path, help="Scenarios JSONL path")
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
    """Generate candidates for every scenario, trusting LLM output with no gating checks."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.candidate_generation import BamlCandidateGenerationClient
    from legal_rag.benchmark_synthesis.candidate_pipeline import (
        CandidatePipelineContext,
        generate_candidates,
    )
    from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        load_evolution_config,
        load_persona_descriptions,
        load_query_type_descriptions,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_candidate_progress,
        read_legal_facts,
        read_scenarios,
        write_candidate_progress,
        write_candidates,
    )
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

    query_type_descriptions = load_query_type_descriptions(arguments.query_taxonomy)
    persona_descriptions = load_persona_descriptions(arguments.personas)
    evolution_config = load_evolution_config(arguments.evolution)
    evolution_operations = evolution_config.operation_ids[: evolution_config.max_rounds]

    generation_client = BamlCandidateGenerationClient(settings, synthesis_settings)
    context = CandidatePipelineContext(
        facts_by_id=facts_by_id,
        persona_descriptions=persona_descriptions,
        query_type_descriptions=query_type_descriptions,
        evolution_operations=evolution_operations,
        generation_client=generation_client,
        generator_model=settings.llm_model,
        llm_max_retries=synthesis_settings.llm_max_retries,
        llm_retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
    )

    existing_progress = read_candidate_progress(progress_output_path) if progress_output_path.exists() else ()
    progress_by_task_id = {record.task_id: record for record in existing_progress}
    task_order = [
        f"{scenario.scenario_id}:{attempt_index}"
        for scenario in scenarios
        for attempt_index in range(arguments.candidates_per_scenario)
    ]

    progress_lock = threading.Lock()

    def _persist(record: CandidateProgressRecord) -> None:
        """Persist one stage checkpoint and every completed candidate atomically per file.

        Runs from worker threads under generate_candidates' bounded concurrency, so the
        shared progress dict and the two output files are serialized behind progress_lock.
        """
        with progress_lock:
            progress_by_task_id[record.task_id] = record
            ordered_progress = tuple(
                progress_by_task_id[task_id] for task_id in task_order if task_id in progress_by_task_id
            )
            write_candidate_progress(progress_output_path, ordered_progress)
            write_candidates(
                arguments.candidates_output_path,
                tuple(item.draft for item in ordered_progress if item.completed),
            )

    failures = generate_candidates(
        scenarios,
        arguments.candidates_per_scenario,
        progress_by_task_id,
        context,
        _persist,
        batch_size=synthesis_settings.candidate_generation_batch_size,
        max_concurrency=synthesis_settings.candidate_generation_max_concurrency,
    )

    drafts = tuple(
        progress_by_task_id[task_id].draft
        for task_id in task_order
        if task_id in progress_by_task_id and progress_by_task_id[task_id].completed
    )
    write_candidates(arguments.candidates_output_path, drafts)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("candidate generation", failures)
        raise incomplete_error

    logger.info(
        "Generated {} candidates from {} scenarios. Wrote {}",
        len(drafts),
        len(scenarios),
        arguments.candidates_output_path,
    )


if __name__ == "__main__":
    main()
