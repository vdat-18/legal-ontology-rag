"""Command-line interface for single-turn candidate generation, trusting LLM output as-is."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
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
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument(
        "--candidates-per-scenario", type=int, required=True, help="Number of generation attempts per scenario"
    )
    parser.add_argument(
        "--regenerate-task-id",
        action="append",
        default=[],
        help="Existing scenario_id:attempt_index task to regenerate; repeat as needed",
    )
    parser.add_argument(
        "--regenerate-task-ids",
        type=Path,
        help="JSON string array of existing task IDs to regenerate",
    )
    parser.add_argument(
        "--replace-scenario-id",
        action="append",
        default=[],
        help="Scenario ID removed by selective scenario replacement; repeat as needed",
    )
    parser.add_argument(
        "--replace-scenario-ids",
        type=Path,
        help="JSON string array of scenario IDs removed from the prior scenario set",
    )
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Generate candidates for every scenario, trusting LLM output with no gating checks."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.candidate_generation import (
        CANDIDATE_GENERATION_PROMPT_VERSION,
        CANDIDATE_SCHEMA_VERSION,
        LEGACY_CANDIDATE_GENERATION_PROMPT_VERSION,
        LEGACY_CANDIDATE_SCHEMA_VERSION,
        BamlCandidateGenerationClient,
    )
    from legal_rag.benchmark_synthesis.candidate_pipeline import (
        CandidatePipelineContext,
        generate_candidates,
    )
    from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
    from legal_rag.benchmark_synthesis.candidate_selection import (
        load_regenerate_task_ids,
        prior_scenarios_from_progress,
        validate_scenario_replacement,
        validate_selective_generation,
    )
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError
    from legal_rag.benchmark_synthesis.scenario_replacement import load_replace_scenario_ids
    from legal_rag.benchmark_synthesis.scenario_sampling import load_persona_descriptions
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_candidate_generation_artifact_fingerprint,
        build_candidate_generation_fingerprint,
        initialize_stage_cache,
        read_stage_cache_fingerprint,
        stage_cache_path,
        write_stage_cache_fingerprint,
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

    facts = read_legal_facts(arguments.facts_path)
    facts_by_id = {fact.fact_id: fact for fact in facts}
    scenarios = read_scenarios(arguments.scenarios_path)

    persona_descriptions = load_persona_descriptions(arguments.personas)
    fingerprint_path = stage_cache_path(arguments.candidates_output_path)
    execution_fingerprint = build_candidate_generation_fingerprint(
        facts,
        scenarios,
        persona_descriptions,
        synthesis_settings,
        candidates_per_scenario=arguments.candidates_per_scenario,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )

    existing_progress = read_candidate_progress(progress_output_path) if progress_output_path.exists() else ()
    progress_by_task_id = {record.task_id: record for record in existing_progress}
    task_order = [
        f"{scenario.scenario_id}:{attempt_index}"
        for scenario in scenarios
        for attempt_index in range(arguments.candidates_per_scenario)
    ]
    regenerate_ids = load_regenerate_task_ids(
        arguments.regenerate_task_id,
        arguments.regenerate_task_ids,
    )
    replace_scenario_ids = load_replace_scenario_ids(
        arguments.replace_scenario_id,
        arguments.replace_scenario_ids,
    )
    selective_mode = bool(regenerate_ids or replace_scenario_ids)
    if selective_mode:
        existing_fingerprint = read_stage_cache_fingerprint(fingerprint_path)
        prior_scenarios = scenarios
        added_task_ids: frozenset[str] = frozenset()
        if replace_scenario_ids:
            prior_scenarios = prior_scenarios_from_progress(existing_progress)
            added_scenario_ids = validate_scenario_replacement(
                prior_scenarios,
                scenarios,
                replace_scenario_ids,
            )
            current_task_ids = frozenset(task_order)
            progress_by_task_id = {
                task_id: record for task_id, record in progress_by_task_id.items() if task_id in current_task_ids
            }
            added_task_ids = frozenset(
                f"{scenario_id}:{attempt_index}"
                for scenario_id in added_scenario_ids
                for attempt_index in range(arguments.candidates_per_scenario)
            )
            logger.info(
                "Migrating candidate generation from {} removed scenario(s) to {} replacement scenario(s)",
                len(replace_scenario_ids),
                len(added_scenario_ids),
            )
        artifact_fingerprint = build_candidate_generation_artifact_fingerprint(
            facts,
            prior_scenarios,
            persona_descriptions,
            synthesis_settings,
            tuple(record.draft for record in existing_progress),
            candidates_per_scenario=arguments.candidates_per_scenario,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        )
        prior_schema_version = (
            LEGACY_CANDIDATE_SCHEMA_VERSION
            if existing_fingerprint.prompt_version == LEGACY_CANDIDATE_GENERATION_PROMPT_VERSION
            else CANDIDATE_SCHEMA_VERSION
        )
        prior_execution_fingerprint = build_candidate_generation_fingerprint(
            facts,
            prior_scenarios,
            persona_descriptions,
            synthesis_settings,
            candidates_per_scenario=arguments.candidates_per_scenario,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
            prompt_version=existing_fingerprint.prompt_version,
            candidate_schema_version=prior_schema_version,
        )
        if existing_fingerprint not in {prior_execution_fingerprint, artifact_fingerprint}:
            msg = "Existing candidate_generation artifacts belong to a different synthesis run"
            raise ValueError(msg)
        validate_selective_generation(
            frozenset(task_order),
            frozenset(progress_by_task_id),
            regenerate_ids | added_task_ids,
        )
        invalidated_ids = {
            task_id
            for task_id in regenerate_ids
            if (record := progress_by_task_id.get(task_id)) is None
            or record.draft.prompt_version != CANDIDATE_GENERATION_PROMPT_VERSION
        }
        progress_by_task_id = {
            task_id: record for task_id, record in progress_by_task_id.items() if task_id not in invalidated_ids
        }
        logger.info("Selectively invalidated {} candidate-generation task(s)", len(invalidated_ids))
    else:
        initialize_stage_cache(
            fingerprint_path,
            (arguments.candidates_output_path, progress_output_path),
            execution_fingerprint,
        )

    generation_client = BamlCandidateGenerationClient(settings, synthesis_settings)
    context = CandidatePipelineContext(
        facts_by_id=facts_by_id,
        persona_descriptions=persona_descriptions,
        generation_client=generation_client,
        generator_model=settings.llm_model,
        llm_max_retries=synthesis_settings.llm_max_retries,
        llm_retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
    )

    progress_lock = threading.Lock()

    def _persist(records: Sequence[CandidateProgressRecord]) -> None:
        """Persist one completed batch and its candidates atomically per file.

        Runs from worker threads under generate_candidates' bounded concurrency, so the
        shared progress dict and the two output files are serialized behind progress_lock.
        """
        with progress_lock:
            progress_by_task_id.update({record.task_id: record for record in records})
            ordered_progress = tuple(
                progress_by_task_id[task_id] for task_id in task_order if task_id in progress_by_task_id
            )
            write_candidate_progress(progress_output_path, ordered_progress)
            write_candidates(
                arguments.candidates_output_path,
                tuple(item.draft for item in ordered_progress if item.completed),
            )
            if selective_mode:
                write_stage_cache_fingerprint(
                    fingerprint_path,
                    build_candidate_generation_artifact_fingerprint(
                        facts,
                        scenarios,
                        persona_descriptions,
                        synthesis_settings,
                        tuple(item.draft for item in ordered_progress if item.completed),
                        candidates_per_scenario=arguments.candidates_per_scenario,
                        llm_base_url=settings.llm_base_url,
                        llm_model=settings.llm_model,
                    ),
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
    if selective_mode:
        write_stage_cache_fingerprint(
            fingerprint_path,
            build_candidate_generation_artifact_fingerprint(
                facts,
                scenarios,
                persona_descriptions,
                synthesis_settings,
                drafts,
                candidates_per_scenario=arguments.candidates_per_scenario,
                llm_base_url=settings.llm_base_url,
                llm_model=settings.llm_model,
            ),
        )
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
