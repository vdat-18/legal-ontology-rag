"""Command-line interface for complete multi-turn conversation generation."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the multi-turn candidate-generation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate multi-turn candidates from conversation scenarios."
    )
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("conversations_path", type=Path, help="Conversation scenarios JSONL path")
    parser.add_argument("candidates_output_path", type=Path, help="Destination multi-turn candidates JSONL path")
    parser.add_argument(
        "--history-groundtruth",
        type=Path,
        required=True,
        help="Reviewed single-turn ground-truth JSONL reused as conversation history",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="Multi-turn checkpoint JSONL path; defaults beside candidates_output_path",
    )
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Generate complete conversations with conversation-level resume."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError
    from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import (
        MULTI_TURN_CANDIDATE_SCHEMA_VERSION,
        MULTI_TURN_GENERATION_PROMPT_VERSION,
        BamlMultiTurnGenerationClient,
        MultiTurnPipelineContext,
        MultiTurnProgressRecord,
        assemble_completed_conversations,
        generate_multi_turn_candidates,
    )
    from legal_rag.benchmark_synthesis.scenario_sampling import load_persona_descriptions
    from legal_rag.benchmark_synthesis.stage_cache import (
        build_multi_turn_generation_fingerprint,
        initialize_stage_cache,
        stage_cache_path,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_conversation_scenarios,
        read_legal_facts,
        read_multi_turn_progress,
        read_reviewed_history_candidates,
        write_multi_turn_candidates,
        write_multi_turn_progress,
    )
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)
    progress_output_path = arguments.progress_output or arguments.candidates_output_path.with_suffix(
        ".progress.jsonl"
    )

    facts = read_legal_facts(arguments.facts_path)
    facts_by_id = {fact.fact_id: fact for fact in facts}
    conversation_scenarios = read_conversation_scenarios(arguments.conversations_path)
    history_candidates = read_reviewed_history_candidates(arguments.history_groundtruth)
    history_candidates_by_scenario_id = {
        candidate.scenario_id: candidate for candidate in history_candidates
    }
    if len(history_candidates_by_scenario_id) != len(history_candidates):
        msg = "History ground truth contains duplicate scenario IDs"
        raise ValueError(msg)
    required_history_ids = {
        turn.anchor_scenario_id
        for conversation in conversation_scenarios
        for turn in conversation.turns[:-1]
    }
    missing_history_ids = required_history_ids - history_candidates_by_scenario_id.keys()
    if missing_history_ids:
        msg = f"History ground truth is missing {len(missing_history_ids)} referenced scenario(s)"
        raise ValueError(msg)
    mismatched_personas = {
        turn.anchor_scenario_id
        for conversation in conversation_scenarios
        for turn in conversation.turns[:-1]
        if history_candidates_by_scenario_id[turn.anchor_scenario_id].persona
        != conversation.persona
    }
    if mismatched_personas:
        msg = f"History ground truth has {len(mismatched_personas)} persona mismatch(es)"
        raise ValueError(msg)
    persona_descriptions = load_persona_descriptions(arguments.personas)

    fingerprint_path = stage_cache_path(arguments.candidates_output_path)
    execution_fingerprint = build_multi_turn_generation_fingerprint(
        facts,
        conversation_scenarios,
        history_candidates,
        persona_descriptions,
        synthesis_settings,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        prompt_version=MULTI_TURN_GENERATION_PROMPT_VERSION,
        candidate_schema_version=MULTI_TURN_CANDIDATE_SCHEMA_VERSION,
    )
    initialize_stage_cache(
        fingerprint_path,
        (arguments.candidates_output_path, progress_output_path),
        execution_fingerprint,
    )

    existing_progress = (
        read_multi_turn_progress(progress_output_path) if progress_output_path.exists() else ()
    )
    progress_by_task_id = {record.task_id: record for record in existing_progress}
    task_order = [
        conversation.conversation_scenario_id for conversation in conversation_scenarios
    ]

    context = MultiTurnPipelineContext(
        facts_by_id=facts_by_id,
        history_candidates_by_scenario_id=history_candidates_by_scenario_id,
        persona_descriptions=persona_descriptions,
        generation_client=BamlMultiTurnGenerationClient(settings, synthesis_settings),
        generator_model=settings.llm_model,
        llm_max_retries=synthesis_settings.llm_max_retries,
        llm_retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
    )

    progress_lock = threading.Lock()

    def _persist(records: Sequence[MultiTurnProgressRecord]) -> None:
        """Persist completed conversations atomically per file.

        Runs from worker threads under generate_multi_turn_candidates' bounded
        concurrency, so the shared progress dict and the two output files are
        serialized behind progress_lock.
        """
        with progress_lock:
            progress_by_task_id.update({record.task_id: record for record in records})
            ordered_progress = tuple(
                progress_by_task_id[task_id] for task_id in task_order if task_id in progress_by_task_id
            )
            write_multi_turn_progress(progress_output_path, ordered_progress)
            write_multi_turn_candidates(
                arguments.candidates_output_path,
                assemble_completed_conversations(
                    conversation_scenarios,
                    progress_by_task_id,
                    generator_model=settings.llm_model,
                ),
            )

    failures = generate_multi_turn_candidates(
        conversation_scenarios,
        progress_by_task_id,
        context,
        _persist,
        batch_size=synthesis_settings.multi_turn_generation_batch_size,
        max_concurrency=synthesis_settings.multi_turn_generation_max_concurrency,
    )

    conversations = assemble_completed_conversations(
        conversation_scenarios,
        progress_by_task_id,
        generator_model=settings.llm_model,
    )
    write_multi_turn_candidates(arguments.candidates_output_path, conversations)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("multi-turn generation", failures)
        raise incomplete_error

    logger.info(
        "Generated {} multi-turn conversations from {} conversation scenarios. Wrote {}",
        len(conversations),
        len(conversation_scenarios),
        arguments.candidates_output_path,
    )


if __name__ == "__main__":
    main()
