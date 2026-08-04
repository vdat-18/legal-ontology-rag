"""Checkpointed orchestration for one single-turn candidate work item."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateDraft,
    CandidateGenerationClient,
    evolve_candidate,
    generate_candidate,
)
from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario


@dataclass(frozen=True, slots=True)
class CandidatePipelineContext:
    """Dependencies and configured limits shared by candidate work items."""

    facts_by_id: Mapping[str, LegalFact]
    persona_descriptions: Mapping[str, str]
    query_type_descriptions: Mapping[str, str]
    evolution_operations: Sequence[str]
    generation_client: CandidateGenerationClient
    generator_model: str
    llm_max_retries: int
    llm_retry_delay_seconds: float


def process_candidate_task(
    scenario: Scenario,
    attempt_index: int,
    progress: CandidateProgressRecord | None,
    context: CandidatePipelineContext,
    checkpoint: Callable[[CandidateProgressRecord], None],
) -> CandidateProgressRecord:
    """Resume one scenario attempt from its latest persisted stage and finish it."""
    task_id = f"{scenario.scenario_id}:{attempt_index}"
    if progress is None:
        draft = execute_llm_task(
            f"{task_id}:generation",
            lambda: generate_candidate(
                scenario,
                context.facts_by_id,
                context.persona_descriptions[scenario.persona],
                context.query_type_descriptions[scenario.query_type],
                context.generation_client,
                generator_model=context.generator_model,
            ),
            max_retries=context.llm_max_retries,
            retry_delay_seconds=context.llm_retry_delay_seconds,
        )
        progress = CandidateProgressRecord(task_id, scenario.scenario_id, attempt_index, "generated", draft)
        checkpoint(progress)

    if len(progress.draft.evolutions) < len(context.evolution_operations):
        progress = _evolve_with_checkpoints(progress, scenario, context, checkpoint)

    completed = CandidateProgressRecord(task_id, scenario.scenario_id, attempt_index, "completed", progress.draft)
    checkpoint(completed)
    return completed


def generate_candidates(
    scenarios: Sequence[Scenario],
    candidates_per_scenario: int,
    progress_by_task_id: Mapping[str, CandidateProgressRecord],
    context: CandidatePipelineContext,
    checkpoint: Callable[[CandidateProgressRecord], None],
    *,
    max_concurrency: int,
) -> tuple[LlmTaskFailure, ...]:
    """Generate every not-yet-completed scenario attempt in bounded concurrent workers.

    process_candidate_task persists after each of its own internal stages, so checkpoint
    runs from worker threads rather than only after a task finishes; the caller must make
    checkpoint safe for concurrent invocation (e.g. with a lock).
    """
    pending: list[tuple[Scenario, int]] = []
    for scenario in scenarios:
        for attempt_index in range(candidates_per_scenario):
            record = progress_by_task_id.get(f"{scenario.scenario_id}:{attempt_index}")
            if record is None or not record.completed:
                pending.append((scenario, attempt_index))

    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                process_candidate_task,
                scenario,
                attempt_index,
                progress_by_task_id.get(f"{scenario.scenario_id}:{attempt_index}"),
                context,
                checkpoint,
            ): f"{scenario.scenario_id}:{attempt_index}"
            for scenario, attempt_index in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=task_id, error=str(error)))
    return tuple(failures)


def _evolve_with_checkpoints(
    progress: CandidateProgressRecord,
    scenario: Scenario,
    context: CandidatePipelineContext,
    checkpoint: Callable[[CandidateProgressRecord], None],
) -> CandidateProgressRecord:
    """Apply remaining evolutions while persisting every successful call."""
    current = progress

    def _checkpoint_draft(draft: CandidateDraft) -> None:
        nonlocal current
        current = CandidateProgressRecord(
            progress.task_id,
            scenario.scenario_id,
            progress.attempt_index,
            "evolution",
            draft,
        )
        checkpoint(current)

    draft = evolve_candidate(
        current.draft,
        context.evolution_operations,
        context.generation_client,
        llm_max_retries=context.llm_max_retries,
        llm_retry_delay_seconds=context.llm_retry_delay_seconds,
        on_operation_applied=_checkpoint_draft,
    )
    current = CandidateProgressRecord(
        progress.task_id,
        scenario.scenario_id,
        progress.attempt_index,
        "evolved",
        draft,
    )
    checkpoint(current)
    return current
