"""Checkpointed orchestration for one single-turn candidate work item."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import cast

from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateGenerationClient,
    generate_candidate,
    generate_candidate_batch,
)
from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.parser.hashing import sha256_json


@dataclass(frozen=True, slots=True)
class CandidatePipelineContext:
    """Dependencies and configured limits shared by candidate work items."""

    facts_by_id: Mapping[str, LegalFact]
    persona_descriptions: Mapping[str, str]
    generation_client: CandidateGenerationClient
    generator_model: str
    llm_max_retries: int
    llm_retry_delay_seconds: float


def process_candidate_task(
    scenario: Scenario,
    attempt_index: int,
    progress: CandidateProgressRecord | None,
    context: CandidatePipelineContext,
    checkpoint: Callable[[Sequence[CandidateProgressRecord]], None],
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
                context.generation_client,
                generator_model=context.generator_model,
            ),
            max_retries=context.llm_max_retries,
            retry_delay_seconds=context.llm_retry_delay_seconds,
        )
    else:
        draft = progress.draft

    completed = CandidateProgressRecord(task_id, scenario.scenario_id, attempt_index, "completed", draft)
    checkpoint((completed,))
    return completed


def generate_candidates(
    scenarios: Sequence[Scenario],
    candidates_per_scenario: int,
    progress_by_task_id: Mapping[str, CandidateProgressRecord],
    context: CandidatePipelineContext,
    checkpoint: Callable[[Sequence[CandidateProgressRecord]], None],
    *,
    batch_size: int,
    max_concurrency: int,
) -> tuple[LlmTaskFailure, ...]:
    """Generate pending scenario attempts in bounded concurrent LLM batches.

    process_candidate_batch persists each completed batch, so checkpoint runs from worker
    threads; the caller must make checkpoint safe for concurrent invocation (e.g. with a lock).
    """
    pending: list[tuple[Scenario, int]] = []
    for scenario in scenarios:
        for attempt_index in range(candidates_per_scenario):
            record = progress_by_task_id.get(f"{scenario.scenario_id}:{attempt_index}")
            if record is None or not record.completed:
                pending.append((scenario, attempt_index))

    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                process_candidate_batch,
                batch,
                progress_by_task_id,
                context,
                checkpoint,
            ): tuple(f"{scenario.scenario_id}:{attempt_index}" for scenario, attempt_index in batch)
            for batch in batches
        }
        for future in as_completed(futures):
            task_ids = futures[future]
            try:
                future.result()
            except LlmTaskExhaustedError as error:
                failures.extend(LlmTaskFailure(task_key=task_id, error=str(error)) for task_id in task_ids)
    return tuple(failures)


def process_candidate_batch(
    tasks: Sequence[tuple[Scenario, int]],
    progress_by_task_id: Mapping[str, CandidateProgressRecord],
    context: CandidatePipelineContext,
    checkpoint: Callable[[Sequence[CandidateProgressRecord]], None],
) -> None:
    """Resume and complete one batch with one consolidated checkpoint write."""
    task_metadata = {
        f"{scenario.scenario_id}:{attempt_index}": (scenario, attempt_index) for scenario, attempt_index in tasks
    }
    current = {task_id: progress_by_task_id.get(task_id) for task_id in task_metadata}
    generation_tasks = [
        (task_id, scenario) for task_id, (scenario, _) in task_metadata.items() if current[task_id] is None
    ]
    if generation_tasks:
        generated = execute_llm_task(
            _batch_task_key(tuple(task_id for task_id, _ in generation_tasks), "generation"),
            lambda: generate_candidate_batch(
                generation_tasks,
                context.facts_by_id,
                context.persona_descriptions,
                context.generation_client,
                generator_model=context.generator_model,
            ),
            max_retries=context.llm_max_retries,
            retry_delay_seconds=context.llm_retry_delay_seconds,
        )
        for task_id, draft in generated:
            scenario, attempt_index = task_metadata[task_id]
            current[task_id] = CandidateProgressRecord(
                task_id,
                scenario.scenario_id,
                attempt_index,
                "completed",
                draft,
            )

    completed_records: list[CandidateProgressRecord] = []
    for task_id, record in current.items():
        completed_record = cast(CandidateProgressRecord, record)
        completed = CandidateProgressRecord(
            task_id,
            completed_record.scenario_id,
            completed_record.attempt_index,
            "completed",
            completed_record.draft,
        )
        completed_records.append(completed)
    checkpoint(completed_records)


def _batch_task_key(task_ids: Sequence[str], stage: str) -> str:
    """Build one stable retry key for an ordered candidate batch."""
    return f"candidate-batch:{sha256_json({'task_ids': list(task_ids), 'stage': stage})}:{stage}"
