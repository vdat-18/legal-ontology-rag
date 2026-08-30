"""Execute independent ontology proposal batches with bounded concurrency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)

_T = TypeVar("_T")


def execute_proposal_batches(
    tasks: Sequence[tuple[str, Callable[[], _T]]],
    *,
    stage: str,
    max_concurrency: int,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_processed: Callable[[_T], None] | None = None,
) -> tuple[_T, ...]:
    """Run batches concurrently, checkpoint successes, and preserve input order."""
    results_by_index: dict[int, _T] = {}
    failures: list[LlmTaskFailure] = []
    exhausted_errors: list[LlmTaskExhaustedError] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                execute_llm_task,
                task_key,
                operation,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            ): (batch_index, task_key)
            for batch_index, (task_key, operation) in enumerate(tasks)
        }
        for future in as_completed(futures):
            batch_index, task_key = futures[future]
            try:
                batch_result = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
                exhausted_errors.append(error)
                continue
            results_by_index[batch_index] = batch_result
            if on_batch_processed is not None:
                on_batch_processed(batch_result)
    if failures:
        if len(exhausted_errors) == 1:
            raise exhausted_errors[0]
        raise IncompleteLlmTasksError.for_stage(stage, failures)
    return tuple(results_by_index[index] for index in range(len(tasks)))
