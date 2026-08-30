"""Tests for concurrent ontology proposal-batch execution."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError, LlmTaskExhaustedError
from legal_rag.ontology_proposal_execution import execute_proposal_batches


def test_execute_proposal_batches_runs_concurrently_and_preserves_order() -> None:
    """Completed batches checkpoint immediately while final results follow input order."""
    barrier = threading.Barrier(2)
    main_thread_id = threading.get_ident()
    callback_threads = []
    checkpointed = []

    def operation(value: str) -> Callable[[], str]:
        def run() -> str:
            barrier.wait(timeout=5)
            return value

        return run

    def checkpoint(value: str) -> None:
        callback_threads.append(threading.get_ident())
        checkpointed.append(value)

    results = execute_proposal_batches(
        (("first", operation("first-result")), ("second", operation("second-result"))),
        stage="test stage",
        max_concurrency=2,
        max_retries=0,
        retry_delay_seconds=0,
        on_batch_processed=checkpoint,
    )

    assert results == ("first-result", "second-result")
    assert set(checkpointed) == {"first-result", "second-result"}
    assert callback_threads == [main_thread_id, main_thread_id]


def test_execute_proposal_batches_preserves_single_exhausted_error() -> None:
    """One failed batch keeps the established exhausted-task exception contract."""
    exhausted_error = LlmTaskExhaustedError.for_task("failed")

    def fail() -> str:
        raise exhausted_error

    with pytest.raises(LlmTaskExhaustedError) as error:
        execute_proposal_batches(
            (("failed", fail),),
            stage="test stage",
            max_concurrency=1,
            max_retries=0,
            retry_delay_seconds=0,
        )

    assert error.value is exhausted_error


def test_execute_proposal_batches_aggregates_multiple_exhausted_errors() -> None:
    """Independent failed batches are reported together after successful work finishes."""
    checkpointed = []

    def exhaust(task_key: str) -> Callable[[], str]:
        def fail() -> str:
            raise LlmTaskExhaustedError.for_task(task_key)

        return fail

    with pytest.raises(IncompleteLlmTasksError) as error:
        execute_proposal_batches(
            (
                ("first", exhaust("first")),
                ("success", lambda: "completed"),
                ("second", exhaust("second")),
            ),
            stage="test stage",
            max_concurrency=3,
            max_retries=0,
            retry_delay_seconds=0,
            on_batch_processed=checkpointed.append,
        )

    assert error.value.stage == "test stage"
    assert {failure.task_key for failure in error.value.failures} == {"first", "second"}
    assert checkpointed == ["completed"]
