"""Shared bounded retry behavior for ground-truth synthesis LLM calls."""

from __future__ import annotations

import http.client
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from loguru import logger

from legal_rag.baml_transport import BamlTransportError

_T = TypeVar("_T")

_RETRYABLE_LLM_ERRORS = (
    BamlTransportError,
    http.client.HTTPException,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


@dataclass(frozen=True, slots=True)
class LlmTaskFailure:
    """One LLM work item that remained incomplete after bounded retries."""

    task_key: str
    error: str


class IncompleteLlmTasksError(RuntimeError):
    """Raised after a stage finishes its successful work but leaves failed tasks."""

    def __init__(self, stage: str, failures: Sequence[LlmTaskFailure]) -> None:
        """Describe every task that must be resumed before downstream stages run."""
        self.stage = stage
        self.failures = tuple(failures)
        keys = ", ".join(failure.task_key for failure in self.failures)
        super().__init__(f"{stage} left {len(self.failures)} incomplete task(s): {keys}")

    @classmethod
    def for_stage(cls, stage: str, failures: Sequence[LlmTaskFailure]) -> IncompleteLlmTasksError:
        """Build an incomplete-stage error from accumulated task failures."""
        return cls(stage, failures)


class LlmTaskExhaustedError(RuntimeError):
    """Raised when one LLM task exhausts its configured attempts."""

    @classmethod
    def for_task(cls, task_key: str) -> LlmTaskExhaustedError:
        """Build an exhausted-retry error for one task key."""
        return cls(f"LLM task {task_key} exhausted retries")


def execute_llm_task(
    task_key: str,
    operation: Callable[[], _T],
    *,
    max_retries: int,
    retry_delay_seconds: float,
) -> _T:
    """Execute one LLM operation with bounded retries and preserve the final cause."""
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except _RETRYABLE_LLM_ERRORS as error:
            logger.warning(
                "LLM task {} failed on attempt {}/{}: {}",
                task_key,
                attempt + 1,
                max_retries + 1,
                error,
            )
            if attempt == max_retries:
                exhausted_error = LlmTaskExhaustedError.for_task(task_key)
                raise exhausted_error from error
            logger.info("Retrying LLM task {} in {} seconds", task_key, retry_delay_seconds)
            time.sleep(retry_delay_seconds)
    raise AssertionError("unreachable")
