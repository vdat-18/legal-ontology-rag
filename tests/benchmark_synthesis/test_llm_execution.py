"""Tests for shared synthesis LLM retry execution."""

from __future__ import annotations

import pytest

from legal_rag.benchmark_synthesis import llm_execution
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError, execute_llm_task


def test_execute_llm_task_waits_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each retry waits for the configured delay before starting the next attempt."""
    attempts = 0
    delays: list[float] = []

    def _operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError
        return "ok"

    monkeypatch.setattr(llm_execution.time, "sleep", delays.append)

    result = execute_llm_task("task", _operation, max_retries=2, retry_delay_seconds=120)

    assert result == "ok"
    assert delays == [120, 120]


def test_execute_llm_task_does_not_wait_after_final_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exhausted task raises immediately after its last configured attempt."""
    delays: list[float] = []

    def _operation() -> None:
        raise ValueError

    monkeypatch.setattr(llm_execution.time, "sleep", delays.append)

    with pytest.raises(LlmTaskExhaustedError):
        execute_llm_task("task", _operation, max_retries=2, retry_delay_seconds=120)

    assert delays == [120, 120]
