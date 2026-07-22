"""Tests for the standalone Gemini key checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_gemini_keys.py"


def _load_script() -> ModuleType:
    """Load the checker script as a testable module."""
    spec = importlib.util.spec_from_file_location("check_gemini_keys", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_script()


def test_load_keys_filters_prefix_comments_and_empty_values(tmp_path: Path) -> None:
    """Load only populated variables matching the requested prefix."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "API_KEY_1=secret-one # owner\nAPI_KEY_2=\nOTHER_KEY=secret-two\n",
        encoding="utf-8",
    )

    assert checker.load_keys(env_file, "API_KEY_") == {"API_KEY_1": "secret-one"}


def test_check_key_reports_working_without_exposing_key() -> None:
    """Report success after model discovery and generation."""
    responses = iter(
        [
            checker.HttpResult(
                status_code=200,
                payload={
                    "models": [
                        {
                            "name": "models/test-model",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            ),
            checker.HttpResult(status_code=200, payload={"candidates": []}),
        ]
    )
    seen: list[tuple[Any, ...]] = []

    def requester(*args: Any) -> Any:
        seen.append(args)
        return next(responses)

    result = checker.check_key("API_KEY_1", "top-secret", "https://example.test", 1.0, requester=requester)

    assert result.status == "WORKING"
    assert result.model == "models/test-model"
    assert "top-secret" not in repr(result)
    assert len(seen) == 2


def test_validate_api_base_rejects_non_google_host() -> None:
    """Do not allow API keys to be sent to an arbitrary configured host."""
    with pytest.raises(ValueError, match=r"generativelanguage\.googleapis\.com"):
        checker._validate_api_base("https://example.test/v1beta")


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(400, "INVALID"), (403, "DENIED"), (429, "QUOTA_LIMITED"), (500, "API_ERROR")],
)
def test_check_key_classifies_api_errors(status_code: int, expected_status: str) -> None:
    """Map structured HTTP failures to stable checker statuses."""
    response = checker.HttpResult(status_code=status_code, payload={"error": {"status": "TEST_STATUS"}})

    result = checker.check_key(
        "API_KEY_1",
        "top-secret",
        "https://example.test",
        1.0,
        requester=lambda *_args: response,
    )

    assert result.status == expected_status
    assert result.detail == f"models.list: HTTP {status_code} (TEST_STATUS)"
