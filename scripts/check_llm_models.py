"""List models supported by the configured OpenAI-compatible LLM router."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from loguru import logger


@dataclass(frozen=True)
class ModelsResult:
    """Represent the outcome of a models.list call against the router."""

    status_code: int | None
    model_ids: list[str]
    detail: str | None = None
    transport_error: str | None = None


def fetch_models(base_url: str, api_key: str, timeout: float) -> ModelsResult:
    """Fetch the list of model IDs advertised by an OpenAI-compatible /models endpoint."""
    url = f"{base_url.rstrip('/')}/models"
    # base_url comes from the user's own .env / --base-url flag, not untrusted input.
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as error:
        detail = _read_error_detail(error)
        return ModelsResult(status_code=error.code, model_ids=[], detail=detail)
    except (TimeoutError, URLError) as error:
        return ModelsResult(status_code=None, model_ids=[], transport_error=type(error).__name__)

    return ModelsResult(status_code=200, model_ids=_extract_model_ids(payload))


def _extract_model_ids(payload: Any) -> list[str]:
    """Pull model IDs out of an OpenAI-style `{"data": [{"id": ...}, ...]}` payload."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return sorted(entry["id"] for entry in data if isinstance(entry, dict) and isinstance(entry.get("id"), str))


def _read_error_detail(error: HTTPError) -> str:
    """Read a short, structured error message from a failed response."""
    try:
        payload = json.load(error)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return f"HTTP {error.code}"
    message = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(message, dict):
        message = message.get("message")
    return f"HTTP {error.code}: {message}" if isinstance(message, str) else f"HTTP {error.code}"


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", help="Override LLM_BASE_URL from the env file")
    parser.add_argument("--api-key", help="Override LLM_API_KEY from the env file")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout in seconds")
    return parser


def main() -> int:
    """Query the configured router for its supported models and print them."""
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")

    load_dotenv(arguments.env_file, override=True)
    base_url = arguments.base_url or os.getenv("LLM_BASE_URL")
    api_key = arguments.api_key or os.getenv("LLM_API_KEY")
    configured_model = os.getenv("LLM_MODEL")
    if not base_url or not api_key:
        parser.error("LLM_BASE_URL and LLM_API_KEY must be set (in the env file or via flags)")

    result = fetch_models(base_url, api_key, arguments.timeout)

    if result.transport_error is not None:
        logger.error("Could not reach {}: {}", base_url, result.transport_error)
        return 1
    if result.status_code != 200:
        logger.error("Request failed: {}", result.detail)
        return 1
    if not result.model_ids:
        logger.warning("Router returned 200 but no model IDs were found in the response")
        return 1

    logger.info("{} supports {} model(s):", base_url, len(result.model_ids))
    for model_id in result.model_ids:
        marker = " (LLM_MODEL)" if model_id == configured_model else ""
        logger.info("  - {}{}", model_id, marker)

    if configured_model and configured_model not in result.model_ids:
        logger.warning("Configured LLM_MODEL={!r} is not in the advertised list", configured_model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
