"""Check Gemini API keys from a dotenv file without exposing their values."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from loguru import logger

DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GENERATE_METHOD = "generateContent"
INVALID_API_BASE_MESSAGE = (
    "API base must use HTTPS on generativelanguage.googleapis.com without credentials or parameters"
)
INVALID_MODEL_MESSAGE = "Model must be a single model ID, optionally prefixed with 'models/'"


@dataclass(frozen=True)
class HttpResult:
    """Represent one Gemini HTTP response."""

    status_code: int | None
    payload: Mapping[str, Any]
    transport_error: str | None = None


@dataclass(frozen=True)
class KeyResult:
    """Represent the test outcome for one named API key."""

    name: str
    status: str
    detail: str
    model: str | None = None


RequestJson = Callable[[str, str, str, Mapping[str, Any] | None, float], HttpResult]


def load_keys(env_file: Path, prefix: str) -> dict[str, str]:
    """Load non-empty API keys whose variable names start with prefix."""
    if not env_file.is_file():
        raise FileNotFoundError(env_file)
    values = dotenv_values(env_file)
    return {
        name: value.strip()
        for name, value in values.items()
        if name.startswith(prefix) and isinstance(value, str) and value.strip()
    }


def request_json(
    method: str,
    url: str,
    api_key: str,
    body: Mapping[str, Any] | None,
    timeout: float,
) -> HttpResult:
    """Send one authenticated JSON request to the validated Gemini API base."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    # The CLI validates that requests use the configured HTTPS Gemini base.
    request = Request(  # noqa: S310
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
        return HttpResult(status_code=response.status, payload=_mapping_or_empty(payload))
    except HTTPError as error:
        return HttpResult(status_code=error.code, payload=_read_error_payload(error))
    except (TimeoutError, URLError) as error:
        return HttpResult(status_code=None, payload={}, transport_error=type(error).__name__)


def check_key(
    name: str,
    api_key: str,
    api_base: str,
    timeout: float,
    configured_model: str | None = None,
    requester: RequestJson = request_json,
) -> KeyResult:
    """Check model access and one minimal generation request for an API key."""
    models_result = requester("GET", f"{api_base}/models", api_key, None, timeout)
    if models_result.status_code != 200:
        return _failed_result(name, models_result, "models.list")

    model = _normalise_model(configured_model) if configured_model else _select_generation_model(models_result.payload)
    if model is None:
        return KeyResult(name=name, status="NO_MODEL", detail="No advertised model supports generateContent")

    model_id = model.removeprefix("models/")
    encoded_model_id = quote(model_id, safe="")
    generation_result = requester(
        "POST",
        f"{api_base}/models/{encoded_model_id}:generateContent",
        api_key,
        {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]},
        timeout,
    )
    if generation_result.status_code == 200:
        return KeyResult(name=name, status="WORKING", detail="Generation succeeded", model=model)
    return _failed_result(name, generation_result, "generateContent", model)


def _select_generation_model(payload: Mapping[str, Any]) -> str | None:
    """Select the first advertised model that supports content generation."""
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        methods = model.get("supportedGenerationMethods")
        if isinstance(name, str) and isinstance(methods, list) and GENERATE_METHOD in methods:
            return _normalise_model(name)
    return None


def _normalise_model(model: str | None) -> str | None:
    """Validate and normalize a Gemini model resource name."""
    if model is None:
        return None
    model_id = model.removeprefix("models/").strip()
    if not model_id or "/" in model_id or "\\" in model_id:
        raise ValueError(INVALID_MODEL_MESSAGE)
    return f"models/{model_id}"


def _failed_result(name: str, result: HttpResult, operation: str, model: str | None = None) -> KeyResult:
    """Convert a failed HTTP result into a stable, secret-free status."""
    if result.transport_error is not None:
        return KeyResult(name=name, status="NETWORK_ERROR", detail=result.transport_error, model=model)

    error = result.payload.get("error")
    api_status = error.get("status") if isinstance(error, dict) else None
    status_by_code = {
        400: "INVALID",
        401: "INVALID",
        403: "DENIED",
        404: "NOT_FOUND",
        429: "QUOTA_LIMITED",
    }
    status = status_by_code.get(result.status_code, "API_ERROR")
    detail = f"{operation}: HTTP {result.status_code}"
    if isinstance(api_status, str):
        detail = f"{detail} ({api_status})"
    return KeyResult(name=name, status=status, detail=detail, model=model)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Return a JSON mapping or an empty mapping for an unexpected response."""
    return value if isinstance(value, dict) else {}


def _read_error_payload(error: HTTPError) -> Mapping[str, Any]:
    """Read a structured API error without retaining raw response text."""
    try:
        return _mapping_or_empty(json.load(error))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _validate_api_base(api_base: str) -> str:
    """Require an HTTPS API base without query parameters or fragments."""
    normalized = api_base.rstrip("/")
    parsed = urlparse(normalized)
    invalid_base = (
        parsed.scheme != "https"
        or parsed.hostname != "generativelanguage.googleapis.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    )
    if invalid_base:
        raise ValueError(INVALID_API_BASE_MESSAGE)
    return normalized


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", type=Path, default=Path(".env"))
    parser.add_argument("--prefix", default="API_KEY_", help="Only test variables with this prefix")
    parser.add_argument(
        "--model", help="Specific model ID to test; otherwise use the first advertised generation model"
    )
    parser.add_argument("--api-base", default=os.getenv("GEMINI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout per HTTP request in seconds")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of keys tested concurrently")
    return parser


def main() -> int:
    """Run the key checks and return a process exit code."""
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    if arguments.workers <= 0:
        parser.error("--workers must be positive")

    try:
        api_base = _validate_api_base(arguments.api_base)
        model = _normalise_model(arguments.model)
        keys = load_keys(arguments.env_file, arguments.prefix)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if not keys:
        parser.error(f"No non-empty variables starting with {arguments.prefix!r} were found")

    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        results = list(
            executor.map(
                lambda item: check_key(item[0], item[1], api_base, arguments.timeout, model),
                sorted(keys.items()),
            )
        )

    for result in results:
        model_suffix = f" | {result.model}" if result.model else ""
        logger.info("{:<18} | {:<13} | {}{}", result.name, result.status, result.detail, model_suffix)

    working_count = sum(result.status == "WORKING" for result in results)
    logger.info("Working keys: {}/{}", working_count, len(results))
    return 0 if working_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
