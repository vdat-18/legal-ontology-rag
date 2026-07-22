"""Typed configuration for the document parser pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

_BASE_URL_ENV = "LLM_BASE_URL"
_API_KEY_ENV = "LLM_API_KEY"
_MODEL_ENV = "LLM_MODEL"
_TIMEOUT_ENV = "LLM_TIMEOUT_SECONDS"
_CONCURRENCY_ENV = "IMAGE_CAPTION_MAX_CONCURRENCY"
_MAX_OUTPUT_TOKENS_ENV = "IMAGE_CAPTION_MAX_OUTPUT_TOKENS"
_MAX_CONTEXT_CHARACTERS_ENV = "IMAGE_CAPTION_MAX_CONTEXT_CHARACTERS"
_TEMPERATURE_ENV = "IMAGE_CAPTION_TEMPERATURE"


class ParserConfigurationError(ValueError):
    """Raised when parser configuration is missing or invalid."""

    @classmethod
    def missing(cls, variable_name: str) -> ParserConfigurationError:
        """Build an error for a required environment variable."""
        return cls(f"Required parser environment variable is missing: {variable_name}")

    @classmethod
    def invalid_url(cls, variable_name: str) -> ParserConfigurationError:
        """Build an error for an unusable service URL."""
        return cls(f"Parser environment variable must be an HTTP(S) URL: {variable_name}")

    @classmethod
    def invalid_number(cls, variable_name: str) -> ParserConfigurationError:
        """Build an error for a numeric environment variable."""
        return cls(f"Parser environment variable must be a valid number: {variable_name}")

    @classmethod
    def non_positive(cls, variable_name: str) -> ParserConfigurationError:
        """Build an error for a non-positive limit."""
        return cls(f"Parser environment variable must be positive: {variable_name}")

    @classmethod
    def invalid_temperature(cls, variable_name: str) -> ParserConfigurationError:
        """Build an error for a temperature outside the API-supported range."""
        return cls(f"Parser environment variable is outside the supported range: {variable_name}")


@dataclass(frozen=True, slots=True)
class ParserSettings:
    """Validated service and bounded-execution settings for parser enrichment."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float
    max_concurrency: int
    max_output_tokens: int
    max_context_characters: int
    temperature: float

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ParserSettings:
        """Load and validate required parser settings."""
        values = os.environ if environment is None else environment
        base_url = _required_value(values, _BASE_URL_ENV)
        parsed_url = urlsplit(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ParserConfigurationError.invalid_url(_BASE_URL_ENV)
        temperature = _number(values, _TEMPERATURE_ENV)
        if not 0 <= temperature <= 2:
            raise ParserConfigurationError.invalid_temperature(_TEMPERATURE_ENV)
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=_required_value(values, _API_KEY_ENV),
            model=_required_value(values, _MODEL_ENV),
            timeout_seconds=_positive_float(values, _TIMEOUT_ENV),
            max_concurrency=_positive_int(values, _CONCURRENCY_ENV),
            max_output_tokens=_positive_int(values, _MAX_OUTPUT_TOKENS_ENV),
            max_context_characters=_positive_int(values, _MAX_CONTEXT_CHARACTERS_ENV),
            temperature=temperature,
        )


def _required_value(environment: Mapping[str, str], variable_name: str) -> str:
    """Return one non-empty environment value."""
    value = environment.get(variable_name)
    if value is None or not value.strip():
        raise ParserConfigurationError.missing(variable_name)
    return value.strip()


def _number(environment: Mapping[str, str], variable_name: str) -> float:
    """Parse one required floating-point environment value."""
    value = _required_value(environment, variable_name)
    try:
        return float(value)
    except ValueError as error:
        raise ParserConfigurationError.invalid_number(variable_name) from error


def _positive_float(environment: Mapping[str, str], variable_name: str) -> float:
    """Parse one required positive floating-point environment value."""
    value = _number(environment, variable_name)
    if value <= 0:
        raise ParserConfigurationError.non_positive(variable_name)
    return value


def _positive_int(environment: Mapping[str, str], variable_name: str) -> int:
    """Parse one required positive integer environment value."""
    raw_value = _required_value(environment, variable_name)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ParserConfigurationError.invalid_number(variable_name) from error
    if value <= 0:
        raise ParserConfigurationError.non_positive(variable_name)
    return value
