"""Typed configuration for legal-exception resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

EXCEPTION_RESOLUTION_CONTRACT_SCHEMA_VERSION = "0.2.0"


@dataclass(frozen=True, slots=True)
class ExceptionResolutionSettings:
    """Candidate and proposal execution settings."""

    max_source_chunks_per_window: int
    max_general_rules_per_window: int
    max_target_rules_per_window: int
    max_total_windows: int
    batch_size: int
    max_output_tokens: int
    temperature: float
    max_retries: int
    retry_delay_seconds: float
    max_response_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> ExceptionResolutionSettings:
        """Load the versioned exception-resolution contract."""
        document = load_yaml(path)
        if document["schema_version"] != EXCEPTION_RESOLUTION_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported exception-resolution contract: {document['schema_version']}"
            raise ValueError(msg)
        candidates = document["candidates"]
        llm = document["llm"]
        return cls(
            max_source_chunks_per_window=candidates["max_source_chunks_per_window"],
            max_general_rules_per_window=candidates["max_general_rules_per_window"],
            max_target_rules_per_window=candidates["max_target_rules_per_window"],
            max_total_windows=candidates["max_total_windows"],
            batch_size=llm["batch_size"],
            max_output_tokens=llm["max_output_tokens"],
            temperature=llm["temperature"],
            max_retries=llm["max_retries"],
            retry_delay_seconds=llm["retry_delay_seconds"],
            max_response_bytes=llm["max_response_bytes"],
        )
