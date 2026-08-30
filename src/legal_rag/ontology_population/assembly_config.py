"""Typed configuration for cross-chunk ontology rule assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

ASSEMBLY_CONTRACT_SCHEMA_VERSION = "0.2.0"


@dataclass(frozen=True, slots=True)
class AssemblySettings:
    """Candidate and proposal settings for rule assembly."""

    max_chunk_span: int
    max_records_per_window: int
    batch_size: int
    max_output_tokens: int
    temperature: float
    max_retries: int
    retry_delay_seconds: float
    max_response_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> AssemblySettings:
        """Load the versioned rule-assembly contract."""
        document = load_yaml(path)
        if document["schema_version"] != ASSEMBLY_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported ontology assembly contract: {document['schema_version']}"
            raise ValueError(msg)
        candidates = document["candidates"]
        llm = document["llm"]
        return cls(
            max_chunk_span=candidates["max_chunk_span"],
            max_records_per_window=candidates["max_records_per_window"],
            batch_size=llm["batch_size"],
            max_output_tokens=llm["max_output_tokens"],
            temperature=llm["temperature"],
            max_retries=llm["max_retries"],
            retry_delay_seconds=llm["retry_delay_seconds"],
            max_response_bytes=llm["max_response_bytes"],
        )
