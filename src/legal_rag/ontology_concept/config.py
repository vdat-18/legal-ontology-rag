"""Typed configuration for ontology concept resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

CONCEPT_RESOLUTION_CONTRACT_SCHEMA_VERSION = "0.3.0"


@dataclass(frozen=True, slots=True)
class ConceptResolutionSettings:
    """Candidate and proposal execution settings."""

    included_kinds: tuple[str, ...]
    max_occurrences_per_inventory: int
    batch_size: int
    max_output_tokens: int
    temperature: float
    max_retries: int
    retry_delay_seconds: float
    max_response_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> ConceptResolutionSettings:
        """Load the versioned concept-resolution contract."""
        document = load_yaml(path)
        if document["schema_version"] != CONCEPT_RESOLUTION_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported concept-resolution contract: {document['schema_version']}"
            raise ValueError(msg)
        candidates = document["candidates"]
        llm = document["llm"]
        return cls(
            included_kinds=tuple(candidates["included_kinds"]),
            max_occurrences_per_inventory=candidates["max_occurrences_per_inventory"],
            batch_size=llm["batch_size"],
            max_output_tokens=llm["max_output_tokens"],
            temperature=llm["temperature"],
            max_retries=llm["max_retries"],
            retry_delay_seconds=llm["retry_delay_seconds"],
            max_response_bytes=llm["max_response_bytes"],
        )
