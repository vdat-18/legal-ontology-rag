"""Typed configuration for ontology population."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

POPULATION_CONTRACT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class PopulationSettings:
    """Execution limits and identity policy for ontology population."""

    ontology_version: str
    instance_namespace: str
    batch_size: int
    max_preceding_chunks: int
    max_following_chunks: int
    max_article_context_chunks: int
    max_output_tokens: int
    temperature: float
    max_retries: int
    retry_delay_seconds: float
    max_response_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> PopulationSettings:
        """Load the versioned ontology population contract."""
        document = load_yaml(path)
        if document["schema_version"] != POPULATION_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported ontology population contract: {document['schema_version']}"
            raise ValueError(msg)
        extraction = document["extraction"]
        reading_window = document["input"]["reading_window"]
        return cls(
            ontology_version=document["ontology_version"],
            instance_namespace=document["instance_namespace"].rstrip("/") + "/",
            batch_size=extraction["batch_size"],
            max_preceding_chunks=reading_window["max_preceding_chunks"],
            max_following_chunks=reading_window["max_following_chunks"],
            max_article_context_chunks=reading_window["max_article_context_chunks"],
            max_output_tokens=extraction["max_output_tokens"],
            temperature=extraction["temperature"],
            max_retries=extraction["max_retries"],
            retry_delay_seconds=extraction["retry_delay_seconds"],
            max_response_bytes=extraction["max_response_bytes"],
        )
