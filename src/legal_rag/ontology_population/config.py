"""Typed configuration for ontology population."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

POPULATION_CONTRACT_SCHEMA_VERSION = "0.3.0"


@dataclass(frozen=True, slots=True)
class PopulationSettings:
    """Execution limits and identity policy for ontology population."""

    ontology_version: str
    instance_namespace: str
    conduct_batch_size: int
    batch_size: int
    max_locator_context_chunks: int
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
        llm = document["llm"]
        reading_window = document["reading_window"]
        return cls(
            ontology_version=document["ontology_version"],
            instance_namespace=document["instance_namespace"].rstrip("/") + "/",
            conduct_batch_size=llm["conduct_batch_size"],
            batch_size=llm["batch_size"],
            max_locator_context_chunks=reading_window["max_locator_context_chunks"],
            max_preceding_chunks=reading_window["max_preceding_chunks"],
            max_following_chunks=reading_window["max_following_chunks"],
            max_article_context_chunks=reading_window["max_article_context_chunks"],
            max_output_tokens=llm["max_output_tokens"],
            temperature=llm["temperature"],
            max_retries=llm["max_retries"],
            retry_delay_seconds=llm["retry_delay_seconds"],
            max_response_bytes=llm["max_response_bytes"],
        )
