"""Typed configuration for cross-chunk ontology rule assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml

ASSEMBLY_CONTRACT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class AssemblySettings:
    """Candidate, proposal, and review settings for rule assembly."""

    candidate_version: str
    max_chunk_span: int
    max_records_per_window: int
    prompt_version: str
    batch_size: int
    max_output_tokens: int
    temperature: float
    max_retries: int
    retry_delay_seconds: float
    max_response_bytes: int
    require_every_proposal_reviewed: bool

    @classmethod
    def from_path(cls, path: Path) -> AssemblySettings:
        """Load the versioned rule-assembly contract."""
        document = load_yaml(path)
        if document["schema_version"] != ASSEMBLY_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported ontology assembly contract: {document['schema_version']}"
            raise ValueError(msg)
        candidates = document["candidate_selection"]
        proposal = document["proposal"]
        review = document["review"]
        return cls(
            candidate_version=candidates["version"],
            max_chunk_span=candidates["max_chunk_span"],
            max_records_per_window=candidates["max_records_per_window"],
            prompt_version=proposal["prompt_version"],
            batch_size=proposal["batch_size"],
            max_output_tokens=proposal["max_output_tokens"],
            temperature=proposal["temperature"],
            max_retries=proposal["max_retries"],
            retry_delay_seconds=proposal["retry_delay_seconds"],
            max_response_bytes=proposal["max_response_bytes"],
            require_every_proposal_reviewed=review["require_every_proposal_reviewed_before_finalization"],
        )
