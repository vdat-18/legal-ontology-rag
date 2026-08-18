"""Persistence for ontology semantic-QA reports."""

from __future__ import annotations

from pathlib import Path

from legal_rag.io_utils import write_json_atomic
from legal_rag.ontology_validation.semantic_models import SemanticQaReport


def write_semantic_qa_report(path: Path, report: SemanticQaReport) -> None:
    """Atomically replace one machine-readable semantic-QA report."""
    write_json_atomic(path, report.to_dict())
