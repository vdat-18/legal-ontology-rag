"""Graphiti integration: client construction and ingestion orchestration."""

from __future__ import annotations

from legal_rag.graph.client import build_graphiti
from legal_rag.graph.ingest import ingest

__all__ = ["build_graphiti", "ingest"]
