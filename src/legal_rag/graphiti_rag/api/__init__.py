"""HTTP and command-line entrypoints for the Graphiti knowledge-graph RAG pipeline."""

from __future__ import annotations

from legal_rag.graphiti_rag.api.server import app, create_app, main

__all__ = ["app", "create_app", "main"]
