"""HTTP and command-line entrypoints for Corrective RAG."""

from __future__ import annotations

from legal_rag.corrective_rag.api.server import app, create_app, main

__all__ = ["app", "create_app", "main"]
