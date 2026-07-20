"""Ingestion: turn source documents into time-stamped Graphiti episodes."""

from __future__ import annotations

from legal_rag.ingestion.documents import LegalEpisode, iter_episodes, iter_episodes_from_config

__all__ = ["LegalEpisode", "iter_episodes", "iter_episodes_from_config"]
