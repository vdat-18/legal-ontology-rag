"""Tests for Graphiti knowledge-graph environment settings."""

from __future__ import annotations

from legal_rag.settings import GraphitiSettings


def test_graphiti_settings_load_from_environment() -> None:
    """Graphiti settings parse from process environment variables."""
    settings = GraphitiSettings.from_environment(
        {
            "EMBEDDING_PROFILE": "qwen3_0_6b",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        }
    )

    assert settings.embedding_profile == "qwen3_0_6b"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "password"  # noqa: S105
