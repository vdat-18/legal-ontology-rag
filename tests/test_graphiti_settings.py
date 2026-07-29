"""Tests for Graphiti knowledge-graph environment settings."""

from __future__ import annotations

from dataclasses import MISSING

from legal_rag.settings import _GRAPHITI_TUNABLES, GraphitiSettings


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


def test_every_declared_tunable_is_overridable_from_the_environment() -> None:
    """No declared tunable can be silently ignored: each maps to a GRAPHITI_<FIELD_NAME> variable."""
    base = {
        "EMBEDDING_PROFILE": "qwen3_0_6b",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
    }
    defaults = GraphitiSettings.from_environment(base)

    for name, converter in _GRAPHITI_TUNABLES.items():
        default = getattr(defaults, name)
        overridden = converter(str(default + 1))
        settings = GraphitiSettings.from_environment({**base, f"GRAPHITI_{name.upper()}": str(overridden)})

        assert getattr(settings, name) == overridden, name


def test_tunables_cover_every_optional_graphiti_setting() -> None:
    """A newly added tunable must be registered, so it cannot be left unreadable from the env."""
    configurable = {name for name, spec in GraphitiSettings.__dataclass_fields__.items() if spec.default is not MISSING}

    assert set(_GRAPHITI_TUNABLES) == configurable
