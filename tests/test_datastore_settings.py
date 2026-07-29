"""Tests for shared datastore environment settings."""

from __future__ import annotations

import pytest

from legal_rag.datastores import neo4j as neo4j_module
from legal_rag.settings import MilvusSettings, Neo4jSettings


def test_milvus_settings_load_shared_connection_from_environment() -> None:
    """Milvus connection and database values use generic environment names."""
    environment = {
        "MILVUS_URI": "http://localhost:19530",
        "MILVUS_DATABASE": "legal_rag",
        "MILVUS_TOKEN": "root:example",
    }

    settings = MilvusSettings.from_environment(environment)

    assert settings.uri == environment["MILVUS_URI"]
    assert settings.database == "legal_rag"
    assert settings.token == environment["MILVUS_TOKEN"]
    assert environment["MILVUS_TOKEN"] not in repr(settings)


def test_milvus_token_is_optional_for_milvus_lite() -> None:
    """Local Milvus Lite connections do not require an authentication token."""
    settings = MilvusSettings.from_environment({"MILVUS_URI": "./test.db", "MILVUS_DATABASE": "legal_rag"})

    assert settings.token is None


def test_neo4j_settings_load_shared_connection_from_environment() -> None:
    """Neo4j connection values use generic environment names and hide the password."""
    environment = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "example-password",
    }

    settings = Neo4jSettings.from_environment(environment)

    assert settings.uri == environment["NEO4J_URI"]
    assert settings.user == environment["NEO4J_USER"]
    assert settings.password == environment["NEO4J_PASSWORD"]
    assert environment["NEO4J_PASSWORD"] not in repr(settings)


def test_build_neo4j_driver_uses_shared_connection_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared Neo4j factory forwards the configured URI and credentials."""
    captured: dict[str, object] = {}

    def fake_driver(uri: str, *, auth: tuple[str, str]) -> object:
        captured.update(uri=uri, auth=auth)
        return object()

    monkeypatch.setattr(neo4j_module.AsyncGraphDatabase, "driver", fake_driver)
    settings = Neo4jSettings("bolt://neo4j:7687", "neo4j", "example-password")

    driver = neo4j_module.build_neo4j_driver(settings)

    assert driver is not None
    assert captured == {"uri": settings.uri, "auth": (settings.user, settings.password)}


@pytest.mark.parametrize(
    ("settings_type", "arguments"),
    [
        (MilvusSettings, {"uri": "", "database": "legal_rag"}),
        (MilvusSettings, {"uri": "http://localhost:19530", "database": ""}),
        (Neo4jSettings, {"uri": "", "user": "neo4j", "password": "example"}),
        (Neo4jSettings, {"uri": "bolt://localhost:7687", "user": "", "password": "example"}),
        (Neo4jSettings, {"uri": "bolt://localhost:7687", "user": "neo4j", "password": ""}),
    ],
)
def test_datastore_settings_reject_empty_required_values(
    settings_type: type[MilvusSettings] | type[Neo4jSettings],
    arguments: dict[str, str],
) -> None:
    """Datastore identifiers and credentials cannot be empty."""
    with pytest.raises(ValueError, match="must not be empty"):
        settings_type(**arguments)
