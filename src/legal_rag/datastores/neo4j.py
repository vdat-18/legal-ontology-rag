"""Connect to the shared Neo4j database."""

from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase

from legal_rag.settings import Neo4jSettings


def build_neo4j_driver(settings: Neo4jSettings) -> AsyncDriver:
    """Open an async Neo4j driver for direct graph queries."""
    return AsyncGraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
