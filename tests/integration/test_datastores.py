"""Smoke tests for explicitly provisioned Neo4j and Milvus services."""

from __future__ import annotations

import asyncio
import os

import pytest
from neo4j import AsyncGraphDatabase
from pymilvus import MilvusClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INFRA_INTEGRATION") != "1",
    reason="set RUN_INFRA_INTEGRATION=1 to test running datastores",
)


def test_milvus_accepts_authenticated_client_connections() -> None:
    """Milvus authentication permits the configured client to list databases."""
    client = MilvusClient(
        uri=os.environ["MILVUS_URI"],
        token=os.environ["MILVUS_TOKEN"],
    )
    try:
        assert client.list_databases()
    finally:
        client.close()


def test_neo4j_accepts_authenticated_driver_connections() -> None:
    """Neo4j accepts the configured Bolt credentials."""

    async def verify_connectivity() -> None:
        driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )
        try:
            await driver.verify_connectivity()
        finally:
            await driver.close()

    asyncio.run(verify_connectivity())
