"""Connect to the shared Milvus database."""

from __future__ import annotations

from pymilvus import MilvusClient

from legal_rag.settings import MilvusSettings


def build_milvus_client(settings: MilvusSettings) -> MilvusClient:
    """Connect to Milvus and select the configured shared database, creating it if absent."""
    connection_options = {"uri": settings.uri}
    if settings.token is not None:
        connection_options["token"] = settings.token
    client = MilvusClient(**connection_options)
    if settings.database not in client.list_databases():
        client.create_database(settings.database)
    client.use_database(settings.database)
    return client
