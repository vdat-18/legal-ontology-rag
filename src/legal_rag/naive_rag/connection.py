"""Connect to Milvus and scope the client to the configured database."""

from __future__ import annotations

from pymilvus import MilvusClient

from legal_rag.settings import NaiveRagSettings


def build_milvus_client(settings: NaiveRagSettings) -> MilvusClient:
    """Connect to Milvus and switch into the configured database, creating it if absent."""
    connection_options = {"uri": settings.milvus_uri}
    if settings.milvus_token is not None:
        connection_options["token"] = settings.milvus_token
    client = MilvusClient(**connection_options)
    if settings.milvus_database not in client.list_databases():
        client.create_database(settings.milvus_database)
    client.use_database(settings.milvus_database)
    return client
