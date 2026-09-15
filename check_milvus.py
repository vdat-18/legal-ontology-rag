"""Inspect the configured Naive RAG Milvus collection schema."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.datastores.milvus import build_milvus_client
from legal_rag.settings import MilvusSettings, NaiveRagSettings


def main() -> None:
    """Log collection field names and types without reading document rows."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    client = build_milvus_client(MilvusSettings.from_environment())
    collection_name = NaiveRagSettings.from_environment().milvus_collection_name
    try:
        description = client.describe_collection(collection_name)
        logger.info("Collection: {}", collection_name)
        for field in description["fields"]:
            logger.info("field={} type={}", field["name"], field["type"])
    finally:
        client.close()


if __name__ == "__main__":
    main()
