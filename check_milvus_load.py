"""Inspect and load the configured Naive RAG Milvus collection."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.datastores.milvus import build_milvus_client
from legal_rag.settings import MilvusSettings, NaiveRagSettings


def main() -> None:
    """Load the collection and report its state and index names."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    client = build_milvus_client(MilvusSettings.from_environment())
    collection_name = NaiveRagSettings.from_environment().milvus_collection_name
    try:
        client.load_collection(collection_name)
        logger.info("Collection state: {}", client.get_load_state(collection_name))
        logger.info("Collection indexes: {}", client.list_indexes(collection_name))
    finally:
        client.close()


if __name__ == "__main__":
    main()
