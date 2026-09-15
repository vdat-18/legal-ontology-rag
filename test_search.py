"""Run a local Naive RAG retrieval smoke check."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.datastores.milvus import build_milvus_client
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.naive_rag.retriever import Retriever
from legal_rag.settings import MilvusSettings, NaiveRagSettings


def main() -> None:
    """Retrieve a small result set without printing document contents."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = NaiveRagSettings.from_environment()
    profile = load_configured_embedding_profile(settings.embedding_profile)
    retriever = Retriever(
        EmbeddingEncoder.from_profile(profile),
        build_milvus_client(MilvusSettings.from_environment()),
        settings,
    )
    try:
        hits = retriever.retrieve("Xe đạp không chấp hành hiệu lệnh bị phạt gì?", top_k=5)
        logger.info("Retrieved {} result(s)", len(hits))
        for hit in hits:
            logger.info("chunk_id={} score={:.4f}", hit.chunk_id, hit.score)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
