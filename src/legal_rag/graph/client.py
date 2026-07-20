"""Construct a Graphiti client wired for the Vietnamese legal domain.

Centralises client construction (Claude for extraction, a configurable
embedder, and a BGE reranker) so ingestion and retrieval share one factory.
"""

from __future__ import annotations

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

from legal_rag.config import Settings
from legal_rag.errors import MissingCredentialError


def build_embedder(settings: Settings) -> EmbedderClient:
    """Return the embedder client selected by configuration."""
    if settings.embedder == "bge":
        from legal_rag.embeddings.vietnamese import BGEM3Embedder

        return BGEM3Embedder()
    return OpenAIEmbedder(config=OpenAIEmbedderConfig())


def build_graphiti(settings: Settings) -> Graphiti:
    """Construct a Graphiti client using Claude for extraction."""
    if not settings.anthropic_api_key:
        raise MissingCredentialError("ANTHROPIC_API_KEY")
    llm_client = AnthropicClient(
        config=LLMConfig(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            small_model=settings.small_llm_model,
        )
    )
    return Graphiti(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        llm_client=llm_client,
        embedder=build_embedder(settings),
        cross_encoder=BGERerankerClient(),
    )
