"""Factory for a Graphiti client wired to the project's LLM and embedding configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from openai import AsyncOpenAI

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder
from legal_rag.settings import GraphitiSettings, Settings


def build_graphiti(
    settings: Settings,
    graphiti_settings: GraphitiSettings,
    profile: EmbeddingProfile,
    api_key: str | None = None,
    embedder: QwenGraphitiEmbedder | None = None,
) -> Graphiti:
    """Construct a Graphiti client backed by the project's OpenAI-compatible LLM and Qwen embedder.

    Args:
        settings: Application LLM connection settings.
        graphiti_settings: Neo4j connection and embedding profile selection.
        profile: The embedding profile to encode text with, when *embedder* isn't already built.
        api_key: Overrides ``settings.llm_api_key`` when given — lets multiple Graphiti
            clients each use a distinct LLM API key, for running several concurrent indexing
            workers without hitting one key's rate limit.
        embedder: A pre-built embedder to reuse across multiple clients. Building one loads the
            full sentence-transformers model into memory, so when constructing several clients
            for concurrent indexing, build it once and pass it in here rather than letting each
            client load its own copy.
    """
    llm_config = LLMConfig(
        api_key=api_key or settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    # OpenAIGenericClient/OpenAIRerankerClient build an AsyncOpenAI client with no timeout when
    # none is given; pass one bound to the project's configured LLM timeout instead.
    openai_client = AsyncOpenAI(
        api_key=llm_config.api_key, base_url=llm_config.base_url, timeout=settings.llm_timeout_seconds
    )
    llm_client = OpenAIGenericClient(config=llm_config, client=openai_client, structured_output_mode="json_object")
    resolved_embedder = embedder or QwenGraphitiEmbedder.from_profile(profile)
    return Graphiti(
        uri=graphiti_settings.neo4j_uri,
        user=graphiti_settings.neo4j_user,
        password=graphiti_settings.neo4j_password,
        llm_client=llm_client,
        embedder=resolved_embedder,
        cross_encoder=OpenAIRerankerClient(config=llm_config, client=openai_client),
    )


def load_llm_api_keys(environment: Mapping[str, str] | None = None) -> list[str]:
    """Collect every configured LLM API key: ``LLM_API_KEY`` plus any ``LLM_API_KEY_2..N``.

    Used to run several concurrent Graphiti indexing workers, each against a different
    account/key, avoiding a single key's rate limit while sharing one work queue.
    """
    values = os.environ if environment is None else environment
    keys = [values["LLM_API_KEY"].strip()]
    index = 2
    while f"LLM_API_KEY_{index}" in values:
        keys.append(values[f"LLM_API_KEY_{index}"].strip())
        index += 1
    return keys
