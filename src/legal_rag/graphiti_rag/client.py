"""Factory for a Graphiti client wired to the project's LLM and embedding configuration."""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from openai import AsyncOpenAI

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder
from legal_rag.graphiti_rag.reranker import LocalCrossEncoderReranker
from legal_rag.settings import GraphitiSettings, Neo4jSettings, Settings


def _build_openai_client(settings: Settings, llm_config: LLMConfig) -> AsyncOpenAI:
    """Build the shared provider client, applying the configured provider-specific request fields.

    Graphiti calls the provider directly for entity extraction and for cross-encoder reranking, so
    those requests bypass this project's BAML transport and would otherwise miss the same settings.
    That matters for DeepSeek: reasoning mode consumes the completion, which empties ``content`` for
    extraction and leaves the reranker's single-token response with no ``logprobs.content`` to score,
    so ranking fails outright. The fields come from configuration, never from a model-name guess.
    """
    client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url, timeout=settings.llm_timeout_seconds)
    configured = dict(settings.llm_extra_request_body)
    if not configured:
        return client

    create = client.chat.completions.create

    @functools.wraps(create)
    async def create_with_configured_body(*args: Any, **kwargs: Any) -> Any:
        kwargs["extra_body"] = {**configured, **(kwargs.get("extra_body") or {})}
        return await create(*args, **kwargs)

    client.chat.completions.create = create_with_configured_body
    return client


def build_graphiti(
    settings: Settings,
    neo4j_settings: Neo4jSettings,
    graphiti_settings: GraphitiSettings,
    profile: EmbeddingProfile,
    api_key: str | None = None,
    embedder: QwenGraphitiEmbedder | None = None,
) -> Graphiti:
    """Construct a Graphiti client backed by the project's OpenAI-compatible LLM and Qwen embedder.

    Args:
        settings: Application LLM connection settings.
        neo4j_settings: Shared Neo4j connection settings.
        graphiti_settings: Graphiti embedding profile selection and algorithm settings.
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
    openai_client = _build_openai_client(settings, llm_config)
    llm_client = OpenAIGenericClient(config=llm_config, client=openai_client, structured_output_mode="json_object")
    resolved_embedder = embedder or QwenGraphitiEmbedder.from_profile(profile)
    return Graphiti(
        uri=neo4j_settings.uri,
        user=neo4j_settings.user,
        password=neo4j_settings.password,
        llm_client=llm_client,
        embedder=resolved_embedder,
        cross_encoder=LocalCrossEncoderReranker(
            graphiti_settings.reranker_model_id, graphiti_settings.reranker_model_revision
        ),
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
