"""Factory for a Graphiti client wired to the project's LLM and embedding configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import openai
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from graphiti_core.llm_client.openai_generic_client import DEFAULT_MODEL, OpenAIGenericClient
from graphiti_core.prompts.models import Message
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder
from legal_rag.settings import GraphitiSettings, Settings


class DeepSeekNonThinkingClient(OpenAIGenericClient):
    """OpenAIGenericClient variant that disables DeepSeek's "thinking" mode on every call.

    DeepSeek's thinking (reasoning) mode spends completion tokens on ``reasoning_content``
    before the final answer, which is unnecessary for Graphiti's short extraction/dedup
    calls and multiplies both latency and cost across a full-corpus indexing run. DeepSeek
    supports an OpenAI-incompatible ``thinking: {"type": "disabled"}`` request field (not
    exposed by ``OpenAIGenericClient``), passed here via the ``openai`` SDK's ``extra_body``.
    """

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Call the chat completion endpoint with DeepSeek's thinking mode disabled."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role == "user":
                openai_messages.append({"role": "user", "content": message.content})
            elif message.role == "system":
                openai_messages.append({"role": "system", "content": message.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),
                extra_body={"thinking": {"type": "disabled"}},
            )
            result = response.choices[0].message.content or ""
            if not result:
                raise EmptyResponseError("LLM returned an empty response")  # noqa: TRY301, TRY003
            return json.loads(self._strip_code_fences(result))
        except openai.RateLimitError as error:
            raise RateLimitError from error
        except Exception as error:
            logger.error("Error in generating LLM response: {}", error)
            raise


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
    llm_client = DeepSeekNonThinkingClient(config=llm_config, structured_output_mode="json_object")
    resolved_embedder = embedder or QwenGraphitiEmbedder.from_profile(profile)
    return Graphiti(
        uri=graphiti_settings.neo4j_uri,
        user=graphiti_settings.neo4j_user,
        password=graphiti_settings.neo4j_password,
        llm_client=llm_client,
        embedder=resolved_embedder,
        cross_encoder=OpenAIRerankerClient(config=llm_config),
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
