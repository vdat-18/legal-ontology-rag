"""Wrap the project's LLM and embedding models as a Ragas evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from legal_rag.settings import Settings


@dataclass(frozen=True, slots=True)
class Evaluator:
    """The judge LLM and embedding model Ragas uses to score answers."""

    llm: Any
    embeddings: Any


def build_evaluator(settings: Settings, embedding_model: str) -> Evaluator:
    """Wrap the configured OpenAI-compatible chat model and a local embedding model for Ragas."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Evaluator(llm=LangchainLLMWrapper(chat), embeddings=LangchainEmbeddingsWrapper(embeddings))
