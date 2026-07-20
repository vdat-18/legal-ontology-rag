"""Service API for generating ground-truth review candidates with Ragas."""

from __future__ import annotations

from datetime import date
from typing import cast

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

from legal_rag.config import Settings
from legal_rag.errors import InvalidRagasComponentError, MissingCredentialError, UnsupportedRagasProviderError
from legal_rag.evaluation.schemas import GroundTruthCandidate
from legal_rag.evaluation.synthesis import synthesize_single_hop_candidates, write_candidates_jsonl
from legal_rag.ingestion import iter_episodes_from_config

_OPENAI_PROVIDER = "openai"


def generate_groundtruth_candidates(
    settings: Settings,
    *,
    testset_size: int,
    as_of_date: date | None = None,
    generator: TestsetGenerator | None = None,
    query_synthesizer: SingleHopSpecificQuerySynthesizer | None = None,
) -> list[GroundTruthCandidate]:
    """Generate and write pending ground-truth candidates from the v1 source corpus."""
    if (generator is None) != (query_synthesizer is None):
        raise InvalidRagasComponentError
    if generator is None or query_synthesizer is None:
        generator, query_synthesizer = build_ragas_generation_components(settings)

    episodes = iter_episodes_from_config(
        settings,
        settings.groundtruth_config,
        archive_only=True,
        skip_missing=False,
    )
    candidates = synthesize_single_hop_candidates(
        generator,
        query_synthesizer,
        episodes,
        testset_size=testset_size,
        as_of_date=as_of_date,
    )
    write_candidates_jsonl(candidates, settings.ragas_candidates_output)
    logger.info("Wrote {} ground-truth candidate(s) to {}", len(candidates), settings.ragas_candidates_output)
    return candidates


def build_ragas_generation_components(
    settings: Settings,
) -> tuple[TestsetGenerator, SingleHopSpecificQuerySynthesizer]:
    """Build the Ragas generator and single-hop synthesizer from settings."""
    llm = _openai_llm(settings)
    embeddings = _openai_embeddings(settings)
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    query_synthesizer = SingleHopSpecificQuerySynthesizer(llm=llm)
    return generator, query_synthesizer


def _openai_llm(settings: Settings) -> LangchainLLMWrapper:
    _require_provider("RAGAS_LLM_PROVIDER", settings.ragas_llm_provider)
    model = _required_setting("RAGAS_LLM_MODEL", settings.ragas_llm_model)
    api_key = _required_setting("OPENAI_API_KEY", settings.openai_api_key)
    return cast(
        LangchainLLMWrapper,
        LangchainLLMWrapper(ChatOpenAI(model=model, api_key=api_key)),
    )


def _openai_embeddings(settings: Settings) -> LangchainEmbeddingsWrapper:
    _require_provider("RAGAS_EMBEDDING_PROVIDER", settings.ragas_embedding_provider)
    model = _required_setting("RAGAS_EMBEDDING_MODEL", settings.ragas_embedding_model)
    api_key = _required_setting("OPENAI_API_KEY", settings.openai_api_key)
    return cast(
        LangchainEmbeddingsWrapper,
        LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=model, api_key=api_key)),
    )


def _require_provider(setting_name: str, provider: str | None) -> None:
    value = _required_setting(setting_name, provider).lower()
    if value != _OPENAI_PROVIDER:
        raise UnsupportedRagasProviderError(setting_name, value, (_OPENAI_PROVIDER,))


def _required_setting(setting_name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise MissingCredentialError(setting_name)
    return value
