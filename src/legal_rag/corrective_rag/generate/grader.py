"""LLM-based retrieval evaluator for Corrective RAG."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import DocumentRelevanceEvaluation as BamlDocumentRelevanceEvaluation
from legal_rag.baml_transport import BamlCaller
from legal_rag.corrective_rag.generate.config import RAGConfig


@dataclass(frozen=True, slots=True)
class RelevanceEvaluation:
    """Evaluator output for one query-document pair."""

    score: float
    rationale: str = ""


class CorrectiveGrader:
    """Evaluates retrieved documents with the configured OpenAI-compatible LLM."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the grader client."""
        self.config = config
        self._caller = BamlCaller(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            max_output_tokens=self.config.evaluator_max_output_tokens,
            temperature=self.config.evaluator_temperature,
            timeout_seconds=self.config.llm_timeout_seconds,
            max_response_bytes=self.config.max_response_bytes,
        )

    def score_document(self, query: str, document_text: str) -> RelevanceEvaluation:
        """Score how relevant a document is to the query."""
        logger.debug("Grading chunk relevance for one query-document pair.")
        request = b.request.ScoreDocumentRelevance(
            query,
            document_text,
            baml_options=self._caller.options,
        )
        raw_output = self._caller.execute(request)
        baml_evaluation = b.parse.ScoreDocumentRelevance(
            raw_output,
            baml_options=self._caller.options,
        )
        evaluation = _to_relevance_evaluation(baml_evaluation)
        logger.info(f"Chunk relevance score: {evaluation.score:.2f}")
        return evaluation

    def grade_document(self, query: str, document_text: str) -> bool:
        """Evaluate whether a document is strongly relevant to the query."""
        evaluation = self.score_document(query, document_text)
        return evaluation.score >= self.config.evaluator_upper_threshold

    def close(self) -> None:
        """Release grader resources."""


def _to_relevance_evaluation(evaluation: BamlDocumentRelevanceEvaluation) -> RelevanceEvaluation:
    """Validate BAML relevance output and map it to the pipeline model."""
    score = float(evaluation.score)
    if not 0.0 <= score <= 1.0:
        msg = f"LLM grading score must be between 0 and 1, got {score}"
        raise ValueError(msg)
    return RelevanceEvaluation(score=score, rationale=evaluation.rationale)
