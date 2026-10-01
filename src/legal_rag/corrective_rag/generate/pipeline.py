"""Pipeline orchestrator for the Corrective RAG (CRAG) system.

Implements query execution, document grading, query reformulation, and final answer generation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

from legal_rag.corrective_rag.generate.actions import ActionTrigger
from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.generator import ResponseGenerator
from legal_rag.corrective_rag.generate.grader import CorrectiveGrader
from legal_rag.corrective_rag.generate.models import GradedChunk, KnowledgeAction, KnowledgeItem
from legal_rag.corrective_rag.generate.refiner import KnowledgeRefiner
from legal_rag.corrective_rag.generate.web_search import WebSearchProvider
from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel
from legal_rag.datastores.milvus import MilvusCorpusFingerprint
from legal_rag.query_rewriting import QueryRewriter, SyncQueryRewriter
from legal_rag.settings import Settings

_CORRECT_ACTIONS: set[KnowledgeAction] = {"CORRECT"}
_AMBIGUOUS_ACTIONS: set[KnowledgeAction] = {"AMBIGUOUS"}
_USABLE_KNOWLEDGE_ACTIONS: set[KnowledgeAction] = {"CORRECT", "AMBIGUOUS"}


class CorrectiveRAGPipeline:
    """Orchestrates the retrieval, grading, query reformulation, and generation pipeline."""

    def __init__(self, config: RAGConfig, *, query_rewriter: QueryRewriter | None = None) -> None:
        """Initialize the pipeline with all necessary adapters.

        Args:
            config: RAG config loaded from settings.
            query_rewriter: Shared query rewriter used to resolve follow-up questions.
        """
        self.config = config
        close_callbacks: list[Callable[[], None]] = []
        try:
            self.embedder = EmbeddingModel(config.embedding_profile)
            close_callbacks.append(self.embedder.close)
            self.db = LegalVectorDB(config, self.embedder)
            close_callbacks.append(self.db.close)
            self.grader = CorrectiveGrader(config)
            close_callbacks.append(self.grader.close)
            self.generator = ResponseGenerator(config)
            close_callbacks.append(self.generator.close)
            self.action_trigger = ActionTrigger(
                lower_threshold=config.evaluator_lower_threshold,
                upper_threshold=config.evaluator_upper_threshold,
            )
            self.refiner = KnowledgeRefiner(config, self.grader)
            self.web_search = WebSearchProvider(config)
            close_callbacks.append(self.web_search.close)
            llm_settings = Settings(
                llm_base_url=config.llm_base_url,
                llm_api_key=config.llm_api_key,
                llm_model=config.llm_model,
                llm_timeout_seconds=config.llm_timeout_seconds,
                llm_extra_request_body=config.llm_extra_request_body,
            )
            self._query_rewriter = query_rewriter or SyncQueryRewriter(
                llm_settings,
                max_output_tokens=config.query_rewrite_max_output_tokens,
                temperature=config.query_rewrite_temperature,
                max_response_bytes=config.max_response_bytes,
            )
            self._close_callbacks = tuple(close_callbacks)
        except BaseException as error:
            for close in reversed(close_callbacks):
                try:
                    close()
                except BaseException as cleanup_error:
                    error.add_note(f"Corrective RAG constructor cleanup failed: {cleanup_error!r}")
            raise

    def run(
        self,
        query: str,
        limit: int = 5,
        history: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        """Execute the Corrective RAG pipeline for the given query.

        Args:
            query: The user query string.
            limit: Maximum retrieved documents to consider.
            history: Sequence of (question, answer) pairs from earlier turns.

        Returns:
            A dictionary containing pipeline metadata and the final answer.
        """
        logger.info("Running Corrective RAG query.")
        standalone_query = query
        if history:
            rewritten = self._query_rewriter.rewrite(query, tuple(history)).rewritten.strip()
            if not rewritten:
                msg = "Query rewriting returned an empty standalone question."
                raise ValueError(msg)
            standalone_query = rewritten

        # Step 1: Initial retrieval
        retrieved_chunks = self.db.search(standalone_query, limit=limit)

        # Step 2: Retrieval evaluator scores each query-document pair
        graded_chunks: list[GradedChunk] = []

        for chunk in retrieved_chunks:
            evaluation = self.grader.score_document(standalone_query, chunk["text"])
            action = self.action_trigger.classify_score(evaluation.score)
            graded_chunks.append(
                GradedChunk(
                    chunk=chunk,
                    score=evaluation.score,
                    action=action,
                    rationale=evaluation.rationale,
                )
            )

        # Step 3: CRAG action trigger
        retrieval_action = self.action_trigger.retrieval_action(graded_chunks)
        reformulated_query = None
        external_documents = []
        internal_knowledge: list[KnowledgeItem] = []
        external_knowledge: list[KnowledgeItem] = []

        if retrieval_action == "CORRECT":
            logger.info("CRAG action: CORRECT. Refining internal retrieved knowledge.")
            internal_knowledge = self.refiner.refine_internal(
                standalone_query,
                graded_chunks,
                allowed_actions=_CORRECT_ACTIONS,
                allowed_strip_actions=_USABLE_KNOWLEDGE_ACTIONS,
            )
        elif retrieval_action == "INCORRECT":
            logger.info("CRAG action: INCORRECT. Discarding retrieval and searching external knowledge.")
            reformulated_query = self.generator.reformulate_query(standalone_query)
            external_documents = self.web_search.search(reformulated_query)
            external_knowledge = self.refiner.refine_external(
                standalone_query,
                external_documents,
                allowed_strip_actions=_USABLE_KNOWLEDGE_ACTIONS,
            )
        else:
            logger.info("CRAG action: AMBIGUOUS. Combining internal refinement and external search.")
            internal_knowledge = self.refiner.refine_internal(
                standalone_query,
                graded_chunks,
                allowed_actions=_AMBIGUOUS_ACTIONS,
                allowed_strip_actions=_USABLE_KNOWLEDGE_ACTIONS,
            )
            reformulated_query = self.generator.reformulate_query(standalone_query)
            external_documents = self.web_search.search(reformulated_query)
            external_knowledge = self.refiner.refine_external(
                standalone_query,
                external_documents,
                allowed_strip_actions=_USABLE_KNOWLEDGE_ACTIONS,
            )

        # Step 4: Final generation from action-selected corrected knowledge
        corrected_knowledge = internal_knowledge + external_knowledge
        logger.info(f"Generating answer based on {len(corrected_knowledge)} corrected knowledge items.")
        answer = self.generator.generate_answer(
            standalone_query,
            corrected_knowledge,
            original_query=query if standalone_query != query else None,
        )

        graded_results = [_format_graded_chunk(chunk) for chunk in graded_chunks]
        pipeline_status = f"SUCCESS_{retrieval_action}"

        return {
            "query": query,
            "rewritten_query": standalone_query if standalone_query != query else None,
            "reformulated_query": reformulated_query,
            "status": pipeline_status,
            "crag_action": retrieval_action,
            "retrieved_count": len(retrieved_chunks),
            "second_retrieved_count": 0,
            "web_search_count": len(external_documents),
            "internal_knowledge_count": len(internal_knowledge),
            "external_knowledge_count": len(external_knowledge),
            "relevant_count": len(corrected_knowledge),
            "graded_chunks": graded_results,
            "generation_contexts": [asdict(item) for item in corrected_knowledge],
            "answer": answer,
        }

    def preflight_corpus(self, processed_dir: str | Path = "data/processed/groundtruth") -> MilvusCorpusFingerprint:
        """Verify that the active Milvus collection matches the configured local corpus."""
        return self.db.preflight_corpus(processed_dir)

    def close(self) -> None:
        """Close every created resource without hiding the first cleanup failure."""
        primary_error: BaseException | None = None
        for close in reversed(self._close_callbacks):
            try:
                close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    primary_error = cleanup_error
                else:
                    primary_error.add_note(f"Additional Corrective RAG cleanup failure: {cleanup_error!r}")
        if primary_error is not None:
            raise primary_error


def _format_graded_chunk(graded_chunk: GradedChunk) -> dict[str, Any]:
    """Format a graded chunk for UI and CLI output."""
    chunk = graded_chunk.chunk
    return {
        "chunk_id": chunk["chunk_id"],
        "source_name": chunk["source_name"],
        "text": chunk["text"],
        "score": chunk["score"],
        "confidence_score": graded_chunk.score,
        "action": graded_chunk.action,
        "rationale": graded_chunk.rationale,
        "is_relevant": graded_chunk.is_relevant,
    }
