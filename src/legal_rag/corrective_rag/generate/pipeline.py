"""Pipeline orchestrator for the Corrective RAG (CRAG) system.

Implements query execution, document grading, query reformulation, and final answer generation.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from legal_rag.corrective_rag.generate.actions import ActionTrigger
from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.generator import ResponseGenerator
from legal_rag.corrective_rag.generate.grader import CorrectiveGrader
from legal_rag.corrective_rag.generate.models import GradedChunk, KnowledgeItem
from legal_rag.corrective_rag.generate.refiner import KnowledgeRefiner
from legal_rag.corrective_rag.generate.web_search import WebSearchProvider
from legal_rag.corrective_rag.retrieval.database import LegalVectorDB
from legal_rag.corrective_rag.retrieval.embeddings import EmbeddingModel


class CorrectiveRAGPipeline:
    """Orchestrates the retrieval, grading, query reformulation, and generation pipeline."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the pipeline with all necessary adapters.

        Args:
            config: RAG config loaded from settings.
        """
        self.config = config
        self.embedder = EmbeddingModel(config.embedding_profile)
        self.db = LegalVectorDB(config, self.embedder)
        self.grader = CorrectiveGrader(config)
        self.generator = ResponseGenerator(config)
        self.action_trigger = ActionTrigger(
            lower_threshold=config.evaluator_lower_threshold,
            upper_threshold=config.evaluator_upper_threshold,
        )
        self.refiner = KnowledgeRefiner(config, self.grader)
        self.web_search = WebSearchProvider(config)

    def run(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Execute the Corrective RAG pipeline for the given query.

        Args:
            query: The user query string.
            limit: Maximum retrieved documents to consider.

        Returns:
            A dictionary containing pipeline metadata and the final answer.
        """
        logger.info(f"Running Corrective RAG for query: '{query}'")

        # Step 1: Initial retrieval
        retrieved_chunks = self.db.search(query, limit=limit)

        # Step 2: Retrieval evaluator scores each query-document pair
        graded_chunks: list[GradedChunk] = []

        for chunk in retrieved_chunks:
            evaluation = self.grader.score_document(query, chunk["text"])
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
            internal_knowledge = self.refiner.refine_internal(query, graded_chunks)
        elif retrieval_action == "INCORRECT":
            logger.info("CRAG action: INCORRECT. Discarding retrieval and searching external knowledge.")
            reformulated_query = self.generator.reformulate_query(query)
            external_documents = self.web_search.search(reformulated_query)
            external_knowledge = self.refiner.refine_external(query, external_documents)
        else:
            logger.info("CRAG action: AMBIGUOUS. Combining internal refinement and external search.")
            internal_knowledge = self.refiner.refine_internal(query, graded_chunks)
            reformulated_query = self.generator.reformulate_query(query)
            external_documents = self.web_search.search(reformulated_query)
            external_knowledge = self.refiner.refine_external(query, external_documents)

        # Step 4: Final generation from action-selected corrected knowledge
        corrected_knowledge = internal_knowledge + external_knowledge
        logger.info(f"Generating answer based on {len(corrected_knowledge)} corrected knowledge items.")
        answer = self.generator.generate_answer(query, corrected_knowledge)

        graded_results = [_format_graded_chunk(chunk) for chunk in graded_chunks]
        pipeline_status = f"SUCCESS_{retrieval_action}"

        return {
            "query": query,
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
            "answer": answer,
        }

    def close(self) -> None:
        """Close connections and release resources."""
        self.grader.close()
        self.generator.close()
        self.web_search.close()
        self.db.close()
        self.embedder.close()


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
