"""Generation-quality evaluation: LLM-judge metrics for Legal RAG answers.

Metrics (4 generation metrics via LLM judge):
- faithfulness (0.0-1.0): Percentage of claims supported by retrieved context
- correctness (1-5): Matching with expected answer (LLM judge)
- answer_relevancy (1-5): Relevance to the question (LLM judge)
- temporal_correctness (binary 0/1): Applying correct law version
"""

from legal_rag.evaluation.generation.core.dataset import EvalSample, load_samples
from legal_rag.evaluation.generation.core.report import EvaluationReport
from legal_rag.evaluation.generation.runner import evaluate_samples

__all__ = ["EvalSample", "EvaluationReport", "evaluate_samples", "load_samples"]
