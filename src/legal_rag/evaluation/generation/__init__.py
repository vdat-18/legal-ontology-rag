"""Generation-quality evaluation: LLM-judge metrics for Legal RAG answers.

Metrics (6 generation metrics via LLM judge):
- faithfulness (0.0-1.0): Percentage of claims supported by retrieved context
- correctness (1-5): Matching with expected answer (LLM judge)
- answer_relevancy (1-5): Relevance to the question (LLM judge)
- temporal_correctness (0/1, null when the answer abstains): Applying correct law version
- answerability_correctness (0/1, null without a gold label): Appropriate answer/qualification/abstention
- required_claim_coverage (0.0-1.0, null without required claims): Expected claims correctly conveyed
"""

from legal_rag.evaluation.generation.core.dataset import EvalSample, load_samples
from legal_rag.evaluation.generation.core.report import EvaluationReport
from legal_rag.evaluation.generation.runner import evaluate_samples

__all__ = ["EvalSample", "EvaluationReport", "evaluate_samples", "load_samples"]
