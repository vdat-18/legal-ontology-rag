"""Assemble the evaluation metric suite: core Ragas RAG metrics plus the QUEST LLM-judge rubric."""

from __future__ import annotations

from typing import Any

from evaluations.quest.rubric import QUEST_RUBRICS, TEMPORAL_CORRECTNESS_DEFINITION, TEMPORAL_CORRECTNESS_NAME


def build_metrics(evaluator_llm: Any, *, with_reference: bool) -> list[Any]:
    """Build the metric list; reference-based metrics are added only when gold answers are present.

    Faithfulness and response relevancy need no reference; context precision/recall, factual
    correctness, and semantic similarity compare against the gold answer. Each QUEST principle
    becomes one rubric-scored judgment and temporal correctness a binary aspect critique.
    """
    from ragas.metrics import (
        AspectCritic,
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
        RubricsScore,
        SemanticSimilarity,
    )

    metrics: list[Any] = [Faithfulness(), ResponseRelevancy()]
    if with_reference:
        metrics += [
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            FactualCorrectness(),
            SemanticSimilarity(),
        ]
    metrics += [
        RubricsScore(name=f"quest_{dimension}", rubrics=rubric, llm=evaluator_llm)
        for dimension, rubric in QUEST_RUBRICS.items()
    ]
    metrics.append(
        AspectCritic(name=TEMPORAL_CORRECTNESS_NAME, definition=TEMPORAL_CORRECTNESS_DEFINITION, llm=evaluator_llm)
    )
    return metrics
