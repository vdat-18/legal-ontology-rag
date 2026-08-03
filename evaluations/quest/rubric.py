"""QUEST evaluation dimensions (Tam et al., npj Digital Medicine 2024) as legal-QA judge metrics.

The five QUEST principles — Quality of Information, Understanding and Reasoning, Expression Style
and Persona, Safety and Harm, and Trust and Confidence — are scored 1-5 by an LLM judge, plus a
binary temporal-correctness aspect specific to this project's time-aware retrieval goal. The full
1-5 scoring criteria live in the BAML judge prompt (``baml_src/answer_evaluation.baml``); this
module is the canonical registry of metric names and their one-line meaning used to label reports.
"""

from __future__ import annotations

QUEST_DESCRIPTIONS: dict[str, str] = {
    "quality_of_information": "Accuracy, relevance, completeness, and currency of the legal answer.",
    "understanding_and_reasoning": "Grasp of the question and soundness of the legal reasoning.",
    "expression_style": "Clarity and structure of the answer.",
    "safety_and_harm": "Freedom from fabricated law or penalties that could cause harm.",
    "trust_and_confidence": "Grounding, citations, and verifiability of the answer.",
}

QUEST_DIMENSIONS: tuple[str, ...] = tuple(QUEST_DESCRIPTIONS)
TEMPORAL_CORRECTNESS_NAME = "temporal_correctness"
