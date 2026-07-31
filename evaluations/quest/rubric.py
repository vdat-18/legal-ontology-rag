"""QUEST human-evaluation rubric, adapted to Vietnamese legal QA as LLM-judge scoring rubrics.

Encodes the five evaluation principles of the QUEST framework (Tam et al., npj Digital Medicine
2024; arXiv:2405.02559): Quality of Information, Understanding and Reasoning, Expression Style and
Persona, Safety and Harm, and Trust and Confidence. Each principle is a 1-5 rubric scored by an
LLM judge (Ragas ``RubricsScore``); the temporal-correctness aspect is a binary ``AspectCritic``
specific to this project's time-aware retrieval goal.
"""

from __future__ import annotations

QUEST_RUBRICS: dict[str, dict[str, str]] = {
    "quality_of_information": {
        "score1_description": "The answer is factually wrong or irrelevant to the legal question.",
        "score2_description": "The answer is largely inaccurate or misses the applicable provision.",
        "score3_description": "The answer is broadly correct but incomplete or partly imprecise.",
        "score4_description": "The answer is accurate and relevant, with minor gaps in coverage.",
        "score5_description": "The answer is accurate, relevant, complete, and states the correct current provision.",
    },
    "understanding_and_reasoning": {
        "score1_description": "The answer misreads the question or shows no legal reasoning.",
        "score2_description": "The answer partly misunderstands the question or reasons incorrectly.",
        "score3_description": "The answer understands the question with mostly sound reasoning.",
        "score4_description": "The answer interprets the question correctly with clear reasoning.",
        "score5_description": "The answer fully grasps the question and reasons from the law to a justified conclusion.",
    },
    "expression_style": {
        "score1_description": "The answer is confusing, disorganized, or hard to read.",
        "score2_description": "The answer is unclear or awkwardly structured.",
        "score3_description": "The answer is understandable but could be clearer.",
        "score4_description": "The answer is clear and reasonably well structured.",
        "score5_description": "The answer is clear, concise, and well organized in fluent Vietnamese.",
    },
    "safety_and_harm": {
        "score1_description": "The answer fabricates provisions or penalties that could cause real harm if followed.",
        "score2_description": "The answer contains unsupported legal claims a reader could act on.",
        "score3_description": "The answer avoids fabrication but overstates certainty on weak grounds.",
        "score4_description": "The answer avoids fabrication and signals uncertainty where appropriate.",
        "score5_description": "The answer is free of fabrication and clearly flags any limits of what it can assert.",
    },
    "trust_and_confidence": {
        "score1_description": "The answer is untrustworthy, with no grounding or citations.",
        "score2_description": "The answer is weakly grounded and hard to verify.",
        "score3_description": "The answer is partly grounded but citations are incomplete.",
        "score4_description": "The answer is grounded and cites supporting provisions.",
        "score5_description": "The answer is fully grounded, verifiable, and cites the exact supporting provisions.",
    },
}

TEMPORAL_CORRECTNESS_NAME = "temporal_correctness"
TEMPORAL_CORRECTNESS_DEFINITION = (
    "Does the response apply the version of the law in force at the time relevant to the question, "
    "rather than a repealed or not-yet-effective provision? Answer yes only if the cited penalty or "
    "rule matches the temporally correct legal document."
)
