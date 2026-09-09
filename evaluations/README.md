# evaluations — LLM-judge evaluation harness (MOVED)

This harness moved to [`src/legal_rag/evaluation/generation/`](../src/legal_rag/evaluation/generation/),
next to the deterministic retrieval metrics in `legal_rag.evaluation`, so every RAG pipeline is
scored by both metric families in one place.

`python -m evaluations` still works as a deprecated alias for
`python -m legal_rag.evaluation.generation`.

See the module docstring in `src/legal_rag/evaluation/generation/__init__.py` for the metrics
(faithfulness, correctness, answer_relevancy, temporal_correctness) and the main project README
for the unified `legal-evaluate run --generation` command.
