# evaluations — LLM-judge evaluation harness

Standalone evaluation harness (outside `src/`, like `tests/`) for the Legal RAG pipelines. It
scores generated answers with [Ragas](https://docs.ragas.io) metrics and the **QUEST** human-
evaluation rubric, adapted to Vietnamese traffic-penalty QA.

It complements `legal_rag.evaluation` (deterministic retrieval metrics: precision@k, recall@k,
MRR, nDCG). This harness covers the **answer-quality and grounding** side, which is a semantic
judgment and therefore made by an LLM judge rather than a handcrafted score.

## Layout

```
evaluations/
  cli.py  __main__.py        # `python -m evaluations`
  core/    dataset.py        # EvalSample + flexible JSONL loader
           report.py         # aggregate + per-sample report
  quest/   rubric.py         # QUEST rubric (5 principles) + temporal aspect
  ragas/   evaluator.py      # wrap the project LLM + embeddings as a Ragas judge
           metrics.py        # assemble the metric suite
           runner.py         # run ragas.evaluate and reduce to a report
```

`core/` and `quest/` are dependency-light and unit-tested; `ragas/` imports Ragas/LangChain only
when actually run.

## Metrics

Core Ragas RAG metrics:

| Metric | Needs gold answer | Measures |
| --- | --- | --- |
| `faithfulness` | no | Answer claims are entailed by the retrieved contexts (no fabrication) |
| `answer_relevancy` | no | Answer actually addresses the question |
| `llm_context_precision_with_reference` | yes | Retrieved contexts are relevant and well ranked |
| `context_recall` | yes | Retrieved contexts cover the gold answer |
| `factual_correctness` | yes | Claim-level precision/recall/F1 vs the gold answer |
| `semantic_similarity` | yes | Embedding similarity of answer to the gold answer |

QUEST rubric (Tam et al., *A framework for human evaluation of large language models in
healthcare*, npj Digital Medicine 2024; [arXiv:2405.02559](https://arxiv.org/abs/2405.02559)),
each scored 1–5 by the judge LLM (`quest/rubric.py`):

- `quest_quality_of_information` — accuracy, relevance, completeness, currency
- `quest_understanding_and_reasoning` — grasp of the question and legal reasoning
- `quest_expression_style` — clarity and structure
- `quest_safety_and_harm` — freedom from harmful fabrication
- `quest_trust_and_confidence` — grounding and verifiability

Plus a project-specific binary aspect:

- `temporal_correctness` — the answer applies the law version in force at the relevant time.

## Input

A JSONL predictions file, one object per line, produced by running a pipeline over the
ground-truth questions. Fields are read flexibly:

- question: `question` / `query` / `user_input`
- answer: `response` / `answer` / `actual_answer`
- retrieved contexts: `retrieved_contexts` / `contexts` (strings or `{text}` objects)
- gold answer (optional): `reference` / `expected_answer` / `ground_truth`
- gold contexts (optional): `reference_contexts` / `expected_contexts`

## Run

```bash
pip install -e ".[ragas-eval]"
python -m evaluations data/evaluation/predictions.jsonl --output data/evaluation/ragas_metrics.json
```

The judge LLM reuses `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`; the judge embeddings use the
`EMBEDDING_PROFILE` model (override with `--embedding-profile`).
