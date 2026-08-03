# evaluations — LLM-judge evaluation harness

Standalone evaluation harness (outside `src/`, like `tests/`) for the Legal RAG pipelines. It
scores generated answers with a **BAML** LLM judge using the **QUEST** human-evaluation rubric
(Tam et al., npj Digital Medicine 2024), adapted to Vietnamese traffic-penalty QA, plus grounding
metrics.

It reuses the project's BAML runtime (same `LLM_*` settings as the rest of the codebase) — no extra
dependencies. It complements `legal_rag.evaluation` (deterministic retrieval metrics: precision@k,
recall@k, MRR, nDCG); this harness covers the **answer-quality and grounding** side, which is a
semantic judgment made by an LLM judge rather than a handcrafted score.

## Layout

```
evaluations/
  cli.py  __main__.py        # `python -m evaluations`
  core/    dataset.py        # EvalSample + flexible JSONL loader
           report.py         # aggregate + per-sample report
  quest/   rubric.py         # QUEST metric registry (criteria live in the BAML prompt)
  baml/    judge.py          # BamlJudgeClient: one typed call scores all metrics for a sample
           runner.py         # judge every sample concurrently, bounded by a semaphore
```

`core/`, `quest/`, and `baml/runner.py` are dependency-light and unit-tested; `baml/judge.py`
imports the generated BAML client only when actually run.

## How it runs (fast)

- **Batched per sample:** one `EvaluateLegalAnswer` BAML call returns *all* metrics for a sample as
  a typed object, so a sample costs one LLM call instead of one per metric.
- **Concurrent:** samples are judged in parallel through `BamlCaller.execute_async`, bounded by
  `--concurrency` so the shared provider is not overwhelmed.

## Metrics (one typed BAML judgment per sample)

QUEST rubric, scored 1–5 (`baml_src/answer_evaluation.baml`):

- `quest_quality_of_information` — accuracy, relevance, completeness, currency
- `quest_understanding_and_reasoning` — grasp of the question and legal reasoning
- `quest_expression_style` — clarity and structure
- `quest_safety_and_harm` — freedom from harmful fabrication
- `quest_trust_and_confidence` — grounding and verifiability

Grounding and correctness:

- `faithfulness` (1–5) — claims are supported by the retrieved contexts
- `answer_relevancy` (1–5) — the answer addresses the question
- `factual_correctness` (1–5, `null` when no gold answer) — match against the gold answer
- `temporal_correctness` (0/1) — applies the law version in force at the relevant time

## Input

A JSONL predictions file, one object per line, produced by running a pipeline over the
ground-truth questions. Fields are read flexibly:

- question: `question` / `query` / `user_input`
- answer: `response` / `answer` / `actual_answer`
- retrieved contexts: `retrieved_contexts` / `contexts` (strings or `{text}` objects)
- gold answer (optional): `reference` / `expected_answer` / `ground_truth`

## Run

```bash
# BAML client must be generated once (project setup): baml generate
python -m evaluations data/evaluation/predictions.jsonl \
  --output data/evaluation/quest_metrics.json --concurrency 8
```

The judge reuses `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`.
