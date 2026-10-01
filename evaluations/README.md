# `evaluations` compatibility package

The generation judge implementation lives in [`src/legal_rag/evaluation/generation/`](../src/legal_rag/evaluation/generation/). This top-level package remains only as a deprecated compatibility alias for `python -m evaluations`.

Use the shared command for new evaluations so retrieval and generation scores share one run identity and manifest:

```powershell
uv run --extra naive-rag legal-evaluate run `
  --pipeline naive `
  --naive-search-mode hybrid `
  --max-samples 20 `
  --limit 10 `
  --reference-date 2026-10-01 `
  --generation `
  --run-dir data/evaluation/runs/naive_hybrid_smoke
```

The judge reports six metrics:

- `faithfulness`
- `correctness`
- `answer_relevancy`
- `temporal_correctness`
- `answerability_correctness`
- `required_claim_coverage`

The last three can be null when a metric does not apply. Aggregates exclude nulls. Unified `metrics.json` uses `generation_metric_coverage` for the scored and null counts; the standalone report uses `metric_coverage`.

Temporal judging uses canonical `query_dates` first, then the legacy
`query_date`, explicit dates in the question, and finally the run's frozen
`reference_date` for current-law questions. Multi-version samples therefore
must preserve every authoritative target date in ground truth; the judge does
not use the machine's wall clock.

For an existing predictions JSONL file, the compatibility command still accepts the predictions path positionally:

```powershell
uv run python -m evaluations data/evaluation/runs/naive_hybrid_smoke/predictions.jsonl `
  --output data/evaluation/runs/naive_hybrid_smoke/generation-standalone.json `
  --batch-size 5 `
  --concurrency 2
```

The standalone path calls the external judge model but does not reconstruct the full run manifest. Prefer `legal-evaluate run --generation` for benchmark comparisons. No full benchmark result is asserted by this document.
