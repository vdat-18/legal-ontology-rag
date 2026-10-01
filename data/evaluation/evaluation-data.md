# Evaluation Data

This directory stores generated evaluation data.

- `data/benchmark_synthesis/groundtruth-corpus-v1-snapshot.json` fingerprints the
  selected ground-truth source corpus.
- `data/benchmark_synthesis/groundtruth.jsonl` contains the active reviewed benchmark.
- `data/evaluation/runs/<run-id>/manifest.json` records the experiment identity,
  status, and output paths for one end-to-end run.
- `data/evaluation/runs/<run-id>/predictions.jsonl` contains normalized outputs
  from one RAG implementation.
- `data/evaluation/runs/<run-id>/metrics.json` contains its aggregate metrics.
- `data/evaluation/runs/<run-id>/generation.json` contains per-sample judge
  results when `--generation` is enabled.

Offline retrieval metrics include:

- `precision_at_k` - Percentage of relevant chunks in top-K
- `recall_at_k` - Percentage of gold chunks found in top-K
- `hit_rate` and `hit_rate_at_k` - Whether at least one relevant chunk appears in top-K

Supported metric fields in `groundtruth.jsonl` are flexible:

- Question: `question` or `query`.
- Expected answer: `expected_answer`, `ground_truth`, or `ground_truth_answer`.
- Expected retrieval labels: `expected_chunk_ids`, `reference_chunk_ids`,
  `ground_truth_chunk_ids`, `expected_source_names`, or
  `reference_source_names`.
- Temporal targets: canonical plural `query_dates` or legacy scalar
  `query_date`. Plural dates take precedence when both are present.

Run offline scoring for a prediction file:

```powershell
python -m legal_rag.evaluation score data/evaluation/predictions.jsonl --k-values 1,3,5,10 --output data/evaluation/metrics.json
```

Run retrieval-only evaluation without LLM calls:

```powershell
docker compose --profile milvus up -d milvus
python -m legal_rag.corrective_rag.api.cli index --data-dir data/processed/groundtruth
python -m legal_rag.evaluation retrieve `
  --ground-truth data/benchmark_synthesis/groundtruth.jsonl --limit 10 --k-values 1,3,5,10
```

Retrieval-only runs write predictions and metrics under the deterministic
`data/evaluation/runs/<run-id>/` directory plus a metrics-adjacent manifest.
Use `--predictions-output` and `--metrics-output` to select different paths.
Existing complete artifacts are digest-validated and replayed; partial,
different, or colliding output paths are rejected instead of overwritten.

Run each complete RAG implementation against the same synthesized ground truth. The shared runner
preserves the original ground-truth row and emits the same `actual_answer`, `retrieved_chunk_ids`,
`retrieved_source_names`, and `retrieved_contexts` fields for every pipeline:

```powershell
uv run --extra naive-rag legal-evaluate run --pipeline naive --limit 10 --k-values 1,3,5,10
uv run --extra corrective-rag legal-evaluate run --pipeline corrective --limit 10 --k-values 1,3,5,10
uv run legal-evaluate run --pipeline graphiti --limit 10 --k-values 1,3,5,10 `
  --group-id traffic_penalty_groundtruth_v1 `
  --corpus-config data/config/groundtruth_corpus_v1.json
```

By default, each command writes `manifest.json`, `predictions.jsonl`, and `metrics.json` under a
deterministic directory in `data/evaluation/runs/`. Add `--generation` to also write
`generation.json`. The manifest records the pipeline, retrieval limit, corpus identity, effective
configuration, and output paths. Index the same chunk corpus into each pipeline's backend before
comparing results. Graphiti requires `--corpus-config` so the local source selection and indexed
group can be validated. For Corrective RAG, retrieval metrics score the initial Milvus ranking while
faithfulness receives the refined internal or external knowledge actually passed to generation.

Multi-turn ground truth must be exported with the matching reviewed scenarios
and legal-facts release. The target scenario supplies `bundle_kind`; only true
temporal targets receive source-fact `valid_from` dates, and only from facts in
the final target's required claims.

The `src/legal_rag/evaluation/` package provides the shared runner, offline scoring,
retrieval-only evaluation, and comparison workflow. Benchmark synthesis code lives in
`src/legal_rag/benchmark_synthesis/`; it no longer uses Ragas or `RAGAS_*`
environment variables. Do not edit packages inside `.venv`.
