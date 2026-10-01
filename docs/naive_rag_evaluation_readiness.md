# Naive RAG evaluation guide

## Current readiness

The Naive RAG implementation and the shared evaluation harness support all five configured search modes. This means the code path can be exercised consistently; it does not mean that an external benchmark result already exists. A full-corpus, five-mode benchmark has **not** been executed from this checkout.

An actual run still depends on:

- a reachable Milvus service and the collection named by `NAIVE_RAG_MILVUS_COLLECTION_NAME`;
- the local `*.chunks.jsonl` corpus under `NAIVE_RAG_CHUNKS_DIR`;
- the embedding model selected by `EMBEDDING_PROFILE`;
- an answer-model endpoint for `legal-evaluate run`; and
- a judge-model endpoint when `--generation` is enabled.

Evaluation preflights the active Milvus collection against the configured local corpus before processing samples. A missing collection, extra or missing chunk IDs, or stale persisted chunk metadata fails the run with a rebuild instruction. Files on disk alone are not evidence that the active collection is ready.

## Search modes

All five modes use the same Milvus collection. They differ only in how candidates are selected or expanded.

| Mode | Retrieval behavior |
|---|---|
| `keyword` | BM25L lexical retrieval |
| `semantic` | Dense-vector retrieval |
| `hybrid` | Reciprocal-rank fusion of lexical and dense candidates |
| `hybrid_rerank` | Hybrid candidate pool followed by Cross-Encoder reranking |
| `hierarchical` | Hybrid seeds expanded into bounded article-level units when the local structure is unambiguous |

Hierarchical mode also reads `NAIVE_RAG_CHUNKS_DIR` to construct its structural catalog. The same active-index preflight therefore matters for both retrieval correctness and structural expansion.

## CLI controls that must not be confused

- `--max-samples` limits loaded evaluation samples and is the control to use for a smoke test. Nested conversations are expanded into turn-level samples first.
- `--limit` is the maximum number of ranked retrieval units supplied for each question. It does not limit the number of questions.
- `--k-values` selects the retrieval cutoffs. Keep `--limit` at least as large as the greatest requested `k`.
- `--concurrency` controls concurrent sample evaluation. Keep it equal across latency comparisons.
- `--reference-date` freezes relative temporal questions. Pass the same explicit date to every compared run.
- Query rewriting for history-dependent rows is enabled by default. `--no-rewrite-query` is an ablation and produces a different experiment identity.
- `--generation` judges generated answers during the same run; there is no required second judging step.

The run manifest records the dataset and corpus digests, effective retrieval settings, model and prompt identities, temporal and history policies, concurrency, and output paths. Comparison requires shared benchmark controls to match while preserving the intended pipeline or search-mode difference.

## Prepare the active index

Install the Naive RAG runtime and index the configured corpus:

```powershell
uv sync --extra naive-rag
uv run --extra naive-rag legal-naive-rag-index data/processed/groundtruth
```

The indexer is resumable for a matching partial collection and rejects stale, mixed, or out-of-scope rows. Starting Milvus and choosing endpoint credentials are environment-specific; the repository does not assert that a local or remote service is currently running or indexed.

## Retrieval-only smoke test

This path does not call the answer model or judge, but it still needs Milvus and the embedding runtime:

```powershell
uv run --extra naive-rag legal-evaluate retrieve `
  --retriever naive `
  --naive-search-mode hybrid `
  --ground-truth data/benchmark_synthesis/groundtruth.jsonl `
  --max-samples 20 `
  --limit 10 `
  --k-values 1,3,5,10
```

The default outputs are addressed by the complete experiment identity:

- `data/evaluation/runs/<run-id>/predictions.jsonl`
- `data/evaluation/runs/<run-id>/metrics.json`
- `data/evaluation/runs/<run-id>/metrics.manifest.json`

Repeating an identical completed retrieval run validates and replays these
artifacts. A different identity, partial output, or colliding explicit path is
rejected instead of overwritten.

## End-to-end five-mode smoke test

The following PowerShell loop keeps the sample population, retrieval budget, temporal policy, and judge controls fixed. Each mode receives its own run directory.

```powershell
$modes = @("keyword", "semantic", "hybrid", "hybrid_rerank", "hierarchical")

foreach ($mode in $modes) {
  uv run --extra naive-rag legal-evaluate run `
    --pipeline naive `
    --naive-search-mode $mode `
    --ground-truth data/benchmark_synthesis/groundtruth.jsonl `
    --max-samples 20 `
    --limit 10 `
    --k-values 1,3,5,10 `
    --concurrency 2 `
    --reference-date 2026-10-01 `
    --generation `
    --judge-batch-size 5 `
    --judge-concurrency 2 `
    --run-dir "data/evaluation/runs/naive_${mode}_smoke"
}
```

Each completed directory contains `manifest.json`, `predictions.jsonl`, `metrics.json`, and, with `--generation`, `generation.json`. An interrupted compatible run can continue with `--resume`; a completed or incompatible run directory is rejected instead of being silently overwritten.

After validating the smoke artifacts, run the full dataset by removing `--max-samples` and choosing new run directories. That full benchmark invokes external model services and is intentionally not presented here as already completed.

## Compare the five modes

```powershell
uv run legal-evaluate compare `
  data/evaluation/runs/naive_keyword_smoke/metrics.json `
  data/evaluation/runs/naive_semantic_smoke/metrics.json `
  data/evaluation/runs/naive_hybrid_smoke/metrics.json `
  data/evaluation/runs/naive_hybrid_rerank_smoke/metrics.json `
  data/evaluation/runs/naive_hierarchical_smoke/metrics.json `
  --output data/evaluation/runs/naive_smoke_comparison.json
```

Comparison validates the evaluated sample set and material run controls before displaying results. It keeps native-unit retrieval, source-evidence coverage, generation quality, context budget, and operational observations separate; it does not calculate a synthetic composite score. Use `--allow-incompatible` only to produce diagnostics for runs that must not be treated as a fair comparison.

## Interpreting readiness

Code and tests establish that the five modes share one artifact contract and can be compared under controlled settings. Benchmark readiness is established at run time by the datastore preflight and provider availability. Performance claims should be made only from completed manifests and their exact metrics artifacts.
