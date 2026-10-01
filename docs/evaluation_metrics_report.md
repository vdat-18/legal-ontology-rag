# Shared RAG evaluation metric contract

The evaluation package reports retrieval and answer quality for Naive, Adaptive, Corrective, Graphiti, and Ontology RAG through one artifact schema. This document describes the implemented metrics and how to compare them. It does not contain benchmark results: a full external-infrastructure benchmark has not been executed from this checkout.

## Retrieval metrics

Retrieval is scored without an LLM. Each backend emits ranked, pipeline-native units plus the source evidence attached to each unit. A unit may be a chunk, a Graphiti fact, an ontology rule, or a bounded structural context. This distinction prevents a graph fact or ontology rule from being presented as if it were one raw chunk.

| Metric | Range | Meaning |
|---|---:|---|
| `hit_rate` | 0-1 | Fraction of labeled samples with at least one relevant returned unit |
| `hit_rate_at_k` | 0-1 | Same test within the first `k` units |
| `precision_at_k` | 0-1 | Relevant native units among the first `k` positions |
| `recall_at_k` | 0-1 | Gold source evidence covered by the first `k` units |
| `context_precision` | 0-1 | Precision across the full returned unit list |
| `context_recall` | 0-1 | Gold source evidence covered by the full returned list |
| `empty_retrieval_rate` | 0-1 | Fraction of labeled samples with no returned units |
| `claim_evidence_any_at_k` | 0-1 or null | Mean fraction of required claims with at least one evidence chunk retrieved |
| `claim_evidence_complete_at_k` | 0-1 or null | Mean fraction of required claims whose complete evidence set was retrieved |

The aggregate includes `retrieval_sample_count` and `claim_sample_count` so consumers can see the applicable denominators. Claim metrics are null when no evaluated sample has claim-level evidence labels.

`--limit` is the per-question native-unit budget. `--max-samples` bounds the loaded evaluation samples after nested conversations are expanded into turns. They are independent controls. Keep `--limit >= max(--k-values)` for an interpretable cutoff comparison.

## Generation metrics

With `legal-evaluate run --generation`, the answer and retrieved context are passed to the BAML judge during the same run. The aggregate is written under `generation`; per-sample scores and coverage are written to `generation.json`.

| Metric | Range | Meaning |
|---|---:|---|
| `faithfulness` | 0-1 | Proportion of answer claims supported by retrieved context |
| `correctness` | 1-5 | Factual agreement with the expected answer |
| `answer_relevancy` | 1-5 | Whether the answer addresses the question directly |
| `temporal_correctness` | 0/1 or null | Whether the answer applies the correct legal version; null for an abstention |
| `answerability_correctness` | 0/1 or null | Whether answering, qualifying, or abstaining matches the gold answerability label; null when the label is absent |
| `required_claim_coverage` | 0-1 or null | Fraction of required gold claims conveyed correctly; null when required claims are absent |

Nullable values are excluded from a metric mean rather than converted to zero. `generation_metric_coverage` reports `scored_count`, `null_count`, and `coverage` for every judge metric. Compare a mean together with its coverage, especially for temporal, answerability, and required-claim scores.

Generation scores are comparable only when the answer-model settings and judge model, prompt, batch size, and concurrency match. The comparison command validates these controls when generation metrics are present.

## Run artifacts and identity

An end-to-end run writes the following files under its run directory:

- `manifest.json`: status, deterministic run ID, experiment identity, and output paths;
- `predictions.jsonl`: per-sample answer, ranked units, source evidence, latency, and provider usage when available;
- `metrics.json`: aggregate retrieval metrics, slice metrics, and optional generation aggregate;
- `generation.json`: per-sample judge report when `--generation` is enabled.

Retrieval-only evaluation writes mode-specific predictions and metrics plus a sibling `*.manifest.json`. The manifests fingerprint the ground truth, local corpus, effective settings, relevant source and prompt files, and material run controls. Active datastore preflights reject a missing or stale index before benchmark samples are processed; a local corpus digest by itself does not prove that Milvus or Neo4j contains matching data.

Comparison resolves every declared output relative to the manifest that names it and verifies its SHA-256 digest before reading it. For generation runs it also binds per-sample judge IDs to the prediction rows, recomputes aggregate scores and coverage from those rows, and requires the recomputed values to match both `generation.json` and `metrics.json`. Same-pipeline ablations may vary their explicitly selected search or ranker mode; their remaining effective configuration and pipeline answer/planning prompts must match.

Provider token counters remain unavailable when the provider does not return usage. The comparison report exposes usage-call coverage and does not replace missing counters with zero.

## Example commands

Retrieval-only Naive evaluation:

```powershell
uv run --extra naive-rag legal-evaluate retrieve `
  --retriever naive `
  --naive-search-mode hybrid_rerank `
  --ground-truth data/benchmark_synthesis/groundtruth.jsonl `
  --max-samples 20 `
  --limit 10 `
  --k-values 1,3,5,10
```

End-to-end Naive evaluation with generation judging:

```powershell
uv run --extra naive-rag legal-evaluate run `
  --pipeline naive `
  --naive-search-mode hybrid_rerank `
  --ground-truth data/benchmark_synthesis/groundtruth.jsonl `
  --max-samples 20 `
  --limit 10 `
  --k-values 1,3,5,10 `
  --concurrency 2 `
  --reference-date 2026-10-01 `
  --generation `
  --judge-batch-size 5 `
  --judge-concurrency 2 `
  --run-dir data/evaluation/runs/naive_hybrid_rerank_smoke
```

Use the `naive-rag` extra for Naive, Adaptive, and Ontology runs and the `corrective-rag` extra for Corrective runs. Graphiti uses the base Graphiti dependency and additionally requires reachable Neo4j state plus a matching `--corpus-config`.

Compare completed artifacts by naming each metrics file explicitly:

```powershell
uv run legal-evaluate compare `
  data/evaluation/runs/naive_hybrid_smoke/metrics.json `
  data/evaluation/runs/naive_hybrid_rerank_smoke/metrics.json `
  --output data/evaluation/runs/naive_hybrid_comparison.json
```

The comparison report validates population and experiment compatibility, then groups native-unit metrics, evidence coverage, generation quality, context budgets, operational observations, and configured slices. It deliberately has no composite score. `--allow-incompatible` returns compatibility diagnostics; it does not make the reported values comparable.

## External execution boundary

Unit and integration tests can verify schemas, aggregation, identity checks, and adapter behavior without claiming external results. Real benchmark execution requires the configured Milvus or Neo4j index, local embedding artifacts, and any answer, rewrite, or judge model endpoints used by the selected pipeline. Record conclusions only from artifacts whose manifest status is `complete`.
