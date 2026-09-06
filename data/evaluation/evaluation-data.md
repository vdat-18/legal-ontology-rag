# Evaluation Data

This directory stores generated evaluation data.

- `corpus-snapshot.json` fingerprints the source archive and document selection.
- `groundtruth-corpus-v1-snapshot.json` fingerprints the first ground-truth
  source corpus.
- `candidates.jsonl` contains machine-generated samples waiting for legal review.
- `data/benchmark_synthesis/groundtruth.jsonl` contains the active reviewed benchmark.
- `<pipeline>_predictions.jsonl` contains normalized outputs from one RAG implementation.
- `<pipeline>_metrics.json` contains its aggregate deterministic metrics.

Offline retrieval metrics include:

- `precision_at_k` - Percentage of relevant chunks in top-K
- `recall_at_k` - Percentage of gold chunks found in top-K
- `hit_rate` and `hit_rate_at_k` - Whether at least one relevant chunk appears in top-K

Supported metric fields in `ground_truth.jsonl` are flexible:

- Question: `question` or `query`.
- Expected answer: `expected_answer`, `ground_truth`, or `ground_truth_answer`.
- Expected retrieval labels: `expected_chunk_ids`, `reference_chunk_ids`,
  `ground_truth_chunk_ids`, `expected_source_names`, or
  `reference_source_names`.

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

Run each complete RAG implementation against the same synthesized ground truth. The shared runner
preserves the original ground-truth row and emits the same `actual_answer`, `retrieved_chunk_ids`,
`retrieved_source_names`, and `retrieved_contexts` fields for every pipeline:

```powershell
uv run legal-evaluate run --pipeline naive --limit 10 --k-values 1,3,5,10
uv run legal-evaluate run --pipeline corrective --limit 10 --k-values 1,3,5,10
uv run legal-evaluate run --pipeline graphiti --limit 10 --k-values 1,3,5,10 `
  --group-id traffic_penalty_groundtruth_v1
```

The commands write pipeline-specific prediction and metric files under `data/evaluation/`; every
metric report records its pipeline and retrieval limit, so one run never overwrites or becomes
indistinguishable from another. Index the same chunk corpus into each pipeline's backend before
comparing results. For Corrective RAG, retrieval metrics score the initial Milvus ranking while
faithfulness receives the refined internal or external knowledge actually passed to generation.

The `src/legal_rag/evaluation/` package only scores existing prediction or
retrieval JSONL files. Benchmark synthesis code lives in
`src/legal_rag/benchmark_synthesis/`; it no longer uses Ragas or `RAGAS_*`
environment variables. Do not edit packages inside `.venv`.
