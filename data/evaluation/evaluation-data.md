# Evaluation Data

This directory stores generated evaluation data.

- `corpus-snapshot.json` fingerprints the source archive and document selection.
- `groundtruth-corpus-v1-snapshot.json` fingerprints the first ground-truth
  source corpus.
- `candidates.jsonl` contains machine-generated samples waiting for legal review.
- `ground_truth.jsonl` contains the active evaluation samples. The current
  `retrieval_content_v2` rows are deterministic starter samples generated from
  content-bearing legal chunks and marked `starter_needs_legal_review`.
- `predictions.jsonl` contains CRAG outputs produced from reviewed samples.
- `metrics.json` contains aggregate offline metrics.

Offline retrieval metrics include:

- `precision_at_k`
- `recall_at_k`
- `mrr`
- `ndcg_at_k`
- `hit_rate` and `hit_rate_at_k`

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
python -m legal_rag.evaluation retrieve --ground-truth data/evaluation/ground_truth.jsonl --limit 10 --k-values 1,3,5,10
```

Run CRAG against reviewed ground truth and score the output:

```powershell
python -m legal_rag.evaluation run --ground-truth data/evaluation/ground_truth.jsonl --k-values 1,3,5,10
```

The `src/legal_rag/evaluation/` package only scores existing prediction or
retrieval JSONL files. Benchmark synthesis code lives in
`src/legal_rag/benchmark_synthesis/`; it no longer uses Ragas or `RAGAS_*`
environment variables. Do not edit packages inside `.venv`.
