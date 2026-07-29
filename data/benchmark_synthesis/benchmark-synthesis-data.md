# Benchmark Synthesis Data

This directory stores generated ground-truth synthesis data. The pipeline
design, schemas, and implementation plan live in
[docs/groundtruth_synthesis_pipeline.md](../../docs/groundtruth_synthesis_pipeline.md).

- `corpus-snapshot.json` fingerprints the source archive and document selection.
- `groundtruth-corpus-v1-snapshot.json` fingerprints the first ground-truth
  source corpus.
- `legal_facts.jsonl`, `fact_ledger.jsonl`, `evidence_relations.jsonl` hold the
  fact ledger described in the pipeline doc.
- `candidates/` holds machine-generated samples waiting for review.
- `reviews/` and `ground_truth/` hold reviewed output, as described in the
  pipeline doc's "Cấu trúc output đề xuất" section.

Generation code lives in `src/legal_rag/benchmark_synthesis/`, following the phased
technical plan in the pipeline doc. Required environment variables are listed
in `.env.example` under `GROUNDTRUTH_*` and `EMBEDDING_PROFILE`. Do not edit
packages inside `.venv`.
