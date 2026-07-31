# Benchmark Synthesis Data

This directory stores generated ground-truth synthesis data. The pipeline
design, schemas, and implementation plan live in
[docs/groundtruth_synthesis_pipeline.md](../../docs/groundtruth_synthesis_pipeline.md).

- `corpus-snapshot.json` fingerprints the source archive and document selection.
- `groundtruth-corpus-v1-snapshot.json` fingerprints the first ground-truth
  source corpus.
- `legal_facts.jsonl` is the validated corpus-level fact assembly.
- `run_manifest.json` records the snapshot, model, prompt/config hashes,
  chunk schema/config hashes, completion state, and deterministic run id.
- `evidence_relations.jsonl`, `evidence_relation_progress.jsonl`, and
  `scenarios.jsonl` are the evidence-graph phase outputs.
- `candidates/single_turn.jsonl` holds generated candidates and gate results.
- `candidates/single_turn.progress.jsonl` checkpoints generation, critique,
  evolution, binding, and verification per scenario attempt.
- `reviews/` and `ground_truth/` hold reviewed output, as described in the
  pipeline doc's "Cấu trúc output đề xuất" section.

Generation code lives in `src/legal_rag/benchmark_synthesis/`, following the phased
technical plan in the pipeline doc. Required environment variables are listed
in `.env.example` under `GROUNDTRUTH_*` and `EMBEDDING_PROFILE`. Do not edit
packages inside `.venv`.

Every LLM phase uses bounded retry and durable resume. A task is added to a
progress file only after its complete typed output is available. Exhausted
tasks remain incomplete and make the command exit with an error, while prior
successful work remains available for the next run.

Per-document screening, metadata, fact, and extraction-progress artifacts stay
beside their corresponding `*.chunks.jsonl` files in
`data/processed/groundtruth/`. Run `legal-assemble-groundtruth` before building
the evidence graph; it rejects incomplete phases, missing chunks, non-exact
evidence spans, inconsistent metadata, and conflicting fact identities.

The current MVP taxonomy is intentionally limited to penalty-schema queries.
Authority, procedure, and exception questions remain out of scope until their
own fact schemas exist.
