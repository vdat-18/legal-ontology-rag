# Evaluation Data

This directory stores generated evaluation data.

- `corpus-snapshot.json` fingerprints the source archive and document selection.
- `candidates.jsonl` contains machine-generated samples waiting for legal review.
- `ground_truth.jsonl` contains only reviewed and approved samples.

Generation code lives in `src/legal_rag/evaluation/`. It uses Ragas through the
`evaluation` optional dependency. Do not edit packages inside `.venv`.
