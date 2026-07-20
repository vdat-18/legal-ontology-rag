# Evaluation Data

This directory stores generated evaluation data.

- `corpus-snapshot.json` fingerprints the source archive and document selection.
- `groundtruth-corpus-v1-snapshot.json` fingerprints the first ground-truth
  source corpus.
- `candidates.jsonl` contains machine-generated samples waiting for legal review.
- `ground_truth.jsonl` contains only reviewed and approved samples.

Generation code lives in `src/legal_rag/evaluation/`. Use the Python service
API `generate_groundtruth_candidates(...)` for candidate generation. It uses
Ragas through the `evaluation` optional dependency. Do not edit packages inside
`.venv`.

Before running Ragas generation, copy `.env.example` to `.env` and set
`OPENAI_API_KEY`, `RAGAS_LLM_PROVIDER`, `RAGAS_LLM_MODEL`,
`RAGAS_EMBEDDING_PROVIDER`, and `RAGAS_EMBEDDING_MODEL`.

Example:

```python
from datetime import date

from dotenv import load_dotenv

from legal_rag.config import Settings
from legal_rag.evaluation import generate_groundtruth_candidates

load_dotenv()
generate_groundtruth_candidates(
    Settings.from_env(),
    testset_size=20,
    as_of_date=date(2025, 2, 1),
)
```
