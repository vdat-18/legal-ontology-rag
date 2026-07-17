# legal-ontology-rag

Temporal Ontology-Enhanced RAG for **time-aware traffic-violation penalty
retrieval** (Vietnamese road-traffic law). Built on
[Graphiti](https://github.com/getzep/graphiti) (bi-temporal knowledge graph) with
Claude for extraction.

See [AGENTS.md](AGENTS.md) for the repository coding standards.

## Project structure

```
src/legal_rag/            # application package (src layout)
├── config.py             # environment-driven settings
├── ingestion/            # DOCX/PDF -> article-level, time-stamped episodes
├── ontology/             # custom legal entity/edge types for extraction
├── embeddings/           # Vietnamese (BGE-M3) embedder
├── graph/                # Graphiti client + ingestion orchestration
├── retrieval/            # point-in-time (bi-temporal) querying
└── cli.py                # `legal-rag ingest` / `legal-rag query`
data/
├── raw/                  # distributable corpus archive (.zip)
└── config/               # penalty_corpus.json (curated in-scope documents)
tests/                    # offline loader tests
third_party/graphiti/     # vendored graphiti-core (git-ignored)
docs/                     # plans and references
```

## Usage

```bash
pip install -e ".[vietnamese]"          # or ".[openai]"
cp .env.example .env                    # then fill Neo4j + ANTHROPIC_API_KEY

legal-rag ingest --reset                # build the temporal graph
legal-rag query "vượt đèn đỏ xe máy" 2025-02-01
```

Neo4j must be running (e.g. `docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.26`).
The loader reads documents directly from the corpus zip; no extraction needed.

## Development

Install the local checks once:

```bash
python -m pip install pre-commit ruff==0.15.21 pytest
pre-commit install
```

Run quality checks before committing:

```bash
ruff format .
ruff check .
pytest
pre-commit run --all-files
```

Keep changes focused, include tests for changed behavior, and explain any
intentional exception to `AGENTS.md` or Ruff. Never commit secrets, unintended
generated artifacts, or local environment files.
