# legal-ontology-rag

Temporal legal RAG for Vietnamese road traffic penalties.

This project uses [Graphiti](https://github.com/getzep/graphiti) as the graph
memory layer. The code in this repository adds the legal-domain ingestion,
ontology, and point-in-time retrieval logic.

## Requirements

- Python 3.11+
- Neo4j 5.26+
- `ANTHROPIC_API_KEY` for extraction
- `OPENAI_API_KEY` only when `GRAPHITI_EMBEDDER=openai`

## Setup

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -V
```

Install the project:

```bash
pip install -e ".[vietnamese]"
```

Use OpenAI embeddings instead of local BGE-M3:

```bash
pip install -e ".[openai]"
```

Install Ragas and DeepEval for dataset generation and evaluation:

```bash
pip install -e ".[evaluation]"
```

Graphiti is installed from PyPI through `graphiti-core`. We do not keep a copy
of Graphiti source code in this repository.

To read Graphiti code, use one of these:

- `.venv/Lib/site-packages/graphiti_core/`
- <https://github.com/getzep/graphiti>

Do not edit files inside `.venv`. Those changes will be lost when dependencies
are reinstalled. Put long-term custom code in `src/legal_rag/`, or use a
separate Graphiti fork.

## Neo4j

Start Neo4j:

```bash
docker compose up -d
```

Default local credentials:

- User: `neo4j`
- Password: `password`
- Browser: <http://localhost:7474>

## Environment

Copy the example file:

```bash
copy .env.example .env
```

Set these values in `.env`:

```text
ANTHROPIC_API_KEY=...
GRAPHITI_EMBEDDER=bge
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Use `GRAPHITI_EMBEDDER=openai` only when you also set `OPENAI_API_KEY`.

## Data

The loader reads documents from:

```text
data/raw/road_traffic_legal_documents.zip
```

The selected penalty documents are listed in:

```text
data/config/penalty_corpus.json
```

Data links are listed in [docs/references/references.md](docs/references/references.md).

Main data Drive folder:

<https://drive.google.com/drive/folders/1RhVMqDr98r5IBqArSrzUYBvoNZTjtLxE>

Demo/data Drive folder:

<https://drive.google.com/drive/folders/1voxi753JGAzF7WfiKLxzIP2rJCLjuSmf>

## Run

Ingest the selected corpus:

```bash
legal-rag ingest --reset
```

Query facts valid at a date:

```bash
legal-rag query "vuot den do xe may" 2025-02-01
```

## Checks

Run before committing:

```bash
ruff format .
ruff check .
pytest
```

## Project Layout

```text
src/legal_rag/       Legal RAG package
data/config/         Corpus selection config
data/raw/            Raw legal document archive
docs/references/     External data and research links
tests/               Offline tests
```
