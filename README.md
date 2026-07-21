# legal-ontology-rag

Ground-truth synthesis from Vietnamese road traffic legal documents.

## Requirements

- Python 3.11+

## Setup

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -V
```

Install the project:

```bash
pip install -e ".[dev]"
```

## Data

Data links are listed in [docs/references/references.md](docs/references/references.md).

Main data Drive folder:

<https://drive.google.com/drive/folders/1RhVMqDr98r5IBqArSrzUYBvoNZTjtLxE>

Demo/data Drive folder:

<https://drive.google.com/drive/folders/1voxi753JGAzF7WfiKLxzIP2rJCLjuSmf>

## Source Extraction

Convert one source document into deterministic JSONL blocks:

```powershell
legal-groundtruth `
  data/raw/road_traffic_legal_documents/168_2024_ND-CP_619502.docx `
  data/processed/groundtruth/168_2024_ND-CP_619502.source-blocks.jsonl `
  --normalized-output data/processed/groundtruth/168_2024_ND-CP_619502.normalized-blocks.jsonl `
  --normalization-manifest data/processed/groundtruth/168_2024_ND-CP_619502.normalization.json
```

Docling performs structural parsing. The project stores each Docling item and
its provenance without regex, keyword matching, or legal interpretation. Raw
blocks remain unchanged; the optional normalized output excludes only empty
text items and records those references in the manifest.

## Checks

Run before committing:

```bash
ruff format .
ruff check .
pytest
```

## Project Layout

```text
src/legal_rag/       Ground-truth synthesis package
data/config/         Corpus selection config
data/raw/            Raw legal document archive
docs/references/     External data and research links
tests/               Offline tests
```
