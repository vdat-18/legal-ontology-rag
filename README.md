# legal-ontology-rag

Ground-truth synthesis from Vietnamese road traffic legal documents.

## Requirements

- Python 3.11+
- LibreOffice console executable for optional PDF sidecars

## Setup

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -V
```

Install the project:

```powershell
$env:UV_LINK_MODE="copy"
uv pip install -e ".[dev]"
```

`UV_LINK_MODE=copy` keeps native Docling resources as regular files when the
repository is inside OneDrive. Pass the LibreOffice console executable through
`--renderer-executable`; the application path is never embedded in source code.

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
  --normalization-manifest data/processed/groundtruth/168_2024_ND-CP_619502.normalization.json `
  --paragraph-output data/processed/groundtruth/168_2024_ND-CP_619502.paragraphs.jsonl `
  --paragraph-manifest data/processed/groundtruth/168_2024_ND-CP_619502.paragraphs.manifest.json `
  --table-output data/processed/groundtruth/168_2024_ND-CP_619502.tables.jsonl `
  --table-manifest data/processed/groundtruth/168_2024_ND-CP_619502.tables.manifest.json `
  --pdf-output data/processed/groundtruth/168_2024_ND-CP_619502.pdf `
  --pdf-block-output data/processed/groundtruth/168_2024_ND-CP_619502.pdf-blocks.jsonl `
  --pdf-manifest data/processed/groundtruth/168_2024_ND-CP_619502.pdf.manifest.json `
  --renderer-executable "$env:LOCALAPPDATA/legal-rag-tools/libreoffice-26.2.4.2/app/program/soffice.com" `
  --render-timeout-seconds 300
```

Docling performs structural parsing. The project stores each Docling item and
its provenance without regex, keyword matching, or legal interpretation. Raw
blocks remain unchanged; the optional normalized output excludes only empty
text items and records those references in the manifest.

Source-block schema `1.1.0` adds an explicit zero-based `sequence_index` in
Docling traversal order. Regenerate older JSONL outputs from their hashed DOCX
source; normalized outputs preserve the original indices and do not renumber
after skipping empty text items.

Native paragraph output is read directly from `word/document.xml`. It keeps all
visible runs from one top-level Word paragraph together, including vertical
alignment such as superscript, without guessing from text content. Table cells
remain outside this paragraph view and will be handled by a dedicated table
normalization layer.

Native table output preserves top-level rows, cells, cell paragraphs, runs,
and OOXML merge metadata. Nested tables fail explicitly until their recursive
schema is defined; they are never silently flattened.

The optional PDF sidecar is rendered by an explicit LibreOffice executable,
then parsed with Docling's `PyPdfiumDocumentBackend` and OCR disabled. PDF
blocks keep Docling page/bounding-box provenance; the manifest links the DOCX
hash, PDF hash, renderer version, backend, and page coverage. No automatic text
matching is used to link native OOXML units to PDF blocks.

LibreOffice includes render-time metadata in each PDF, so a later render may
have a different PDF hash. Page citations always belong to the exact immutable
PDF hash recorded in its manifest.

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
