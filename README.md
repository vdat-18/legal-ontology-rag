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

Generate the local BAML client from the versioned definitions:

```powershell
baml-cli check --from baml_src
baml-cli generate --from baml_src --no-tests
```

`src/legal_rag/baml_client/` is generated code and is intentionally excluded
from Git. Run these commands after cloning the repository and whenever a file
in `baml_src/` changes.

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

Native table output preserves top-level rows, cells, runs, and OOXML merge
metadata. Each cell holds an ordered list of content blocks, where a block is
either a paragraph or a nested table, so nested tables keep their own rows,
cells, and merge structure in document order instead of being flattened.

The optional PDF sidecar is rendered by an explicit LibreOffice executable,
then parsed with Docling's `PyPdfiumDocumentBackend` and OCR disabled. PDF
blocks keep Docling page/bounding-box provenance; the manifest links the DOCX
hash, PDF hash, renderer version, backend, and page coverage. No automatic text
matching is used to link native OOXML units to PDF blocks.

LibreOffice includes render-time metadata in each PDF, so a later render may
have a different PDF hash. Page citations always belong to the exact immutable
PDF hash recorded in its manifest.

## Unified Source View

Merge the native DOCX paragraphs and tables into one ordered source file:

```powershell
legal-parse `
  data/raw/road_traffic_legal_documents/168_2024_ND-CP_619502.docx `
  data/processed/groundtruth/168_2024_ND-CP_619502.source.jsonl
```

Each JSONL record is one native paragraph or table in native document order,
carrying its exact OOXML source reference and content hash. The command reads
`word/document.xml` directly, so tables keep their native row, cell, and merge
structure without Docling. By default it writes only this single file; pass
`--manifest <path>` to also write an audit sidecar that records skipped empty
paragraphs and unit counts. This merged view is the deterministic source for
downstream ground-truth synthesis; page provenance stays in the optional PDF
sidecar and is never matched back into this view.

## Image Captioning

Configure the `LLM_*` and `IMAGE_CAPTION_*` variables shown in `.env.example`.

Caption one DOCX directly without changing the deterministic source view:

```powershell
legal-caption-docx `
  data/raw/road_traffic_legal_documents/14_2025_TT-BXD_m_664151.docx `
  data/processed/groundtruth/14_2025_TT-BXD_m_664151.image-captions.jsonl `
  --renderer-executable "$env:LOCALAPPDATA/legal-rag-tools/libreoffice-26.2.4.2/app/program/soffice.com"
```

The command extracts images in memory, joins each occurrence to structural
context, calls the BAML `CaptionImage` function through 9Router, and writes only
the typed caption sidecar. Table images receive text from their native row. Paragraph images use
the same paragraph, or adjacent source units when the paragraph contains no
text. Context is bounded by `IMAGE_CAPTION_MAX_CONTEXT_CHARACTERS` and every
caption records the exact source references disclosed to the model.

PNG/JPEG assets are sent directly. Legacy EMF/WMF assets require the explicit
LibreOffice renderer and are converted to PNG in an isolated temporary
profile. Captions are model-derived enrichment and never modify source hashes
or native source records.

The prompt, multimodal input, and typed output schema live in `baml_src/`.
Regenerate the local Python client after changing BAML files:

```powershell
baml-cli check --from baml_src
baml-cli generate --from baml_src --no-tests
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
src/legal_rag/       Ground-truth synthesis package
data/config/         Corpus selection config
data/raw/            Raw legal document archive
docs/references/     External data and research links
tests/               Offline tests
```
