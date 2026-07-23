# legal-ontology-rag

Ground-truth synthesis from Vietnamese road traffic legal documents.

## Requirements

- Python 3.11+
- LibreOffice console executable when DOCX files contain legacy EMF/WMF images or DOC/XLS attachments

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

`UV_LINK_MODE=copy` avoids link-related installation issues when the repository
is inside OneDrive. Pass the LibreOffice console executable through
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

## Parser Pipeline

Configure the `LLM_*` variables shown in `.env.example`, then parse one DOCX
into the final enriched source file:

```powershell
legal-parse `
  data/raw/road_traffic_legal_documents/168_2024_ND-CP_619502.docx `
  data/processed/groundtruth/168_2024_ND-CP_619502.parsed.jsonl `
  --renderer-executable "$env:LOCALAPPDATA/legal-rag-tools/libreoffice-26.2.4.2/app/program/soffice.com"
```

Each JSONL record is one native paragraph or table in native document order,
carrying its exact OOXML source reference and content hash. Its `images` field
contains the typed captions for image occurrences at that body position. Its
`attachments` field contains metadata and provenance for embedded OLE packages.
Attachment payloads are written under the sibling `assets/` directory using
their SHA-256 hashes, while their display icons are excluded from image
captioning. Supported DOC, DOCX and XLS documents inside attachments are parsed
into child records immediately after their owning parent record. Child records
carry the parent attachment ID, archive member path and hash, source format and
conversion provenance. ZIP members are read with bounded count and byte limits
without extracting source paths to disk. Legacy DOC and XLS members are converted
in LibreOffice batches; XLSX rows retain sheet, cell reference, value and formula.
The pipeline reads native paragraphs, tables, images, and attachments, builds structural context,
calls the BAML `CaptionImage` function through the configured OpenAI-compatible
provider, and writes the final JSONL plus attachment assets when present. Table images receive text from their native row. Paragraph
images use the same paragraph when available. Every image also receives up to
three source units before and after its body position, labeled by structural
relation and distance. Parser defaults bound captioning to four concurrent
requests, 2,048 output tokens, 4,000 context characters, and temperature zero.
Every caption records the exact source references disclosed to the model.

PNG/JPEG assets are sent directly. Legacy EMF/WMF assets require the explicit
LibreOffice renderer and are converted to PNG in an isolated temporary
profile. The same executable converts legacy DOC/XLS attachment members. The
`--renderer-executable` option may be omitted only when the document contains
neither legacy image formats nor legacy Office attachments. Captions are model-derived
enrichment and never modify source hashes or native source records. Any parser
stage failure aborts the output instead of silently producing a partial result.

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
baml_src/                    Versioned BAML definitions
src/legal_rag/parser/        Document parser and enrichment pipeline
src/legal_rag/baml_client/   Generated BAML Python client (Git-ignored)
data/config/                 Corpus selection config
data/raw/                    Raw legal document archive
docs/references/             External data and research links
tests/parser/                Offline parser tests
```
