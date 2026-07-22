"""Tests for rendered, page-aware PDF sidecars."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat

from legal_rag.parser.pdf_sidecar import (
    ParsedPdf,
    PdfSidecar,
    RenderedPdf,
    extract_pdf_blocks,
    render_docx_pdf,
    write_pdf_sidecar,
)
from legal_rag.parser.source_blocks import SOURCE_BLOCK_SCHEMA_VERSION, SourceBlock


def _block(*, sequence_index: int, page_number: int | None) -> SourceBlock:
    """Build one source block with optional PDF provenance."""
    payload = {"text": "source text"}
    if page_number is not None:
        payload["prov"] = [{"page_no": page_number, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]
    return SourceBlock(
        schema_version=SOURCE_BLOCK_SCHEMA_VERSION,
        block_id=f"block-{sequence_index}",
        source_name="source.pdf",
        source_sha256="pdf-hash",
        parser_name="docling",
        parser_version="test",
        parser_core_version="test",
        docling_ref=f"#/texts/{sequence_index}",
        docling_item_type="TextItem",
        docling_label="text",
        sequence_index=sequence_index,
        depth=1,
        payload_sha256=f"payload-{sequence_index}",
        payload=payload,
    )


def test_render_docx_pdf_uses_explicit_renderer_and_isolated_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rendering uses structured arguments and records exact provenance."""
    source_path = tmp_path / "source.docx"
    renderer_path = tmp_path / "soffice.com"
    pdf_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"docx bytes")
    renderer_path.write_bytes(b"renderer")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "LibreOffice test", "")
        output_directory = Path(command[command.index("--outdir") + 1])
        output_directory.joinpath("source.pdf").write_bytes(b"pdf bytes")
        return subprocess.CompletedProcess(command, 0, "converted", "")

    monkeypatch.setattr("legal_rag.parser.pdf_sidecar.subprocess.run", fake_run)

    rendered = render_docx_pdf(
        source_path,
        pdf_path,
        renderer_executable=renderer_path,
        timeout_seconds=30,
    )

    assert pdf_path.read_bytes() == b"pdf bytes"
    assert rendered.renderer_version == "LibreOffice test"
    assert commands[0] == [str(renderer_path.resolve()), "--version"]
    assert "--headless" in commands[1]
    assert any(argument.startswith("-env:UserInstallation=file:") for argument in commands[1])


def test_write_pdf_sidecar_preserves_page_provenance(tmp_path: Path) -> None:
    """PDF blocks and manifest retain page coverage and renderer provenance."""
    blocks = (_block(sequence_index=0, page_number=1), _block(sequence_index=1, page_number=None))
    rendered = RenderedPdf(
        source_name="source.docx",
        source_sha256="docx-hash",
        pdf_name="source.pdf",
        pdf_sha256="pdf-hash",
        renderer_executable_name="soffice.com",
        renderer_version="LibreOffice test",
    )
    parsed = ParsedPdf(
        pdf_name="source.pdf",
        pdf_sha256="pdf-hash",
        page_count=1,
        blocks=blocks,
        blocks_with_page_provenance=1,
        referenced_pages=(1,),
    )
    output_path = tmp_path / "pdf-blocks.jsonl"
    manifest_path = tmp_path / "pdf.manifest.json"

    write_pdf_sidecar(output_path, manifest_path, PdfSidecar(rendered=rendered, parsed=parsed))

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert records == [block.to_dict() for block in blocks]
    assert manifest["page_count"] == 1
    assert manifest["blocks_with_page_provenance"] == 1
    assert manifest["referenced_pages"] == [1]


def test_extract_pdf_blocks_uses_pdfium_without_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PDF extraction configures the Unicode-safe backend and disables OCR."""
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf bytes")
    captured: dict[str, object] = {}

    class FakeConverter:
        """Capture Docling PDF format options."""

        def __init__(self, *, format_options):
            captured["format_options"] = format_options

        def convert(self, source):
            captured["source_name"] = source.name
            return type("Result", (), {"document": type("Document", (), {"pages": {1: object()}})()})()

    monkeypatch.setattr("legal_rag.parser.pdf_sidecar.DocumentConverter", FakeConverter)
    monkeypatch.setattr(
        "legal_rag.parser.pdf_sidecar.build_source_blocks",
        lambda *args, **kwargs: [_block(sequence_index=0, page_number=1)],
    )

    parsed = extract_pdf_blocks(pdf_path)

    option = captured["format_options"][InputFormat.PDF]  # type: ignore[index]
    assert option.backend is PyPdfiumDocumentBackend
    assert option.pipeline_options.do_ocr is False
    assert parsed.page_count == 1
