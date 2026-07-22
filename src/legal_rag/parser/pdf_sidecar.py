"""Render a DOCX PDF sidecar and extract page-aware Docling blocks."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.io import DocumentStream

from legal_rag.parser.source_blocks import SourceBlock, build_source_blocks, write_source_blocks

PDF_SIDECAR_MANIFEST_SCHEMA_VERSION = "1.0.0"
_PDF_EXPORT_FILTER = "pdf:writer_pdf_Export"


class PdfSidecarError(ValueError):
    """Raised when a page-aware PDF sidecar cannot be created safely."""

    @classmethod
    def missing_renderer(cls, path: Path) -> PdfSidecarError:
        """Build an error for a missing renderer executable."""
        return cls(f"Renderer executable does not exist: {path}")

    @classmethod
    def invalid_timeout(cls) -> PdfSidecarError:
        """Build an error for a non-positive renderer timeout."""
        return cls("Renderer timeout must be greater than zero")

    @classmethod
    def renderer_timeout(cls, timeout_seconds: float) -> PdfSidecarError:
        """Build an error for a renderer process timeout."""
        return cls(f"Renderer did not finish within {timeout_seconds} seconds")

    @classmethod
    def renderer_failed(cls, return_code: int, details: str) -> PdfSidecarError:
        """Build an error for a non-zero renderer exit code."""
        return cls(f"Renderer failed with exit code {return_code}: {details}")

    @classmethod
    def missing_renderer_version(cls, path: Path) -> PdfSidecarError:
        """Build an error for a renderer without version output."""
        return cls(f"Renderer did not report its version: {path}")

    @classmethod
    def missing_pdf_output(cls, path: Path) -> PdfSidecarError:
        """Build an error when rendering reports success without a PDF."""
        return cls(f"Renderer did not create the expected PDF: {path}")

    @classmethod
    def empty_pdf_document(cls, path: Path) -> PdfSidecarError:
        """Build an error for a parsed PDF without pages."""
        return cls(f"PDF contains no pages: {path}")

    @classmethod
    def missing_page_provenance(cls, path: Path) -> PdfSidecarError:
        """Build an error for PDF blocks without any page provenance."""
        return cls(f"Docling produced no page provenance for PDF: {path}")

    @classmethod
    def invalid_page_number(cls, page_number: object) -> PdfSidecarError:
        """Build an error for provenance outside the parsed PDF."""
        return cls(f"Docling produced an invalid PDF page number: {page_number}")

    @classmethod
    def pdf_hash_mismatch(cls) -> PdfSidecarError:
        """Build an error when parsing a different PDF than the rendered file."""
        return cls("Rendered PDF hash does not match the parsed PDF hash")


@dataclass(frozen=True, slots=True)
class RenderedPdf:
    """One rendered PDF and the exact renderer provenance used to create it."""

    source_name: str
    source_sha256: str
    pdf_name: str
    pdf_sha256: str
    renderer_executable_name: str
    renderer_version: str


@dataclass(frozen=True, slots=True)
class ParsedPdf:
    """Page-aware Docling blocks extracted from one rendered PDF."""

    pdf_name: str
    pdf_sha256: str
    page_count: int
    blocks: tuple[SourceBlock, ...]
    blocks_with_page_provenance: int
    referenced_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PdfSidecar:
    """Rendered and parsed PDF sidecar with an auditable manifest."""

    rendered: RenderedPdf
    parsed: ParsedPdf

    def manifest(self) -> dict[str, Any]:
        """Return the complete DOCX-to-PDF provenance manifest."""
        if self.rendered.pdf_sha256 != self.parsed.pdf_sha256:
            raise PdfSidecarError.pdf_hash_mismatch()
        return {
            "schema_version": PDF_SIDECAR_MANIFEST_SCHEMA_VERSION,
            "source_docx_name": self.rendered.source_name,
            "source_docx_sha256": self.rendered.source_sha256,
            "pdf_name": self.rendered.pdf_name,
            "pdf_sha256": self.rendered.pdf_sha256,
            "renderer_executable_name": self.rendered.renderer_executable_name,
            "renderer_version": self.rendered.renderer_version,
            "pdf_backend": PyPdfiumDocumentBackend.__name__,
            "ocr_enabled": False,
            "page_count": self.parsed.page_count,
            "pdf_block_count": len(self.parsed.blocks),
            "blocks_with_page_provenance": self.parsed.blocks_with_page_provenance,
            "blocks_without_page_provenance": len(self.parsed.blocks) - self.parsed.blocks_with_page_provenance,
            "referenced_pages": list(self.parsed.referenced_pages),
        }


def render_docx_pdf(
    source_path: Path,
    pdf_output_path: Path,
    *,
    renderer_executable: Path,
    timeout_seconds: float,
) -> RenderedPdf:
    """Render one DOCX to PDF with an isolated LibreOffice profile."""
    if not renderer_executable.is_file():
        raise PdfSidecarError.missing_renderer(renderer_executable)
    if timeout_seconds <= 0:
        raise PdfSidecarError.invalid_timeout()

    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    renderer_version = _renderer_version(renderer_executable, timeout_seconds)
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=pdf_output_path.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        render_directory = temporary_root / "output"
        profile_directory = temporary_root / "profile"
        render_directory.mkdir()
        profile_directory.mkdir()
        command = [
            str(renderer_executable.resolve()),
            "--headless",
            f"-env:UserInstallation={profile_directory.resolve().as_uri()}",
            "--convert-to",
            _PDF_EXPORT_FILTER,
            "--outdir",
            str(render_directory),
            str(source_path.resolve()),
        ]
        completed = _run_renderer(command, renderer_executable.parent, timeout_seconds)
        if completed.returncode != 0:
            details = _process_details(completed)
            raise PdfSidecarError.renderer_failed(completed.returncode, details)

        rendered_path = render_directory / f"{source_path.stem}.pdf"
        if not rendered_path.is_file():
            raise PdfSidecarError.missing_pdf_output(rendered_path)
        rendered_path.replace(pdf_output_path)

    pdf_sha256 = hashlib.sha256(pdf_output_path.read_bytes()).hexdigest()
    return RenderedPdf(
        source_name=source_path.name,
        source_sha256=source_sha256,
        pdf_name=pdf_output_path.name,
        pdf_sha256=pdf_sha256,
        renderer_executable_name=renderer_executable.name,
        renderer_version=renderer_version,
    )


def extract_pdf_blocks(pdf_path: Path) -> ParsedPdf:
    """Extract PDF blocks with page and bounding-box provenance using Docling."""
    pdf_bytes = pdf_path.read_bytes()
    pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    source = DocumentStream(name=pdf_path.name, stream=io.BytesIO(pdf_bytes))
    pipeline_options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options,
            )
        }
    )
    document = converter.convert(source).document
    page_count = len(document.pages)
    if page_count == 0:
        raise PdfSidecarError.empty_pdf_document(pdf_path)

    blocks = build_source_blocks(document, source_name=pdf_path.name, source_sha256=pdf_sha256)
    blocks_with_provenance, referenced_pages = _page_provenance(blocks, page_count)
    if blocks_with_provenance == 0:
        raise PdfSidecarError.missing_page_provenance(pdf_path)
    return ParsedPdf(
        pdf_name=pdf_path.name,
        pdf_sha256=pdf_sha256,
        page_count=page_count,
        blocks=tuple(blocks),
        blocks_with_page_provenance=blocks_with_provenance,
        referenced_pages=referenced_pages,
    )


def write_pdf_sidecar(block_output_path: Path, manifest_path: Path, sidecar: PdfSidecar) -> None:
    """Write PDF source blocks and their DOCX-to-PDF manifest."""
    write_source_blocks(block_output_path, list(sidecar.parsed.blocks))
    _write_json(manifest_path, sidecar.manifest())


def _renderer_version(renderer_executable: Path, timeout_seconds: float) -> str:
    """Return the renderer's self-reported version."""
    completed = _run_renderer(
        [str(renderer_executable.resolve()), "--version"],
        renderer_executable.parent,
        timeout_seconds,
    )
    if completed.returncode != 0:
        details = _process_details(completed)
        raise PdfSidecarError.renderer_failed(completed.returncode, details)
    version = completed.stdout.strip() or completed.stderr.strip()
    if not version:
        raise PdfSidecarError.missing_renderer_version(renderer_executable)
    return version


def _run_renderer(
    command: list[str], working_directory: Path, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run the renderer without a shell and with an explicit timeout."""
    try:
        # The explicit executable path is validated as a file; shell=False prevents command injection.
        return subprocess.run(  # noqa: S603
            command,
            cwd=working_directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise PdfSidecarError.renderer_timeout(timeout_seconds) from error


def _process_details(completed: subprocess.CompletedProcess[str]) -> str:
    """Return bounded renderer diagnostics without document content."""
    return completed.stderr.strip() or completed.stdout.strip() or "no diagnostics"


def _page_provenance(blocks: list[SourceBlock], page_count: int) -> tuple[int, tuple[int, ...]]:
    """Validate and summarize Docling page provenance."""
    blocks_with_provenance = 0
    referenced_pages: set[int] = set()
    for block in blocks:
        provenance = block.payload.get("prov", [])
        if provenance:
            blocks_with_provenance += 1
        for item in provenance:
            page_number = item.get("page_no")
            if not isinstance(page_number, int) or not 1 <= page_number <= page_count:
                raise PdfSidecarError.invalid_page_number(page_number)
            referenced_pages.add(page_number)
    return blocks_with_provenance, tuple(sorted(referenced_pages))


def _write_json(output_path: Path, value: dict[str, Any]) -> None:
    """Atomically write one UTF-8 JSON document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
