"""Parse supported documents stored inside extracted attachments."""

from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import olefile

from legal_rag.parser.attachments import DocxAttachment
from legal_rag.parser.models import AttachmentChildDocument, ParsedDocument
from legal_rag.parser.spreadsheets import extract_xlsx_rows
from legal_rag.settings import Settings

FORMAT_DOC = "doc"
FORMAT_DOCX = "docx"
FORMAT_XLS = "xls"
FORMAT_XLSX = "xlsx"
FORMAT_ZIP = "zip"

_DOCX_DOCUMENT_PART = "word/document.xml"
_XLSX_WORKBOOK_PART = "xl/workbook.xml"
_OLE_WORD_STREAM = ("WordDocument",)
_OLE_WORKBOOK_STREAMS = frozenset({("Workbook",), ("Book",)})


class AttachmentDocumentIngestionError(ValueError):
    """Raised when attachment child documents cannot be ingested safely."""


@dataclass(frozen=True, slots=True)
class _ChildSource:
    """One supported attachment payload or archive member awaiting parsing."""

    attachment: DocxAttachment
    member_index: int
    member_path: str
    source_bytes: bytes
    source_format: str


def ingest_attachment_documents(
    attachments: tuple[DocxAttachment, ...],
    settings: Settings,
    parse_docx: Callable[[Path], ParsedDocument],
    *,
    renderer_executable: Path | None,
) -> tuple[AttachmentChildDocument, ...]:
    """Parse supported direct payloads and ZIP members in attachment order."""
    sources: list[_ChildSource] = []
    for attachment in attachments:
        payload_format = _detect_format(attachment.payload_bytes)
        members = (
            _read_archive_members(attachment, settings)
            if payload_format == FORMAT_ZIP
            else ((0, attachment.filename, attachment.payload_bytes, payload_format),)
        )
        for member_index, member_path, member_bytes, member_format in members:
            if member_format not in {FORMAT_DOC, FORMAT_DOCX, FORMAT_XLS, FORMAT_XLSX}:
                continue
            sources.append(
                _ChildSource(
                    attachment=attachment,
                    member_index=member_index,
                    member_path=member_path,
                    source_bytes=member_bytes,
                    source_format=member_format,
                )
            )

    converted_sources = _convert_legacy_documents(sources, settings, renderer_executable)
    child_documents: list[AttachmentChildDocument] = []
    for source_index, source in enumerate(sources):
        parsed_format = source.source_format
        converted_from_format: str | None = None
        source_bytes = source.source_bytes
        if source.source_format in {FORMAT_DOC, FORMAT_XLS}:
            parsed_format = FORMAT_DOCX if source.source_format == FORMAT_DOC else FORMAT_XLSX
            converted_from_format = source.source_format
            source_bytes = converted_sources[source_index]
        child_documents.append(
            _parse_child_document(
                source,
                source_bytes,
                parsed_format,
                converted_from_format,
                parse_docx,
            )
        )
    return tuple(child_documents)


def _convert_legacy_documents(
    sources: list[_ChildSource],
    settings: Settings,
    renderer_executable: Path | None,
) -> dict[int, bytes]:
    """Convert legacy Word and Excel sources in at most two LibreOffice batches."""
    legacy_sources = [
        (source_index, source)
        for source_index, source in enumerate(sources)
        if source.source_format in {FORMAT_DOC, FORMAT_XLS}
    ]
    if not legacy_sources:
        return {}
    if renderer_executable is None:
        formats = ", ".join(sorted({source.source_format.upper() for _, source in legacy_sources}))
        message = f"Legacy {formats} attachments require --renderer-executable"
        raise AttachmentDocumentIngestionError(message)

    renderer = renderer_executable.resolve()
    converted: dict[int, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="legal-rag-office-") as temporary_directory:
        workspace = Path(temporary_directory)
        output_directory = workspace / "output"
        profile_directory = workspace / "profile"
        output_directory.mkdir()
        profile_directory.mkdir()
        for source_format, target_format in ((FORMAT_DOC, FORMAT_DOCX), (FORMAT_XLS, FORMAT_XLSX)):
            batch = [(index, source) for index, source in legacy_sources if source.source_format == source_format]
            if not batch:
                continue
            input_paths: list[Path] = []
            for source_index, source in batch:
                input_path = workspace / f"source-{source_index}.{source_format}"
                input_path.write_bytes(source.source_bytes)
                input_paths.append(input_path)
            _run_office_conversion(
                renderer,
                profile_directory,
                output_directory,
                input_paths,
                source_format,
                target_format,
                settings.office_conversion_timeout_seconds,
            )
            for (source_index, _), input_path in zip(batch, input_paths, strict=True):
                output_path = output_directory / f"{input_path.stem}.{target_format}"
                if not output_path.is_file():
                    message = (
                        f"LibreOffice did not produce {target_format.upper()} output "
                        f"for a legacy {source_format.upper()} attachment"
                    )
                    raise AttachmentDocumentIngestionError(message)
                converted[source_index] = output_path.read_bytes()
    return converted


def _run_office_conversion(
    renderer: Path,
    profile_directory: Path,
    output_directory: Path,
    input_paths: list[Path],
    source_format: str,
    target_format: str,
    timeout_seconds: float,
) -> None:
    """Run one bounded LibreOffice conversion batch."""
    command = [
        str(renderer),
        "--headless",
        f"-env:UserInstallation={profile_directory.as_uri()}",
        "--convert-to",
        target_format,
        "--outdir",
        str(output_directory),
        *(str(path) for path in input_paths),
    ]
    try:
        subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        message = f"LibreOffice conversion timed out for legacy {source_format.upper()} attachments"
        raise AttachmentDocumentIngestionError(message) from error
    except subprocess.CalledProcessError as error:
        message = f"LibreOffice could not convert legacy {source_format.upper()} attachments"
        raise AttachmentDocumentIngestionError(message) from error


def _parse_child_document(
    source: _ChildSource,
    source_bytes: bytes,
    parsed_format: str,
    converted_from_format: str | None,
    parse_docx: Callable[[Path], ParsedDocument],
) -> AttachmentChildDocument:
    """Parse one prepared child document."""
    if parsed_format == FORMAT_DOCX:
        with tempfile.TemporaryDirectory(prefix="legal-rag-child-docx-") as temporary_directory:
            input_path = Path(temporary_directory) / "source.docx"
            input_path.write_bytes(source_bytes)
            document = parse_docx(input_path)
        native_records = tuple(unit.to_dict() for unit in document.units)
        nested_attachments = tuple(
            nested_attachment for unit in document.units for nested_attachment in unit.attachments
        )
    else:
        native_records = tuple(row.to_dict() for row in extract_xlsx_rows(source_bytes, source.member_path))
        nested_attachments = ()

    return AttachmentChildDocument(
        parent_attachment_id=source.attachment.attachment_id,
        member_index=source.member_index,
        member_path=source.member_path,
        member_sha256=hashlib.sha256(source.source_bytes).hexdigest(),
        source_format=parsed_format,
        converted_from_format=converted_from_format,
        native_records=native_records,
        attachments=nested_attachments,
    )


def _read_archive_members(
    attachment: DocxAttachment,
    settings: Settings,
) -> tuple[tuple[int, str, bytes, str | None], ...]:
    """Read bounded, non-encrypted ZIP members without extracting filesystem paths."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(attachment.payload_bytes))
    except zipfile.BadZipFile as error:
        message = f"Attachment is not a readable ZIP archive: {attachment.filename}"
        raise AttachmentDocumentIngestionError(message) from error
    with archive:
        files = [member for member in archive.infolist() if not member.is_dir()]
        if len(files) > settings.attachment_archive_max_members:
            message = f"Attachment ZIP exceeds the member limit: {attachment.filename}"
            raise AttachmentDocumentIngestionError(message)
        if sum(member.file_size for member in files) > settings.attachment_archive_max_total_bytes:
            message = f"Attachment ZIP exceeds the total uncompressed-byte limit: {attachment.filename}"
            raise AttachmentDocumentIngestionError(message)
        members: list[tuple[int, str, bytes, str | None]] = []
        actual_total = 0
        for member_index, member in enumerate(files):
            member_path = _safe_member_path(member.filename, attachment.filename)
            if member.flag_bits & 1:
                message = f"Encrypted ZIP members are not supported: {attachment.filename}/{member_path}"
                raise AttachmentDocumentIngestionError(message)
            if member.file_size > settings.attachment_archive_max_member_bytes:
                message = f"Attachment ZIP member exceeds the byte limit: {attachment.filename}/{member_path}"
                raise AttachmentDocumentIngestionError(message)
            with archive.open(member) as member_file:
                member_bytes = member_file.read(settings.attachment_archive_max_member_bytes + 1)
            if len(member_bytes) > settings.attachment_archive_max_member_bytes:
                message = f"Attachment ZIP member exceeds the byte limit: {attachment.filename}/{member_path}"
                raise AttachmentDocumentIngestionError(message)
            actual_total += len(member_bytes)
            if actual_total > settings.attachment_archive_max_total_bytes:
                message = f"Attachment ZIP exceeds the total uncompressed-byte limit: {attachment.filename}"
                raise AttachmentDocumentIngestionError(message)
            members.append((member_index, member_path, member_bytes, _detect_format(member_bytes)))
    return tuple(members)


def _detect_format(source_bytes: bytes) -> str | None:
    """Identify supported Office containers from their package structure."""
    buffer = io.BytesIO(source_bytes)
    if olefile.isOleFile(buffer):
        buffer.seek(0)
        with olefile.OleFileIO(buffer) as compound_file:
            streams = {tuple(path) for path in compound_file.listdir(streams=True, storages=False)}
        if _OLE_WORD_STREAM in streams:
            return FORMAT_DOC
        if streams & _OLE_WORKBOOK_STREAMS:
            return FORMAT_XLS
        return None
    buffer.seek(0)
    if zipfile.is_zipfile(buffer):
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            names = set(archive.namelist())
        if _DOCX_DOCUMENT_PART in names:
            return FORMAT_DOCX
        if _XLSX_WORKBOOK_PART in names:
            return FORMAT_XLSX
        return FORMAT_ZIP
    return None


def _safe_member_path(member_name: str, attachment_name: str) -> str:
    """Normalize one archive member path and reject traversal syntax."""
    normalized_name = member_name.replace("\\", "/")
    path = PurePosixPath(normalized_name)
    windows_path = PureWindowsPath(member_name)
    if path.is_absolute() or windows_path.drive or not path.parts or ".." in path.parts:
        message = f"Attachment ZIP contains an unsafe member path: {attachment_name}"
        raise AttachmentDocumentIngestionError(message)
    return path.as_posix()
