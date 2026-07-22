"""Caption parser images with bounded structural context through BAML."""

from __future__ import annotations

import base64
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from baml_py import ClientRegistry, Image
from baml_py.errors import BamlError
from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry
from legal_rag.parser.config import ParserSettings
from legal_rag.parser.document_source import DocumentSource
from legal_rag.parser.image_context import ImageContext, build_image_context
from legal_rag.parser.images import DocxImage, DocxImageExtraction
from legal_rag.parser.models import IMAGE_CAPTION_SCHEMA_VERSION, CaptionModelResponse, ImageCaption

_SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_PNG_MIME_TYPE = "image/png"


class ImageCaptionError(ValueError):
    """Raised when an image cannot produce a valid caption record."""

    @classmethod
    def model_call_failed(cls) -> ImageCaptionError:
        """Build a sanitized error for a failed BAML model call."""
        return cls("BAML image-caption call failed; inspect service availability without logging document content")

    @classmethod
    def unsupported_media(cls, mime_type: str) -> ImageCaptionError:
        """Build an error when legacy media requires an explicit rasterizer."""
        return cls(f"Image media requires --renderer-executable before captioning: {mime_type}")

    @classmethod
    def missing_renderer(cls, path: Path) -> ImageCaptionError:
        """Build an error for an absent rasterizer executable."""
        return cls(f"Image rasterizer executable does not exist: {path}")

    @classmethod
    def renderer_timeout(cls, timeout_seconds: float) -> ImageCaptionError:
        """Build an error for a timed-out rasterizer process."""
        return cls(f"Image rasterizer exceeded its timeout of {timeout_seconds} seconds")

    @classmethod
    def renderer_failed(cls, return_code: int) -> ImageCaptionError:
        """Build an error for a failed rasterizer process."""
        return cls(f"Image rasterizer failed with exit code {return_code}")

    @classmethod
    def missing_rasterized_output(cls, asset_name: str) -> ImageCaptionError:
        """Build an error when the renderer reports success without PNG output."""
        return cls(f"Image rasterizer did not create PNG output for asset: {asset_name}")


class CaptionClient(Protocol):
    """Boundary implemented by an image-capable language-model client."""

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Return one validated caption response."""


class BamlCaptionClient:
    """Typed BAML client backed by 9Router's OpenAI-compatible API."""

    def __init__(self, settings: ParserSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = ClientRegistry()
        self._registry.add_llm_client(
            "RuntimeRouter",
            "openai-generic",
            {
                "base_url": settings.base_url,
                "api_key": settings.api_key,
                "model": settings.model,
                "max_tokens": settings.max_output_tokens,
                "temperature": settings.temperature,
                "http": {"request_timeout_ms": int(settings.timeout_seconds * 1000)},
            },
        )
        self._registry.set_primary("RuntimeRouter")

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Stream one BAML call and return its generated typed result."""
        entries = [StructuralContextEntry(source_ref=entry.source_ref, text=entry.text) for entry in context.entries]
        image = Image.from_base64(mime_type, base64.b64encode(image_bytes).decode("ascii"))
        try:
            result = b.stream.CaptionImage(
                entries,
                image,
                baml_options={"client_registry": self._registry},
            ).get_final_response()
        except BamlError:
            # BAML exceptions may contain prompts and raw model output, so do not preserve them in logs.
            raise ImageCaptionError.model_call_failed() from None
        return CaptionModelResponse(result.description_vi, result.visible_text)


def caption_images(
    extraction: DocxImageExtraction,
    source: DocumentSource,
    settings: ParserSettings,
    client: CaptionClient,
    *,
    renderer_executable: Path | None = None,
) -> tuple[ImageCaption, ...]:
    """Caption extracted images concurrently while preserving source order."""

    def caption_one(image: DocxImage) -> ImageCaption:
        context = build_image_context(image, source, max_characters=settings.max_context_characters)
        caption_bytes, caption_mime_type = _prepare_caption_media(
            image,
            renderer_executable=renderer_executable,
            timeout_seconds=settings.timeout_seconds,
        )
        response = client.caption(image_bytes=caption_bytes, mime_type=caption_mime_type, context=context)
        return ImageCaption(
            schema_version=IMAGE_CAPTION_SCHEMA_VERSION,
            image_id=image.image_id,
            source_name=image.source_name,
            source_sha256=image.source_sha256,
            body_index=image.body_index,
            source_ref=image.source_ref,
            parent_source_ref=image.parent_source_ref,
            media_sha256=image.media_sha256,
            source_mime_type=image.mime_type,
            caption_mime_type=caption_mime_type,
            context_source_refs=context.source_refs,
            requested_model=settings.model,
            description_vi=response.description_vi,
            visible_text=response.visible_text,
        )

    with ThreadPoolExecutor(max_workers=settings.max_concurrency) as executor:
        return tuple(executor.map(caption_one, extraction.images))


def _prepare_caption_media(
    image: DocxImage,
    *,
    renderer_executable: Path | None,
    timeout_seconds: float,
) -> tuple[bytes, str]:
    """Return API-supported image bytes, rasterizing legacy formats when required."""
    if image.mime_type in _SUPPORTED_IMAGE_MIME_TYPES:
        return image.media_bytes, image.mime_type
    if renderer_executable is None:
        raise ImageCaptionError.unsupported_media(image.mime_type)
    return _rasterize_to_png(image, renderer_executable, timeout_seconds), _PNG_MIME_TYPE


def _rasterize_to_png(image: DocxImage, renderer_executable: Path, timeout_seconds: float) -> bytes:
    """Convert one legacy office image to PNG with an isolated LibreOffice profile."""
    renderer = renderer_executable.resolve()
    if not renderer.is_file():
        raise ImageCaptionError.missing_renderer(renderer)
    with tempfile.TemporaryDirectory(prefix="legal-rag-image-") as temporary_directory:
        workspace = Path(temporary_directory)
        input_path = workspace / image.asset_name
        output_directory = workspace / "output"
        profile_directory = workspace / "profile"
        input_path.write_bytes(image.media_bytes)
        output_directory.mkdir()
        profile_directory.mkdir()
        command = [
            str(renderer),
            "--headless",
            f"-env:UserInstallation={profile_directory.as_uri()}",
            "--convert-to",
            "png",
            "--outdir",
            str(output_directory),
            str(input_path),
        ]
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ImageCaptionError.renderer_timeout(timeout_seconds) from error
        if completed.returncode != 0:
            raise ImageCaptionError.renderer_failed(completed.returncode)
        output_path = output_directory / f"{input_path.stem}.png"
        if not output_path.is_file():
            raise ImageCaptionError.missing_rasterized_output(image.asset_name)
        return output_path.read_bytes()
