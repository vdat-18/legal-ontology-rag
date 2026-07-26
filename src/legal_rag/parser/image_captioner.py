"""Caption parser images with bounded structural context through BAML."""

from __future__ import annotations

import base64
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from baml_py import Image
from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry, StructuralContextRelation
from legal_rag.baml_transport import BamlRateLimitError, BamlTransportError, build_client_registry, execute_baml_request
from legal_rag.parser.document_source import DocumentSource
from legal_rag.parser.image_context import ImageContext, build_image_context
from legal_rag.parser.images import DocxImage, DocxImageExtraction
from legal_rag.parser.models import IMAGE_CAPTION_SCHEMA_VERSION, CaptionModelResponse, ImageCaption
from legal_rag.settings import Settings

_SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_PNG_MIME_TYPE = "image/png"


class ImageCaptionError(ValueError):
    """Raised when an image cannot produce a valid caption record."""

    @classmethod
    def unsupported_media(cls, mime_type: str) -> ImageCaptionError:
        """Build an error when legacy media requires an explicit rasterizer."""
        return cls(f"Image media requires --renderer-executable before captioning: {mime_type}")


class ImageCaptionRateLimitError(ImageCaptionError):
    """Raised when the caption provider rejects a request due to quota limits."""

    @classmethod
    def provider_rate_limited(cls, details: str | None = None) -> ImageCaptionRateLimitError:
        """Build a retryable error for any CaptionClient implementation, not only BAML."""
        message = "Image-caption request was rate limited by the provider"
        return cls(f"{message}: {details}" if details else message)


class CaptionClient(Protocol):
    """Boundary implemented by an image-capable language-model client."""

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Return one typed caption response."""


class _RateLimitGate:
    """Pause all caption workers after any provider rate-limit response."""

    def __init__(self, sleep: Callable[[float], None]) -> None:
        """Initialize an open gate backed by the injected sleeper."""
        self._available = threading.Event()
        self._available.set()
        self._pause_lock = threading.Lock()
        self._sleep = sleep

    def wait(self) -> None:
        """Block until caption requests may be sent again."""
        self._available.wait()

    def pause(self, delay_seconds: float) -> None:
        """Close the gate once and reopen it after the shared delay."""
        self._available.clear()
        with self._pause_lock:
            if self._available.is_set():
                return
            self._sleep(delay_seconds)
            self._available.set()


class BamlCaptionClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=settings.image_caption_max_output_tokens,
            temperature=settings.image_caption_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = settings.image_caption_max_response_bytes

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
    ) -> CaptionModelResponse:
        """Run one bounded BAML call and return its generated typed result."""
        entries = [
            StructuralContextEntry(
                source_ref=entry.source_ref,
                relation=StructuralContextRelation(entry.relation.value),
                distance=entry.distance,
                text=entry.text,
            )
            for entry in context.entries
        ]
        image = Image.from_base64(mime_type, base64.b64encode(image_bytes).decode("ascii"))
        request = b.request.CaptionImage(
            entries,
            image,
            baml_options={"client_registry": self._registry},
        )
        try:
            raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        except BamlRateLimitError as error:
            raise ImageCaptionRateLimitError(str(error)) from error
        except BamlTransportError as error:
            raise ImageCaptionError(str(error)) from error
        result = b.parse.CaptionImage(raw_output, baml_options={"client_registry": self._registry})
        return CaptionModelResponse(result.description_vi)


def caption_images(
    extraction: DocxImageExtraction,
    source: DocumentSource,
    settings: Settings,
    client: CaptionClient,
    *,
    renderer_executable: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ImageCaption, ...]:
    """Caption extracted images concurrently while preserving source order."""
    rate_limit_gate = _RateLimitGate(sleep)

    def caption_one(image: DocxImage) -> ImageCaption:
        context = build_image_context(
            image,
            source,
            max_characters=settings.image_caption_max_context_characters,
            surrounding_units=settings.image_caption_surrounding_context_units,
        )
        caption_bytes, caption_mime_type = _prepare_caption_media(
            image,
            renderer_executable=renderer_executable,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        for attempt in range(settings.image_caption_max_rate_limit_retries + 1):
            rate_limit_gate.wait()
            try:
                response = client.caption(image_bytes=caption_bytes, mime_type=caption_mime_type, context=context)
                break
            except ImageCaptionRateLimitError:
                if attempt == settings.image_caption_max_rate_limit_retries:
                    raise
                logger.warning(
                    "Image caption {}/{} was rate limited; retrying in {} seconds",
                    image.sequence_index + 1,
                    len(extraction.images),
                    settings.image_caption_rate_limit_retry_delay_seconds,
                )
                rate_limit_gate.pause(settings.image_caption_rate_limit_retry_delay_seconds)
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
            requested_model=settings.llm_model,
            description_vi=response.description_vi,
        )

    with ThreadPoolExecutor(max_workers=settings.image_caption_max_concurrency) as executor:
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
        subprocess.run(  # noqa: S603
            command,
            check=True,
            timeout=timeout_seconds,
        )
        output_path = output_directory / f"{input_path.stem}.png"
        return output_path.read_bytes()
