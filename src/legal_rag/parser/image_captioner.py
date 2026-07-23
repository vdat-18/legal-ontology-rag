"""Caption parser images with bounded structural context through BAML."""

from __future__ import annotations

import base64
import http.client
import json
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from baml_py import ClientRegistry, HTTPRequest, Image
from baml_py.errors import BamlError
from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry, StructuralContextRelation
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
    def model_call_failed(cls) -> ImageCaptionError:
        """Build a sanitized error for a failed BAML model call."""
        return cls("BAML image-caption call failed; inspect service availability without logging document content")

    @classmethod
    def unsupported_media(cls, mime_type: str) -> ImageCaptionError:
        """Build an error when legacy media requires an explicit rasterizer."""
        return cls(f"Image media requires --renderer-executable before captioning: {mime_type}")

    @classmethod
    def response_too_large(cls, max_response_bytes: int) -> ImageCaptionError:
        """Build an error for a response exceeding its configured byte limit."""
        return cls(f"LLM provider image-caption response exceeded {max_response_bytes} bytes")


class ImageCaptionRateLimitError(ImageCaptionError):
    """Raised when the caption provider rejects a request due to quota limits."""

    @classmethod
    def provider_rate_limited(cls) -> ImageCaptionRateLimitError:
        """Build a sanitized error for an HTTP 429 response."""
        return cls("BAML image-caption call was rate limited by the provider")


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
        self._registry = ClientRegistry()
        self._registry.add_llm_client(
            "RuntimeLLMProvider",
            "openai-generic",
            {
                "base_url": settings.llm_base_url,
                "api_key": settings.llm_api_key,
                "model": settings.llm_model,
                "max_tokens": settings.image_caption_max_output_tokens,
                "temperature": settings.image_caption_temperature,
                "supports_streaming": False,
                "http": {"request_timeout_ms": int(settings.llm_timeout_seconds * 1000)},
            },
        )
        self._registry.set_primary("RuntimeLLMProvider")
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
        try:
            request = b.request.CaptionImage(
                entries,
                image,
                baml_options={"client_registry": self._registry},
            )
            raw_output = _execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
            result = b.parse.CaptionImage(raw_output, baml_options={"client_registry": self._registry})
        except BamlError:
            # BAML exceptions may contain prompts and raw model output, so do not preserve them in logs.
            raise ImageCaptionError.model_call_failed() from None
        return CaptionModelResponse(result.description_vi, result.visible_text)


def _execute_baml_request(request: HTTPRequest, timeout_seconds: float, max_response_bytes: int) -> str:
    """Execute one non-streaming OpenAI-compatible request and return assistant content."""
    parsed_url = urlsplit(request.url)
    connection_type = http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed_url.hostname, parsed_url.port, timeout=timeout_seconds)
    request_path = parsed_url.path or "/"
    if parsed_url.query:
        request_path = f"{request_path}?{parsed_url.query}"
    headers = {name: value for name, value in request.headers.items() if name.lower() != "baml-original-url"}
    request_body = request.body.json()
    request_body["stream"] = False
    request_body["response_format"] = {"type": "json_object"}
    try:
        connection.request(
            request.method,
            request_path,
            body=json.dumps(request_body).encode("utf-8"),
            headers=headers,
        )
        response = connection.getresponse()
        if response.status == HTTPStatus.TOO_MANY_REQUESTS:
            raise ImageCaptionRateLimitError.provider_rate_limited()
        if response.status != HTTPStatus.OK:
            raise ImageCaptionError.model_call_failed()
        response_bytes = response.read(max_response_bytes + 1)
    finally:
        connection.close()
    if len(response_bytes) > max_response_bytes:
        raise ImageCaptionError.response_too_large(max_response_bytes)
    return _openai_response_content(response_bytes)


def _openai_response_content(response_bytes: bytes) -> str:
    """Return assistant content from one OpenAI-compatible JSON response."""
    response = json.loads(response_bytes)
    provider_error = response.get("error")
    if provider_error is not None:
        if isinstance(provider_error, dict) and provider_error.get("code") == HTTPStatus.TOO_MANY_REQUESTS:
            raise ImageCaptionRateLimitError.provider_rate_limited()
        raise ImageCaptionError.model_call_failed()
    return response["choices"][0]["message"]["content"]


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
            visible_text=response.visible_text,
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
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output_path = output_directory / f"{input_path.stem}.png"
        return output_path.read_bytes()
