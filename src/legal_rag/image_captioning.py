"""Caption DOCX images with bounded structural context through 9Router."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from baml_py import ClientRegistry, Image
from baml_py.errors import BamlError
from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import StructuralContextEntry
from legal_rag.document_source import DocumentSource, extract_document_source
from legal_rag.docx_images import DocxImage, DocxImageExtraction, extract_docx_images
from legal_rag.image_context import ImageContext, build_image_context

IMAGE_CAPTION_SCHEMA_VERSION = "1.1.0"

_BASE_URL_ENV = "LLM_BASE_URL"
_API_KEY_ENV = "LLM_API_KEY"
_MODEL_ENV = "LLM_MODEL"
_TIMEOUT_ENV = "LLM_TIMEOUT_SECONDS"
_CONCURRENCY_ENV = "IMAGE_CAPTION_MAX_CONCURRENCY"
_MAX_OUTPUT_TOKENS_ENV = "IMAGE_CAPTION_MAX_OUTPUT_TOKENS"
_MAX_CONTEXT_CHARACTERS_ENV = "IMAGE_CAPTION_MAX_CONTEXT_CHARACTERS"
_TEMPERATURE_ENV = "IMAGE_CAPTION_TEMPERATURE"

_SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_PNG_MIME_TYPE = "image/png"


class ImageCaptionConfigurationError(ValueError):
    """Raised when image-caption configuration is missing or invalid."""

    @classmethod
    def missing(cls, variable_name: str) -> ImageCaptionConfigurationError:
        """Build an error for a required environment variable."""
        return cls(f"Required image-caption environment variable is missing: {variable_name}")

    @classmethod
    def invalid_url(cls, variable_name: str) -> ImageCaptionConfigurationError:
        """Build an error for an unusable service URL."""
        return cls(f"Image-caption environment variable must be an HTTP(S) URL: {variable_name}")

    @classmethod
    def invalid_number(cls, variable_name: str) -> ImageCaptionConfigurationError:
        """Build an error for a numeric environment variable."""
        return cls(f"Image-caption environment variable must be a valid number: {variable_name}")

    @classmethod
    def non_positive(cls, variable_name: str) -> ImageCaptionConfigurationError:
        """Build an error for a non-positive limit."""
        return cls(f"Image-caption environment variable must be positive: {variable_name}")

    @classmethod
    def invalid_temperature(cls, variable_name: str) -> ImageCaptionConfigurationError:
        """Build an error for a temperature outside the API-supported range."""
        return cls(f"Image-caption environment variable is outside the supported range: {variable_name}")


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


@dataclass(frozen=True, slots=True)
class ImageCaptionSettings:
    """Typed service and bounded-execution settings loaded from environment variables."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float
    max_concurrency: int
    max_output_tokens: int
    max_context_characters: int
    temperature: float

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ImageCaptionSettings:
        """Load and validate required image-caption settings."""
        values = os.environ if environment is None else environment
        base_url = _required_value(values, _BASE_URL_ENV)
        parsed_url = urlsplit(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ImageCaptionConfigurationError.invalid_url(_BASE_URL_ENV)
        temperature = _number(values, _TEMPERATURE_ENV)
        if not 0 <= temperature <= 2:
            raise ImageCaptionConfigurationError.invalid_temperature(_TEMPERATURE_ENV)
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=_required_value(values, _API_KEY_ENV),
            model=_required_value(values, _MODEL_ENV),
            timeout_seconds=_positive_float(values, _TIMEOUT_ENV),
            max_concurrency=_positive_int(values, _CONCURRENCY_ENV),
            max_output_tokens=_positive_int(values, _MAX_OUTPUT_TOKENS_ENV),
            max_context_characters=_positive_int(values, _MAX_CONTEXT_CHARACTERS_ENV),
            temperature=temperature,
        )


@dataclass(frozen=True, slots=True)
class CaptionModelResponse:
    """Validated semantic fields and model identity from one API response."""

    description_vi: str
    visible_text: str | None


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


@dataclass(frozen=True, slots=True)
class ImageCaption:
    """One caption tied to immutable image and context provenance."""

    schema_version: str
    image_id: str
    source_name: str
    source_sha256: str
    source_ref: str
    media_sha256: str
    source_mime_type: str
    caption_mime_type: str
    context_source_refs: tuple[str, ...]
    requested_model: str
    description_vi: str
    visible_text: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable caption record."""
        return {
            "schema_version": self.schema_version,
            "image_id": self.image_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_ref": self.source_ref,
            "media_sha256": self.media_sha256,
            "source_mime_type": self.source_mime_type,
            "caption_mime_type": self.caption_mime_type,
            "context_source_refs": list(self.context_source_refs),
            "requested_model": self.requested_model,
            "description_vi": self.description_vi,
            "visible_text": self.visible_text,
        }


class BamlCaptionClient:
    """Typed BAML client backed by 9Router's OpenAI-compatible API."""

    def __init__(self, settings: ImageCaptionSettings) -> None:
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


def caption_docx_document(
    input_path: Path,
    settings: ImageCaptionSettings,
    client: CaptionClient,
    *,
    renderer_executable: Path | None = None,
) -> tuple[ImageCaption, ...]:
    """Extract and caption one DOCX without persisted intermediate image files."""
    extraction = extract_docx_images(input_path)
    if not extraction.images:
        return ()
    source = extract_document_source(input_path)
    return caption_docx_images(
        extraction,
        source,
        settings,
        client,
        renderer_executable=renderer_executable,
    )


def caption_docx_images(
    extraction: DocxImageExtraction,
    source: DocumentSource,
    settings: ImageCaptionSettings,
    client: CaptionClient,
    *,
    renderer_executable: Path | None = None,
) -> tuple[ImageCaption, ...]:
    """Caption extracted images concurrently while preserving source order."""

    def caption_one(image: DocxImage) -> ImageCaption:
        context = build_image_context(
            image,
            source,
            max_characters=settings.max_context_characters,
        )
        caption_bytes, caption_mime_type = _prepare_caption_media(
            image,
            renderer_executable=renderer_executable,
            timeout_seconds=settings.timeout_seconds,
        )
        response = client.caption(
            image_bytes=caption_bytes,
            mime_type=caption_mime_type,
            context=context,
        )
        return ImageCaption(
            schema_version=IMAGE_CAPTION_SCHEMA_VERSION,
            image_id=image.image_id,
            source_name=image.source_name,
            source_sha256=image.source_sha256,
            source_ref=image.source_ref,
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


def write_image_captions(output_path: Path, captions: tuple[ImageCaption, ...]) -> None:
    """Atomically write caption records as UTF-8 JSON Lines."""
    records = (caption.to_dict() for caption in captions)
    serialized = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    )
    _write_text(output_path, serialized)


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
            # The executable is validated as a file and shell=False prevents command injection.
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


def _required_value(environment: Mapping[str, str], variable_name: str) -> str:
    """Return one non-empty environment value."""
    value = environment.get(variable_name)
    if value is None or not value.strip():
        raise ImageCaptionConfigurationError.missing(variable_name)
    return value.strip()


def _number(environment: Mapping[str, str], variable_name: str) -> float:
    """Parse one required floating-point environment value."""
    value = _required_value(environment, variable_name)
    try:
        return float(value)
    except ValueError as error:
        raise ImageCaptionConfigurationError.invalid_number(variable_name) from error


def _positive_float(environment: Mapping[str, str], variable_name: str) -> float:
    """Parse one required positive floating-point environment value."""
    value = _number(environment, variable_name)
    if value <= 0:
        raise ImageCaptionConfigurationError.non_positive(variable_name)
    return value


def _positive_int(environment: Mapping[str, str], variable_name: str) -> int:
    """Parse one required positive integer environment value."""
    raw_value = _required_value(environment, variable_name)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ImageCaptionConfigurationError.invalid_number(variable_name) from error
    if value <= 0:
        raise ImageCaptionConfigurationError.non_positive(variable_name)
    return value


def _write_text(output_path: Path, parts: Iterable[str]) -> None:
    """Atomically write text parts to one output path."""
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
            temporary_file.writelines(parts)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
