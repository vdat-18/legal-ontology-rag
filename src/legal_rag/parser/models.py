"""Models for enriched parser output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.parser.document_source import DocumentSource, DocumentSourceUnit

PARSER_OUTPUT_SCHEMA_VERSION = "1.0.0"
IMAGE_CAPTION_SCHEMA_VERSION = "1.2.0"
KIND_IMAGE = "image"


@dataclass(frozen=True, slots=True)
class CaptionModelResponse:
    """Typed semantic fields returned by the caption model."""

    description_vi: str
    visible_text: str | None


@dataclass(frozen=True, slots=True)
class ImageCaption:
    """One caption tied to immutable image and context provenance."""

    schema_version: str
    image_id: str
    source_name: str
    source_sha256: str
    body_index: int
    source_ref: str
    parent_source_ref: str
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
            "body_index": self.body_index,
            "source_ref": self.source_ref,
            "parent_source_ref": self.parent_source_ref,
            "media_sha256": self.media_sha256,
            "source_mime_type": self.source_mime_type,
            "caption_mime_type": self.caption_mime_type,
            "context_source_refs": list(self.context_source_refs),
            "requested_model": self.requested_model,
            "description_vi": self.description_vi,
            "visible_text": self.visible_text,
        }


@dataclass(frozen=True, slots=True)
class ParsedDocumentUnit:
    """One source unit or image-only position in final document order."""

    sequence_index: int
    source: DocumentSourceUnit | None
    images: tuple[ImageCaption, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return one final parser-output record."""
        if self.source is not None:
            record = self.source.to_dict()
            record["source_schema_version"] = record["schema_version"]
            record["schema_version"] = PARSER_OUTPUT_SCHEMA_VERSION
            record["sequence_index"] = self.sequence_index
            record["images"] = [image.to_dict() for image in self.images]
            return record

        image = self.images[0]
        return {
            "schema_version": PARSER_OUTPUT_SCHEMA_VERSION,
            "unit_id": image.image_id,
            "source_name": image.source_name,
            "source_sha256": image.source_sha256,
            "sequence_index": self.sequence_index,
            "body_index": image.body_index,
            "source_ref": image.parent_source_ref,
            "kind": KIND_IMAGE,
            "images": [image.to_dict()],
        }


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """One ordered source document after all parser enrichment stages."""

    source_name: str
    source_sha256: str
    units: tuple[ParsedDocumentUnit, ...]

    @property
    def image_count(self) -> int:
        """Return the number of captioned image occurrences."""
        return sum(len(unit.images) for unit in self.units)

    @classmethod
    def from_source(
        cls,
        source: DocumentSource,
        captions: tuple[ImageCaption, ...],
    ) -> ParsedDocument:
        """Attach captions to their exact top-level source units."""
        captions_by_body_index: dict[int, list[ImageCaption]] = {}
        for caption in captions:
            captions_by_body_index.setdefault(caption.body_index, []).append(caption)

        source_body_indices = {unit.body_index for unit in source.units}
        ordered_items: list[DocumentSourceUnit | ImageCaption] = [*source.units]
        ordered_items.extend(caption for caption in captions if caption.body_index not in source_body_indices)
        ordered_items.sort(key=lambda item: item.body_index)

        units: list[ParsedDocumentUnit] = []
        for sequence_index, item in enumerate(ordered_items):
            if isinstance(item, DocumentSourceUnit):
                unit_images = tuple(captions_by_body_index.get(item.body_index, ()))
                units.append(ParsedDocumentUnit(sequence_index, item, unit_images))
            else:
                units.append(ParsedDocumentUnit(sequence_index, None, (item,)))
        return cls(source.source_name, source.source_sha256, tuple(units))
