"""Models for enriched parser output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.parser.attachments import DocxAttachment
from legal_rag.parser.document_source import DocumentSource, DocumentSourceUnit
from legal_rag.parser.hashing import sha256_json

PARSER_OUTPUT_SCHEMA_VERSION = "2.0.0"
IMAGE_CAPTION_SCHEMA_VERSION = "2.0.0"
KIND_IMAGE = "image"
KIND_ATTACHMENT = "attachment"


@dataclass(frozen=True, slots=True)
class CaptionModelResponse:
    """Typed Vietnamese description returned by the caption model."""

    description_vi: str


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
        }


@dataclass(frozen=True, slots=True)
class ParsedDocumentUnit:
    """One source unit or media-only position in final document order."""

    sequence_index: int
    source: DocumentSourceUnit | None
    images: tuple[ImageCaption, ...]
    attachments: tuple[DocxAttachment, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return one final parser-output record."""
        if self.source is not None:
            record = self.source.to_dict()
            record["source_schema_version"] = record["schema_version"]
            record["schema_version"] = PARSER_OUTPUT_SCHEMA_VERSION
            record["sequence_index"] = self.sequence_index
            record["images"] = [image.to_dict() for image in self.images]
            record["attachments"] = [attachment.to_dict() for attachment in self.attachments]
            return record

        media = self.images[0] if self.images else self.attachments[0]
        return {
            "schema_version": PARSER_OUTPUT_SCHEMA_VERSION,
            "unit_id": media.image_id if isinstance(media, ImageCaption) else media.attachment_id,
            "source_name": media.source_name,
            "source_sha256": media.source_sha256,
            "sequence_index": self.sequence_index,
            "body_index": media.body_index,
            "source_ref": media.parent_source_ref,
            "kind": KIND_IMAGE if self.images else KIND_ATTACHMENT,
            "images": [image.to_dict() for image in self.images],
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }


@dataclass(frozen=True, slots=True)
class AttachmentChildDocument:
    """Parsed records from one supported document inside an attachment."""

    parent_attachment_id: str
    member_index: int
    member_path: str
    member_sha256: str
    source_format: str
    converted_from_format: str | None
    native_records: tuple[dict[str, Any], ...]
    attachments: tuple[DocxAttachment, ...] = ()

    def records(self, root_source_name: str, root_source_sha256: str) -> tuple[dict[str, Any], ...]:
        """Add immutable attachment provenance to each native child record."""
        records: list[dict[str, Any]] = []
        for child_sequence_index, native_record in enumerate(self.native_records):
            record = dict(native_record)
            if self.converted_from_format is None:
                native_unit_id = record["unit_id"]
                native_source_name = record["source_name"]
                native_source_sha256 = record["source_sha256"]
            else:
                native_unit_id = sha256_json(
                    {
                        "kind": record["kind"],
                        "member_index": self.member_index,
                        "member_path": self.member_path,
                        "member_sha256": self.member_sha256,
                        "source_ref": record["source_ref"],
                    }
                )
                native_source_name = self.member_path
                native_source_sha256 = self.member_sha256
                if "images" in record:
                    images: list[dict[str, Any]] = []
                    for native_image in record["images"]:
                        image = dict(native_image)
                        image["image_id"] = sha256_json(
                            {
                                "media_sha256": image["media_sha256"],
                                "member_index": self.member_index,
                                "member_path": self.member_path,
                                "member_sha256": self.member_sha256,
                                "source_ref": image["source_ref"],
                            }
                        )
                        image["source_name"] = self.member_path
                        image["source_sha256"] = self.member_sha256
                        images.append(image)
                    record["images"] = images

                if "attachments" in record:
                    attachments: list[dict[str, Any]] = []
                    for native_attachment in record["attachments"]:
                        attachment = dict(native_attachment)
                        attachment["attachment_id"] = sha256_json(
                            {
                                "member_index": self.member_index,
                                "member_path": self.member_path,
                                "member_sha256": self.member_sha256,
                                "payload_sha256": attachment["payload_sha256"],
                                "source_ref": attachment["source_ref"],
                            }
                        )
                        attachment["source_name"] = self.member_path
                        attachment["source_sha256"] = self.member_sha256
                        attachments.append(attachment)
                    record["attachments"] = attachments
            child_parser_schema_version = record["schema_version"]
            record.setdefault("source_schema_version", child_parser_schema_version)
            record["schema_version"] = PARSER_OUTPUT_SCHEMA_VERSION
            record["unit_id"] = sha256_json(
                {
                    "member_sha256": self.member_sha256,
                    "native_unit_id": native_unit_id,
                    "parent_attachment_id": self.parent_attachment_id,
                }
            )
            record["native_unit_id"] = native_unit_id
            record["child_parser_schema_version"] = child_parser_schema_version
            record["native_source_name"] = native_source_name
            record["native_source_sha256"] = native_source_sha256
            record["source_name"] = self.member_path
            record["source_sha256"] = self.member_sha256
            record["record_scope"] = "attachment_child"
            record["root_source_name"] = root_source_name
            record["root_source_sha256"] = root_source_sha256
            record["parent_attachment_id"] = self.parent_attachment_id
            record["attachment_member_index"] = self.member_index
            record["attachment_member_path"] = self.member_path
            record["attachment_member_sha256"] = self.member_sha256
            record["child_sequence_index"] = child_sequence_index
            record["source_format"] = self.source_format
            record["converted_from_format"] = self.converted_from_format
            records.append(record)
        return tuple(records)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """One ordered source document after all parser enrichment stages."""

    source_name: str
    source_sha256: str
    units: tuple[ParsedDocumentUnit, ...]
    child_documents: tuple[AttachmentChildDocument, ...] = ()

    @property
    def image_count(self) -> int:
        """Return the number of captioned image occurrences."""
        return sum(len(unit.images) for unit in self.units)

    @property
    def attachment_count(self) -> int:
        """Return the number of extracted attachment occurrences."""
        return sum(len(unit.attachments) for unit in self.units)

    @property
    def child_record_count(self) -> int:
        """Return the number of searchable records produced from child documents."""
        return sum(len(document.native_records) for document in self.child_documents)

    @classmethod
    def from_source(
        cls,
        source: DocumentSource,
        captions: tuple[ImageCaption, ...],
        attachments: tuple[DocxAttachment, ...] = (),
        child_documents: tuple[AttachmentChildDocument, ...] = (),
    ) -> ParsedDocument:
        """Attach captions and attachments to their exact source positions."""
        captions_by_body_index: dict[int, list[ImageCaption]] = {}
        for caption in captions:
            captions_by_body_index.setdefault(caption.body_index, []).append(caption)

        attachments_by_body_index: dict[int, list[DocxAttachment]] = {}
        for attachment in attachments:
            attachments_by_body_index.setdefault(attachment.body_index, []).append(attachment)

        source_by_body_index = {unit.body_index: unit for unit in source.units}
        body_indices = sorted(
            source_by_body_index.keys() | captions_by_body_index.keys() | attachments_by_body_index.keys()
        )

        units: list[ParsedDocumentUnit] = []
        for sequence_index, body_index in enumerate(body_indices):
            units.append(
                ParsedDocumentUnit(
                    sequence_index=sequence_index,
                    source=source_by_body_index.get(body_index),
                    images=tuple(captions_by_body_index.get(body_index, ())),
                    attachments=tuple(attachments_by_body_index.get(body_index, ())),
                )
            )
        return cls(source.source_name, source.source_sha256, tuple(units), child_documents)
