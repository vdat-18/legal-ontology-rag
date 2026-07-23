"""Extract inline fields, hyperlinks, and bookmarks from DOCX paragraphs."""

from __future__ import annotations

import shlex
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

from defusedxml import ElementTree as SafeElementTree

_RELATIONSHIPS_PART = "word/_rels/document.xml.rels"

_WORDPROCESSINGML_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WORDPROCESSINGML_PREFIX = f"{{{_WORDPROCESSINGML_NAMESPACE}}}"
_OFFICE_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_OFFICE_RELATIONSHIP_PREFIX = f"{{{_OFFICE_RELATIONSHIP_NAMESPACE}}}"
_PACKAGE_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
_PACKAGE_RELATIONSHIP_PREFIX = f"{{{_PACKAGE_RELATIONSHIP_NAMESPACE}}}"

_FIELD_CHARACTER_TAG = f"{_WORDPROCESSINGML_PREFIX}fldChar"
_FIELD_INSTRUCTION_TAG = f"{_WORDPROCESSINGML_PREFIX}instrText"
_SIMPLE_FIELD_TAG = f"{_WORDPROCESSINGML_PREFIX}fldSimple"
_HYPERLINK_TAG = f"{_WORDPROCESSINGML_PREFIX}hyperlink"
_BOOKMARK_START_TAG = f"{_WORDPROCESSINGML_PREFIX}bookmarkStart"
_BOOKMARK_END_TAG = f"{_WORDPROCESSINGML_PREFIX}bookmarkEnd"
_TEXT_TAG = f"{_WORDPROCESSINGML_PREFIX}t"
_TAB_TAG = f"{_WORDPROCESSINGML_PREFIX}tab"
_BREAK_TAG = f"{_WORDPROCESSINGML_PREFIX}br"
_CARRIAGE_RETURN_TAG = f"{_WORDPROCESSINGML_PREFIX}cr"
_NO_BREAK_HYPHEN_TAG = f"{_WORDPROCESSINGML_PREFIX}noBreakHyphen"
_SOFT_HYPHEN_TAG = f"{_WORDPROCESSINGML_PREFIX}softHyphen"
_RELATIONSHIP_TAG = f"{_PACKAGE_RELATIONSHIP_PREFIX}Relationship"

_FIELD_CHARACTER_TYPE_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}fldCharType"
_FIELD_INSTRUCTION_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}instr"
_BOOKMARK_ID_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}id"
_BOOKMARK_NAME_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}name"
_HYPERLINK_ANCHOR_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}anchor"
_RELATIONSHIP_ID_ATTRIBUTE = f"{_OFFICE_RELATIONSHIP_PREFIX}id"

_FIELD_BEGIN = "begin"
_FIELD_SEPARATE = "separate"
_FIELD_END = "end"
_HYPERLINK_FIELD_TYPE = "hyperlink"
_HYPERLINK_LOCATION_SWITCH = "\\l"
_HYPERLINK_RELATIONSHIP_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


@dataclass(frozen=True, slots=True)
class DocxField:
    """One native Word field and its visible result span."""

    field_index: int
    source_ref: str
    instruction: str
    field_type: str | None
    result_text: str
    start_offset: int
    end_offset: int
    target: str | None
    anchor: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable field annotation."""
        return {
            "field_index": self.field_index,
            "source_ref": self.source_ref,
            "instruction": self.instruction,
            "field_type": self.field_type,
            "result_text": self.result_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "target": self.target,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class DocxHyperlink:
    """One native hyperlink element resolved through document relationships."""

    hyperlink_index: int
    source_ref: str
    result_text: str
    start_offset: int
    end_offset: int
    relationship_id: str | None
    target: str | None
    external: bool | None
    anchor: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable hyperlink annotation."""
        return {
            "hyperlink_index": self.hyperlink_index,
            "source_ref": self.source_ref,
            "result_text": self.result_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "relationship_id": self.relationship_id,
            "target": self.target,
            "external": self.external,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class DocxBookmark:
    """One bookmark marker pair intersecting a paragraph."""

    bookmark_id: str
    name: str | None
    source_ref: str
    start_offset: int | None
    end_offset: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable bookmark annotation."""
        return {
            "bookmark_id": self.bookmark_id,
            "name": self.name,
            "source_ref": self.source_ref,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass(frozen=True, slots=True)
class ParagraphAnnotations:
    """Inline OOXML annotations attached to one paragraph."""

    fields: tuple[DocxField, ...]
    hyperlinks: tuple[DocxHyperlink, ...]
    bookmarks: tuple[DocxBookmark, ...]

    @property
    def present(self) -> bool:
        """Return whether the paragraph contains any preserved annotation."""
        return bool(self.fields or self.hyperlinks or self.bookmarks)


@dataclass(frozen=True, slots=True)
class DocumentRelationship:
    """One main-document relationship used by a hyperlink element."""

    relationship_type: str
    target: str
    external: bool


class DocxAnnotationExtractionError(ValueError):
    """Raised when inline DOCX relationship markup is structurally invalid."""

    def __init__(self, detail: str, source_ref: str) -> None:
        """Identify the invalid markup and its native source reference."""
        super().__init__(f"Invalid DOCX inline annotation ({detail}): {source_ref}")

    @classmethod
    def unclosed_field(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for a field without its closing marker."""
        return cls("unclosed field", source_ref)

    @classmethod
    def orphan_instruction(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for instruction text outside a field."""
        return cls("instruction without field", source_ref)

    @classmethod
    def orphan_separator(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for a separator outside a field."""
        return cls("separator without field", source_ref)

    @classmethod
    def orphan_end(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for a closing marker outside a field."""
        return cls("end without field", source_ref)

    @classmethod
    def malformed_instruction(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for field instruction syntax that cannot be tokenized."""
        return cls("malformed field instruction", source_ref)

    @classmethod
    def unknown_relationship(cls, relationship_id: str) -> DocxAnnotationExtractionError:
        """Build an error for an unresolved hyperlink relationship."""
        return cls("unknown relationship", relationship_id)

    @classmethod
    def invalid_relationship_type(cls, relationship_id: str) -> DocxAnnotationExtractionError:
        """Build an error for a hyperlink using another relationship type."""
        return cls("invalid hyperlink relationship type", relationship_id)

    @classmethod
    def bookmark_without_id(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for a bookmark marker without an ID."""
        return cls("bookmark without ID", source_ref)

    @classmethod
    def bookmark_without_name(cls, source_ref: str) -> DocxAnnotationExtractionError:
        """Build an error for a bookmark start without a name."""
        return cls("bookmark start without name", source_ref)


@dataclass(slots=True)
class _OpenField:
    """Mutable state while reading one complex field."""

    field_index: int
    source_ref: str
    instruction_parts: list[str]
    result_start: int | None = None


@dataclass(slots=True)
class _BookmarkState:
    """Mutable bookmark markers collected within one paragraph."""

    bookmark_id: str
    name: str | None
    source_ref: str
    start_offset: int | None
    end_offset: int | None


def read_document_relationships(archive: zipfile.ZipFile) -> dict[str, DocumentRelationship]:
    """Read main-document relationships keyed by their OOXML IDs."""
    try:
        relationship_bytes = archive.read(_RELATIONSHIPS_PART)
    except KeyError:
        return {}

    root = SafeElementTree.fromstring(relationship_bytes)
    relationships: dict[str, DocumentRelationship] = {}
    for element in root.findall(_RELATIONSHIP_TAG):
        relationship_id = element.get("Id")
        relationship_type = element.get("Type")
        target = element.get("Target")
        if relationship_id is None or relationship_type is None or target is None:
            continue
        relationships[relationship_id] = DocumentRelationship(
            relationship_type=relationship_type,
            target=target,
            external=element.get("TargetMode") == "External",
        )
    return relationships


class _ParagraphAnnotationParser:
    """Stateful OOXML walker for one paragraph's inline annotations."""

    def __init__(self, paragraph_ref: str, relationships: Mapping[str, DocumentRelationship]) -> None:
        """Initialize native identity, relationships, and parser state."""
        self.paragraph_ref = paragraph_ref
        self.relationships = relationships
        self.visible_parts: list[str] = []
        self.visible_length = 0
        self.fields: list[DocxField] = []
        self.hyperlinks: list[DocxHyperlink] = []
        self.bookmark_states: list[_BookmarkState] = []
        self.bookmark_starts: dict[str, _BookmarkState] = {}
        self.open_fields: list[_OpenField] = []
        self.field_count = 0
        self.hyperlink_count = 0

    def parse(self, paragraph: ElementTree.Element) -> ParagraphAnnotations:
        """Walk one paragraph and return immutable annotations."""
        self._walk(paragraph)
        if self.open_fields:
            raise DocxAnnotationExtractionError.unclosed_field(self.paragraph_ref)

        self.fields.sort(key=lambda field: field.field_index)
        self.hyperlinks.sort(key=lambda hyperlink: hyperlink.hyperlink_index)
        bookmarks = tuple(
            DocxBookmark(
                bookmark_id=state.bookmark_id,
                name=state.name,
                source_ref=state.source_ref,
                start_offset=state.start_offset,
                end_offset=state.end_offset,
            )
            for state in self.bookmark_states
        )
        return ParagraphAnnotations(tuple(self.fields), tuple(self.hyperlinks), bookmarks)

    def _walk(self, element: ElementTree.Element) -> None:
        """Dispatch one OOXML element while retaining native order."""
        if element.tag in {_FIELD_CHARACTER_TAG, _FIELD_INSTRUCTION_TAG, _SIMPLE_FIELD_TAG}:
            self._walk_field_markup(element)
            return
        if element.tag == _HYPERLINK_TAG:
            self._walk_hyperlink(element)
            return
        if element.tag in {_BOOKMARK_START_TAG, _BOOKMARK_END_TAG}:
            self._walk_bookmark(element)
            return

        visible_text = _visible_element_text(element)
        if visible_text is not None:
            self.visible_parts.append(visible_text)
            self.visible_length += len(visible_text)
            return
        for child in element:
            self._walk(child)

    def _walk_field_markup(self, element: ElementTree.Element) -> None:
        """Process one complex or simple Word field element."""
        if element.tag == _FIELD_CHARACTER_TAG:
            self._handle_field_character(element)
        elif element.tag == _FIELD_INSTRUCTION_TAG:
            if not self.open_fields:
                raise DocxAnnotationExtractionError.orphan_instruction(self.paragraph_ref)
            self.open_fields[-1].instruction_parts.append(element.text or "")
        else:
            self._walk_simple_field(element)

    def _handle_field_character(self, element: ElementTree.Element) -> None:
        """Update the complex-field stack from one native marker."""
        field_character_type = element.get(_FIELD_CHARACTER_TYPE_ATTRIBUTE)
        if field_character_type == _FIELD_BEGIN:
            self.open_fields.append(
                _OpenField(
                    field_index=self.field_count,
                    source_ref=f"{self.paragraph_ref}/field/{self.field_count}",
                    instruction_parts=[],
                )
            )
            self.field_count += 1
        elif field_character_type == _FIELD_SEPARATE:
            if not self.open_fields:
                raise DocxAnnotationExtractionError.orphan_separator(self.paragraph_ref)
            self.open_fields[-1].result_start = self.visible_length
        elif field_character_type == _FIELD_END:
            if not self.open_fields:
                raise DocxAnnotationExtractionError.orphan_end(self.paragraph_ref)
            self._finish_field(self.open_fields.pop())

    def _walk_simple_field(self, element: ElementTree.Element) -> None:
        """Read one field encoded as a single fldSimple element."""
        instruction = (element.get(_FIELD_INSTRUCTION_ATTRIBUTE) or "").strip()
        field_index = self.field_count
        self.field_count += 1
        start_offset = self.visible_length
        for child in element:
            self._walk(child)
        self.fields.append(
            _build_field(
                field_index=field_index,
                source_ref=f"{self.paragraph_ref}/field/{field_index}",
                instruction=instruction,
                visible_text="".join(self.visible_parts),
                start_offset=start_offset,
                end_offset=self.visible_length,
            )
        )

    def _finish_field(self, state: _OpenField) -> None:
        """Materialize one completed complex field."""
        start_offset = state.result_start if state.result_start is not None else self.visible_length
        self.fields.append(
            _build_field(
                field_index=state.field_index,
                source_ref=state.source_ref,
                instruction="".join(state.instruction_parts).strip(),
                visible_text="".join(self.visible_parts),
                start_offset=start_offset,
                end_offset=self.visible_length,
            )
        )

    def _walk_hyperlink(self, element: ElementTree.Element) -> None:
        """Resolve one native hyperlink wrapper and its visible result."""
        hyperlink_index = self.hyperlink_count
        self.hyperlink_count += 1
        start_offset = self.visible_length
        for child in element:
            self._walk(child)

        relationship_id = element.get(_RELATIONSHIP_ID_ATTRIBUTE)
        relationship = None
        if relationship_id is not None:
            relationship = self.relationships.get(relationship_id)
            if relationship is None:
                raise DocxAnnotationExtractionError.unknown_relationship(relationship_id)
            if relationship.relationship_type != _HYPERLINK_RELATIONSHIP_TYPE:
                raise DocxAnnotationExtractionError.invalid_relationship_type(relationship_id)
        visible_text = "".join(self.visible_parts)
        self.hyperlinks.append(
            DocxHyperlink(
                hyperlink_index=hyperlink_index,
                source_ref=f"{self.paragraph_ref}/hyperlink/{hyperlink_index}",
                result_text=visible_text[start_offset : self.visible_length],
                start_offset=start_offset,
                end_offset=self.visible_length,
                relationship_id=relationship_id,
                target=relationship.target if relationship is not None else None,
                external=relationship.external if relationship is not None else None,
                anchor=element.get(_HYPERLINK_ANCHOR_ATTRIBUTE),
            )
        )

    def _walk_bookmark(self, element: ElementTree.Element) -> None:
        """Preserve one bookmark start or end marker at its text offset."""
        bookmark_id = element.get(_BOOKMARK_ID_ATTRIBUTE)
        if bookmark_id is None:
            raise DocxAnnotationExtractionError.bookmark_without_id(self.paragraph_ref)
        if element.tag == _BOOKMARK_START_TAG:
            self._start_bookmark(bookmark_id, element)
            return

        state = self.bookmark_starts.get(bookmark_id)
        if state is not None:
            state.end_offset = self.visible_length
            return
        self.bookmark_states.append(
            _BookmarkState(
                bookmark_id=bookmark_id,
                name=None,
                source_ref=f"{self.paragraph_ref}/bookmark/{len(self.bookmark_states)}",
                start_offset=None,
                end_offset=self.visible_length,
            )
        )

    def _start_bookmark(self, bookmark_id: str, element: ElementTree.Element) -> None:
        """Record one bookmark start marker for later pairing."""
        name = element.get(_BOOKMARK_NAME_ATTRIBUTE)
        if name is None:
            raise DocxAnnotationExtractionError.bookmark_without_name(self.paragraph_ref)
        state = _BookmarkState(
            bookmark_id=bookmark_id,
            name=name,
            source_ref=f"{self.paragraph_ref}/bookmark/{len(self.bookmark_states)}",
            start_offset=self.visible_length,
            end_offset=None,
        )
        self.bookmark_states.append(state)
        self.bookmark_starts[bookmark_id] = state


def extract_paragraph_annotations(
    paragraph: ElementTree.Element,
    paragraph_ref: str,
    relationships: Mapping[str, DocumentRelationship],
) -> ParagraphAnnotations:
    """Extract fields, hyperlinks, and bookmark markers in visible-text order."""
    return _ParagraphAnnotationParser(paragraph_ref, relationships).parse(paragraph)


def _build_field(
    *,
    field_index: int,
    source_ref: str,
    instruction: str,
    visible_text: str,
    start_offset: int,
    end_offset: int,
) -> DocxField:
    """Build one immutable field from its instruction and visible span."""
    try:
        field_type, target, anchor = _parse_field_instruction(instruction)
    except ValueError as error:
        raise DocxAnnotationExtractionError.malformed_instruction(source_ref) from error
    return DocxField(
        field_index=field_index,
        source_ref=source_ref,
        instruction=instruction,
        field_type=field_type,
        result_text=visible_text[start_offset:end_offset],
        start_offset=start_offset,
        end_offset=end_offset,
        target=target,
        anchor=anchor,
    )


def _visible_element_text(element: ElementTree.Element) -> str | None:
    """Return the visible text represented directly by one OOXML element."""
    if element.tag == _TEXT_TAG:
        return element.text or ""
    if element.tag == _TAB_TAG:
        return "\t"
    if element.tag in {_BREAK_TAG, _CARRIAGE_RETURN_TAG}:
        return "\n"
    if element.tag == _NO_BREAK_HYPHEN_TAG:
        return "\u2011"
    if element.tag == _SOFT_HYPHEN_TAG:
        return "\u00ad"
    return None


def _parse_field_instruction(instruction: str) -> tuple[str | None, str | None, str | None]:
    """Parse the bounded Word field instruction syntax needed for hyperlink targets."""
    if not instruction:
        return None, None, None

    tokens = shlex.split(instruction, posix=False)
    field_type = _unquote(tokens[0]) if tokens else None
    if field_type is None or field_type.casefold() != _HYPERLINK_FIELD_TYPE:
        return field_type, None, None

    target = None
    if len(tokens) > 1 and not tokens[1].startswith("\\"):
        target = _unquote(tokens[1])

    anchor = None
    for index, token in enumerate(tokens[:-1]):
        if token.casefold() == _HYPERLINK_LOCATION_SWITCH:
            anchor = _unquote(tokens[index + 1])
            break
    return field_type, target, anchor


def _unquote(token: str) -> str:
    """Remove one matching quote pair emitted by the standard-library lexer."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token
