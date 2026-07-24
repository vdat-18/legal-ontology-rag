"""Tests for structure-aware oversized-unit splitting."""

from __future__ import annotations

from typing import Any

import pytest

from legal_rag.chunker.models import ChunkKind, NormalizedUnit, SourceKind, SourceRecord, SourceScope
from legal_rag.chunker.rendering import render_source_scope, render_spreadsheet_row, render_table
from legal_rag.chunker.splitting import split_unit
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.embedding.tokenizer import EmbeddingTokenizer


class CharacterTokenizer:
    """Fast test tokenizer that maps every source character to one token."""

    is_fast = True

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode every character and optionally add two wrapper tokens."""
        token_ids = list(range(len(text)))
        if add_special_tokens:
            token_ids = [0, *token_ids, 1]
        return token_ids[:max_length] if truncation and max_length is not None else token_ids

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return wrapper-token counts for single or paired inputs."""
        return 3 if pair else 2

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        stride: int,
        return_overflowing_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        """Return overflowing character windows in Transformers format."""
        assert add_special_tokens
        assert truncation
        assert return_overflowing_tokens
        assert return_offsets_mapping
        offsets = [(index, index + 1) for index in range(len(text))]
        content_budget = max_length - self.num_special_tokens_to_add()
        token_windows: list[list[int]] = []
        offset_windows: list[list[tuple[int, int]]] = []
        start = 0
        while start < len(offsets):
            end = min(start + content_budget, len(offsets))
            token_windows.append([0, *range(start, end), 1])
            offset_windows.append([(0, 0), *offsets[start:end], (0, 0)])
            if end == len(offsets):
                break
            start = end - stride
        return {"input_ids": token_windows, "offset_mapping": offset_windows}


def _tokenizer() -> EmbeddingTokenizer:
    """Build a bounded tokenizer without external model downloads."""
    profile = EmbeddingProfile(
        profile_id="test",
        model_id="test/model",
        model_revision="model-revision",
        tokenizer_id="test/tokenizer",
        tokenizer_revision="tokenizer-revision",
        dimensions=1024,
        max_input_tokens=4096,
        query_instruction=None,
    )
    return EmbeddingTokenizer(profile, CharacterTokenizer())


def _unit(kind: SourceKind, text: str) -> NormalizedUnit:
    """Build one text-only normalized unit."""
    return NormalizedUnit(
        scope_id="scope-id",
        kind=kind,
        sequence_index=2,
        unit_ids=("unit-id",),
        source_refs=("source/2",),
        text=text,
        source_payload={},
    )


def _record(kind: SourceKind, payload: dict[str, Any]) -> SourceRecord:
    """Build one structural parser record."""
    return SourceRecord(
        unit_id="unit-id",
        source_name="source.docx",
        source_sha256="source-hash",
        source_ref="source/0",
        kind=kind,
        record_scope="root",
        order_index=0,
        payload=payload,
    )


def _render(record: SourceRecord) -> NormalizedUnit:
    """Render one structural record through the production renderer."""
    scope = SourceScope(
        scope_id="scope-id",
        source_name="source.docx",
        source_sha256="source-hash",
        record_scope="root",
        root_source_name="source.docx",
        root_source_sha256="source-hash",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        records=(record,),
    )
    return render_source_scope(scope)[0]


def _paragraph_block(source_ref: str, text: str) -> dict[str, Any]:
    """Build one table paragraph block."""
    return {"kind": "paragraph", "source_ref": source_ref, "text": text}


def _cell(index: int, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one native table cell."""
    return {"cell_index": index, "source_ref": f"cell/{index}", "blocks": blocks}


def _row(index: int, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one native table row."""
    return {"row_index": index, "source_ref": f"row/{index}", "cells": cells}


def test_unit_within_limit_remains_one_exact_fragment() -> None:
    """Normal units do not receive overlap or text changes."""
    tokenizer = _tokenizer()
    unit = _unit("paragraph", "Điều 5.")

    fragments = split_unit(unit, tokenizer, max_tokens=20, overlap_tokens=4)

    assert len(fragments) == 1
    assert fragments[0].text == unit.text
    assert fragments[0].source_refs == unit.source_refs
    assert fragments[0].token_count == tokenizer.count_tokens(unit.text)


@pytest.mark.parametrize(
    ("kind", "fragment_kind"),
    [("paragraph", "paragraph_group"), ("image", "image")],
)
def test_split_text_unit_uses_exact_offsets_and_overlap(kind: SourceKind, fragment_kind: ChunkKind) -> None:
    """Oversized paragraph and image text use tokenizer windows without decode changes."""
    tokenizer = _tokenizer()
    source_text = "abcdefghij"

    fragments = split_unit(_unit(kind, source_text), tokenizer, max_tokens=8, overlap_tokens=2)

    assert [fragment.text for fragment in fragments] == ["abcdef", "efghij"]
    assert fragments[0].text[-2:] == fragments[1].text[:2]
    assert all(fragment.kind == fragment_kind for fragment in fragments)
    assert all(fragment.token_count <= 8 for fragment in fragments)
    assert all(fragment.sequence_start == fragment.sequence_end == 2 for fragment in fragments)


def test_split_table_prefers_consecutive_rows_without_overlap() -> None:
    """An oversized table separates rows before descending into cells or text windows."""
    tokenizer = _tokenizer()
    rows = [
        _row(0, [_cell(0, [_paragraph_block("row/0/block/0", "AAAA")])]),
        _row(1, [_cell(0, [_paragraph_block("row/1/block/0", "BBBB")])]),
    ]
    unit = _render(_record("table", {"rows": rows, "images": []}))
    row_contents = [render_table("source/0", (row,)) for row in rows]
    row_limit = max(tokenizer.count_tokens(content.text) for content in row_contents if content is not None)

    fragments = split_unit(unit, tokenizer, max_tokens=row_limit, overlap_tokens=2)

    assert len(fragments) == 2
    assert "AAAA" in fragments[0].text
    assert "BBBB" not in fragments[0].text
    assert "BBBB" in fragments[1].text
    assert "AAAA" not in fragments[1].text
    assert fragments[0].source_refs == ("source/0", "row/0", "cell/0", "row/0/block/0")
    assert fragments[1].source_refs == ("source/0", "row/1", "cell/0", "row/1/block/0")


def test_split_oversized_row_prefers_cells() -> None:
    """One oversized row is divided into consecutive cells without token overlap."""
    tokenizer = _tokenizer()
    row = _row(
        0,
        [
            _cell(0, [_paragraph_block("cell/0/block/0", "AAAA")]),
            _cell(1, [_paragraph_block("cell/1/block/0", "BBBB")]),
        ],
    )
    unit = _render(_record("table", {"rows": [row], "images": []}))
    cell_contents = []
    for cell in row["cells"]:
        row_payload = dict(row)
        row_payload["cells"] = [cell]
        cell_contents.append(render_table("source/0", (row_payload,)))
    cell_limit = max(tokenizer.count_tokens(content.text) for content in cell_contents if content is not None)

    fragments = split_unit(unit, tokenizer, max_tokens=cell_limit, overlap_tokens=2)

    assert len(fragments) == 2
    assert "AAAA" in fragments[0].text
    assert "BBBB" not in fragments[0].text
    assert "BBBB" in fragments[1].text
    assert "AAAA" not in fragments[1].text


def test_split_oversized_cell_prefers_blocks_then_forces_token_windows() -> None:
    """Cell blocks stay atomic until one block itself exceeds the configured limit."""
    tokenizer = _tokenizer()
    long_text = "N" * 80
    row = _row(
        0,
        [
            _cell(
                0,
                [
                    _paragraph_block("block/0", "SHORT"),
                    _paragraph_block("block/1", long_text),
                ],
            )
        ],
    )
    unit = _render(_record("table", {"rows": [row], "images": []}))

    fragments = split_unit(unit, tokenizer, max_tokens=50, overlap_tokens=5)

    assert "SHORT" in fragments[0].text
    assert len(fragments) > 2
    assert all(fragment.token_count <= 50 for fragment in fragments)
    assert any("NNNN" in fragment.text for fragment in fragments[1:])
    assert fragments[1].text[-5:] == fragments[2].text[:5]
    assert all("block/1" in fragment.source_refs for fragment in fragments[1:])


def test_split_table_preserves_nested_table_order_and_provenance() -> None:
    """A nested-table block remains ordered and traceable when outer rows split."""
    tokenizer = _tokenizer()
    nested = {
        "kind": "table",
        "table_index": 0,
        "source_ref": "nested/table/0",
        "rows": [_row(0, [_cell(0, [_paragraph_block("nested/block/0", "INNER")])])],
    }
    rows = [
        _row(0, [_cell(0, [nested])]),
        _row(1, [_cell(0, [_paragraph_block("outer/block/1", "OUTER")])]),
    ]
    unit = _render(_record("table", {"rows": rows, "images": []}))
    row_contents = [render_table("source/0", (row,)) for row in rows]
    row_limit = max(tokenizer.count_tokens(content.text) for content in row_contents if content is not None)

    fragments = split_unit(unit, tokenizer, max_tokens=row_limit, overlap_tokens=2)

    assert len(fragments) == 2
    assert fragments[0].text.index("[TABLE 0]") < fragments[0].text.index("INNER")
    assert "nested/table/0" in fragments[0].source_refs
    assert "nested/block/0" in fragments[0].source_refs


def test_split_spreadsheet_prefers_cells_and_forces_only_oversized_cell() -> None:
    """Spreadsheet cells split structurally before an oversized value uses overlap."""
    tokenizer = _tokenizer()
    cells = [
        {"cell_ref": "A1", "value": "SHORT", "formula": None},
        {"cell_ref": "B1", "value": "V" * 80, "formula": "1+1"},
    ]
    unit = _render(
        _record(
            "spreadsheet_row",
            {"sheet_name": "Sheet1", "row_index": 1, "cells": cells},
        )
    )
    first_cell = render_spreadsheet_row("source/0", "Sheet1", 1, (cells[0],))
    assert first_cell is not None
    max_tokens = tokenizer.count_tokens(first_cell.text)

    fragments = split_unit(unit, tokenizer, max_tokens=max_tokens, overlap_tokens=4)

    assert "SHORT" in fragments[0].text
    assert len(fragments) > 2
    assert all(fragment.token_count <= max_tokens for fragment in fragments)
    assert any("[FORMULA]" in fragment.text or "1+1" in fragment.text for fragment in fragments[1:])
