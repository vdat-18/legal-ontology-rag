"""Split oversized normalized units while preserving source structure and provenance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

from legal_rag.chunker.models import ChunkFragment, ChunkKind, NormalizedUnit, SourceKind
from legal_rag.chunker.rendering import RenderedContent, render_spreadsheet_row, render_table
from legal_rag.embedding.tokenizer import EmbeddingTokenizer

_Structure = TypeVar("_Structure")


def split_units(
    units: Sequence[NormalizedUnit],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[ChunkFragment, ...]:
    """Split normalized units in source order using one tokenizer configuration."""
    fragments: list[ChunkFragment] = []
    for unit in units:
        fragments.extend(
            split_unit(
                unit,
                tokenizer,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return tuple(fragments)


def split_unit(
    unit: NormalizedUnit,
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[ChunkFragment, ...]:
    """Split one normalized unit only when its rendered input exceeds the token limit."""
    tokenizer.validate_window_settings(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if tokenizer.fits_within(unit.text, max_tokens=max_tokens):
        return (_fragment(unit, RenderedContent(unit.text, unit.source_refs)),)

    if unit.kind in ("paragraph", "image"):
        contents = _force_split_content(
            RenderedContent(unit.text, unit.source_refs),
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    elif unit.kind == "table":
        contents = _split_table(
            unit,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    elif unit.kind == "spreadsheet_row":
        contents = _split_spreadsheet(
            unit,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    else:
        msg = f"Unsupported normalized unit kind: {unit.kind}"
        raise ValueError(msg)
    return tuple(_fragment(unit, content) for content in contents)


def _split_table(
    unit: NormalizedUnit,
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Split an oversized table by rows before descending into its structure."""
    rows = cast(Sequence[Mapping[str, Any]], unit.source_payload["rows"])
    source_ref = unit.source_refs[0]
    return _split_structures(
        rows,
        lambda group: render_table(source_ref, group),
        lambda row: _split_table_row(
            source_ref,
            row,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ),
        tokenizer,
        max_tokens=max_tokens,
    )


def _split_table_row(
    source_ref: str,
    row: Mapping[str, Any],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Split one oversized table row by its native cells."""
    cells = cast(Sequence[Mapping[str, Any]], row["cells"])
    return _split_structures(
        cells,
        lambda group: _render_table_cells(source_ref, row, group),
        lambda cell: _split_table_cell(
            source_ref,
            row,
            cell,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ),
        tokenizer,
        max_tokens=max_tokens,
    )


def _split_table_cell(
    source_ref: str,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Split one oversized table cell by its ordered paragraph or nested-table blocks."""
    blocks = cast(Sequence[Mapping[str, Any]], cell["blocks"])
    return _split_structures(
        blocks,
        lambda group: _render_table_blocks(source_ref, row, cell, group),
        lambda block: _split_table_block(
            source_ref,
            row,
            cell,
            block,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ),
        tokenizer,
        max_tokens=max_tokens,
    )


def _split_table_block(
    source_ref: str,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    block: Mapping[str, Any],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Force-split one atomic table block by tokenizer character offsets."""
    content = _render_table_blocks(source_ref, row, cell, (block,))
    if content is None:
        return []
    return _force_split_content(
        content,
        tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def _split_spreadsheet(
    unit: NormalizedUnit,
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Split one oversized spreadsheet row by its native cells."""
    cells = cast(Sequence[Mapping[str, Any]], unit.source_payload["cells"])
    source_ref = unit.source_refs[0]
    sheet_name = cast(str, unit.source_payload["sheet_name"])
    row_index = cast(int, unit.source_payload["row_index"])
    return _split_structures(
        cells,
        lambda group: render_spreadsheet_row(source_ref, sheet_name, row_index, group),
        lambda cell: _split_spreadsheet_cell(
            source_ref,
            sheet_name,
            row_index,
            cell,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ),
        tokenizer,
        max_tokens=max_tokens,
    )


def _split_spreadsheet_cell(
    source_ref: str,
    sheet_name: str,
    row_index: int,
    cell: Mapping[str, Any],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Force-split one atomic spreadsheet cell by tokenizer character offsets."""
    content = render_spreadsheet_row(source_ref, sheet_name, row_index, (cell,))
    if content is None:
        return []
    return _force_split_content(
        content,
        tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def _split_structures(
    items: Sequence[_Structure],
    render: Callable[[Sequence[_Structure]], RenderedContent | None],
    split_item: Callable[[_Structure], list[RenderedContent]],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
) -> list[RenderedContent]:
    """Greedily group adjacent structures and descend only when one item is oversized."""
    contents: list[RenderedContent] = []
    group: list[_Structure] = []
    for item in items:
        candidate = render((*group, item))
        if candidate is None or tokenizer.fits_within(candidate.text, max_tokens=max_tokens):
            group.append(item)
            continue

        if group:
            grouped_content = render(group)
            if grouped_content is not None:
                contents.append(grouped_content)
            group = []

        item_content = render((item,))
        if item_content is None or tokenizer.fits_within(item_content.text, max_tokens=max_tokens):
            group.append(item)
        else:
            contents.extend(split_item(item))

    grouped_content = render(group)
    if grouped_content is not None:
        contents.append(grouped_content)
    return contents


def _render_table_cells(
    source_ref: str,
    row: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> RenderedContent | None:
    """Render a contiguous subset of cells from one native row."""
    row_payload = dict(row)
    row_payload["cells"] = list(cells)
    return render_table(source_ref, (row_payload,))


def _render_table_blocks(
    source_ref: str,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
) -> RenderedContent | None:
    """Render a contiguous subset of blocks from one native cell."""
    cell_payload = dict(cell)
    cell_payload["blocks"] = list(blocks)
    return _render_table_cells(source_ref, row, (cell_payload,))


def _force_split_content(
    content: RenderedContent,
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[RenderedContent]:
    """Split one atomic rendered value into exact source-text slices."""
    return [
        RenderedContent(window.text_from(content.text), content.source_refs)
        for window in tokenizer.window_offsets(
            content.text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]


def _fragment(unit: NormalizedUnit, content: RenderedContent) -> ChunkFragment:
    """Build one bounded fragment with its normalized-unit provenance."""
    return ChunkFragment(
        kind=_chunk_kind(unit.kind),
        packing_key=_packing_key(unit),
        sequence_start=unit.sequence_index,
        sequence_end=unit.sequence_index,
        unit_ids=unit.unit_ids,
        source_refs=content.source_refs,
        text=content.text,
        structural_context=unit.structural_context,
    )


def _packing_key(unit: NormalizedUnit) -> tuple[str, ...]:
    """Identify fragments that may share one structure-preserving chunk."""
    if unit.kind == "table":
        return unit.unit_ids
    if unit.kind == "spreadsheet_row":
        return (cast(str, unit.source_payload["sheet_name"]),)
    return ()


def _chunk_kind(kind: SourceKind) -> ChunkKind:
    """Map normalized source kinds to their persisted chunk categories."""
    if kind == "paragraph":
        return "paragraph_group"
    if kind == "spreadsheet_row":
        return "spreadsheet_group"
    return kind
