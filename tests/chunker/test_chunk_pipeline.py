"""Tests for deterministic fragment packing and chunk links."""

from __future__ import annotations

from legal_rag.chunker.models import ChunkFragment, ChunkKind, SourceScope, StructuralContextEntry
from legal_rag.chunker.pipeline import pack_scope_fragments
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.embedding.tokenizer import EmbeddingTokenizer


class CharacterTokenizer:
    """Fast test tokenizer that maps each character to one token."""

    is_fast = True

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode characters with two optional wrapper tokens."""
        token_ids = list(range(len(text)))
        if add_special_tokens:
            token_ids = [0, *token_ids, 1]
        return token_ids[:max_length] if truncation and max_length is not None else token_ids

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return wrapper-token counts for single or paired inputs."""
        return 3 if pair else 2


def _tokenizer() -> EmbeddingTokenizer:
    """Build a deterministic tokenizer without external model downloads."""
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


def _scope(scope_id: str = "scope-root", *, child: bool = False) -> SourceScope:
    """Build one root or attachment-child source scope."""
    return SourceScope(
        scope_id=scope_id,
        source_name="appendix.docx" if child else "source.docx",
        source_sha256="member-hash" if child else "source-hash",
        record_scope="attachment_child" if child else "root",
        parent_attachment_id="attachment-id" if child else None,
        attachment_member_path="appendix.docx" if child else None,
        attachment_member_sha256="member-hash" if child else None,
        records=(),
    )


def _fragment(
    index: int,
    text: str,
    *,
    kind: ChunkKind = "paragraph_group",
    packing_key: tuple[str, ...] = (),
    structural_context: tuple[StructuralContextEntry, ...] = (),
) -> ChunkFragment:
    """Build one bounded fragment with stable provenance."""
    return ChunkFragment(
        kind=kind,
        packing_key=packing_key,
        sequence_start=index,
        sequence_end=index,
        unit_ids=(f"unit-{index}",),
        source_refs=(f"source/{index}",),
        text=text,
        structural_context=structural_context,
    )


def _pack(scope: SourceScope, fragments: tuple[ChunkFragment, ...], *, max_tokens: int = 20):
    """Pack test fragments with the shared deterministic settings."""
    return pack_scope_fragments(
        scope,
        fragments,
        _tokenizer(),
        max_tokens=max_tokens,
        split_overlap_tokens=2,
    )


def test_paragraphs_pack_greedily_and_flush_at_token_boundary() -> None:
    """Adjacent paragraphs fill the current chunk without exceeding its budget."""
    fragments = (
        _fragment(0, "Điều 1"),
        _fragment(1, "Khoản 1"),
        _fragment(2, "Nội dung dài"),
    )

    chunks = _pack(_scope(), fragments)

    assert [chunk.text for chunk in chunks] == ["Điều 1\n\nKhoản 1", "Nội dung dài"]
    assert chunks[0].unit_ids == ("unit-0", "unit-1")
    assert chunks[0].sequence_start == 0
    assert chunks[0].sequence_end == 1
    assert all(chunk.text_token_count <= 20 for chunk in chunks)


def test_structural_boundaries_control_compatible_packing() -> None:
    """Tables and sheets pack only within their native container while images stay independent."""
    fragments = (
        _fragment(0, "row-a", kind="table", packing_key=("table-a",)),
        _fragment(1, "row-b", kind="table", packing_key=("table-a",)),
        _fragment(2, "row-c", kind="table", packing_key=("table-b",)),
        _fragment(3, "sheet-a-1", kind="spreadsheet_group", packing_key=("Sheet A",)),
        _fragment(4, "sheet-a-2", kind="spreadsheet_group", packing_key=("Sheet A",)),
        _fragment(5, "image-a", kind="image"),
        _fragment(6, "image-b", kind="image"),
    )

    chunks = _pack(_scope(), fragments, max_tokens=30)

    assert [chunk.text for chunk in chunks] == [
        "row-a\n\nrow-b",
        "row-c",
        "sheet-a-1\n\nsheet-a-2",
        "image-a",
        "image-b",
    ]
    assert [chunk.kind for chunk in chunks] == [
        "table",
        "table",
        "spreadsheet_group",
        "image",
        "image",
    ]


def test_chunk_ids_links_and_output_are_deterministic() -> None:
    """Repeated packing produces identical IDs, indexes, hashes, and neighbor links."""
    fragments = (
        _fragment(0, "A"),
        _fragment(1, "B", kind="image"),
        _fragment(2, "C"),
    )

    first = _pack(_scope(), fragments)
    second = _pack(_scope(), fragments)

    assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]
    assert [chunk.chunk_index for chunk in first] == [0, 1, 2]
    assert first[0].previous_chunk_id is None
    assert first[0].next_chunk_id == first[1].chunk_id
    assert first[1].previous_chunk_id == first[0].chunk_id
    assert first[1].next_chunk_id == first[2].chunk_id
    assert first[2].previous_chunk_id == first[1].chunk_id
    assert first[2].next_chunk_id is None
    assert all(chunk.embedding_text == chunk.text for chunk in first)
    assert all(chunk.embedding_token_count == chunk.text_token_count for chunk in first)


def test_scopes_have_distinct_ids_metadata_and_neighbor_chains() -> None:
    """Identical content in root and attachment streams never shares identity or links."""
    root_chunks = _pack(_scope(), (_fragment(0, "Nội dung"),))
    child_scope = _scope("scope-child", child=True)
    child_chunks = _pack(child_scope, (_fragment(0, "Nội dung"),))

    assert root_chunks[0].chunk_id != child_chunks[0].chunk_id
    assert root_chunks[0].scope_id == "scope-root"
    assert child_chunks[0].scope_id == "scope-child"
    assert child_chunks[0].record_scope == "attachment_child"
    assert child_chunks[0].parent_attachment_id == "attachment-id"
    assert root_chunks[0].previous_chunk_id is None
    assert root_chunks[0].next_chunk_id is None
    assert child_chunks[0].previous_chunk_id is None
    assert child_chunks[0].next_chunk_id is None


def test_boundary_configuration_changes_config_and_chunk_identity() -> None:
    """A token-budget change creates a distinct deterministic configuration identity."""
    fragments = (_fragment(0, "Nội dung"),)

    smaller = _pack(_scope(), fragments, max_tokens=20)[0]
    larger = _pack(_scope(), fragments, max_tokens=30)[0]

    assert smaller.chunker_config_sha256 != larger.chunker_config_sha256
    assert smaller.chunk_id != larger.chunk_id


def test_chunks_render_single_or_boundary_spanning_structural_context() -> None:
    """A packed chunk retains every article ancestry when it crosses a boundary."""
    article_six = (
        StructuralContextEntry(
            level="article",
            text="Điều 6. Xử phạt người điều khiển xe ô tô",
            source_refs=("source/article-6",),
        ),
    )
    shared = _pack(
        _scope(),
        (
            _fragment(0, "a)", structural_context=article_six),
            _fragment(1, "b)", structural_context=article_six),
        ),
    )[0]
    article_seven = (
        StructuralContextEntry(
            level="article",
            text="Điều 7. Xử phạt người điều khiển xe mô tô",
            source_refs=("source/article-7",),
        ),
    )
    mixed = _pack(
        _scope(),
        (
            _fragment(0, "d)", structural_context=article_six),
            _fragment(1, "Điều 7", structural_context=article_seven),
        ),
    )[0]

    assert shared.context_text == "\n".join(
        (
            "[LEGAL_STRUCTURE 1.0.0]",
            "Điều 6. Xử phạt người điều khiển xe ô tô",
            "[/LEGAL_STRUCTURE]",
        )
    )
    assert mixed.context_text == "\n".join(
        (
            "[LEGAL_STRUCTURE_VARIANTS 1.0.0]",
            "[VARIANT 1]",
            "Điều 6. Xử phạt người điều khiển xe ô tô",
            "[/VARIANT]",
            "[VARIANT 2]",
            "Điều 7. Xử phạt người điều khiển xe mô tô",
            "[/VARIANT]",
            "[/LEGAL_STRUCTURE_VARIANTS]",
        )
    )
