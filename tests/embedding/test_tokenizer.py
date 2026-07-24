"""Tests for exact embedding token counts and source-text windows."""

from __future__ import annotations

from typing import Any

import pytest

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.embedding.tokenizer import EmbeddingTokenizer


class CharacterTokenizer:
    """Minimal fast tokenizer backend with deterministic character offsets."""

    is_fast = True

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Represent every non-whitespace character as one token."""
        token_ids = [index for index, character in enumerate(text) if not character.isspace()]
        return [0, *token_ids, 1] if add_special_tokens else token_ids

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return the two wrapper tokens used by this test tokenizer."""
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
        """Return overflowing character-token windows in Transformers format."""
        assert add_special_tokens
        assert truncation
        assert return_overflowing_tokens
        assert return_offsets_mapping
        offsets = [(index, index + 1) for index, character in enumerate(text) if not character.isspace()]
        content_budget = max_length - self.num_special_tokens_to_add()
        input_windows: list[list[int]] = []
        offset_windows: list[list[tuple[int, int]]] = []
        start = 0
        while start < len(offsets):
            end = min(start + content_budget, len(offsets))
            content_offsets = offsets[start:end]
            input_windows.append([0, *range(start, end), 1])
            offset_windows.append([(0, 0), *content_offsets, (0, 0)])
            if end == len(offsets):
                break
            start = end - stride
        if not offsets:
            input_windows.append([0, 1])
            offset_windows.append([(0, 0), (0, 0)])
        return {"input_ids": input_windows, "offset_mapping": offset_windows}


class SlowCharacterTokenizer(CharacterTokenizer):
    """Tokenizer backend that cannot provide reliable source offsets."""

    is_fast = False


def _profile() -> EmbeddingProfile:
    """Build one profile without loading remote model files."""
    return EmbeddingProfile(
        profile_id="test",
        model_id="test/model",
        model_revision="model-revision",
        tokenizer_id="test/tokenizer",
        tokenizer_revision="tokenizer-revision",
        dimensions=1024,
        max_input_tokens=32,
        query_instruction=None,
    )


def test_count_tokens_includes_model_special_tokens() -> None:
    """Token counts match the final model input rather than content tokens alone."""
    tokenizer = EmbeddingTokenizer(_profile(), CharacterTokenizer())

    assert tokenizer.count_tokens("Điều 5") == 7


def test_embedding_tokenizer_requires_character_offsets() -> None:
    """A slow tokenizer fails before chunk splitting can lose source provenance."""
    with pytest.raises(ValueError, match="does not support character offsets"):
        EmbeddingTokenizer(_profile(), SlowCharacterTokenizer())


def test_window_offsets_preserve_vietnamese_unicode_and_bound_token_counts() -> None:
    """Every forced window remains an exact source slice within its token budget."""
    source_text = "Điều 5 § 2\nMức phạt 1.000.000 đồng, diện tích 20m²."
    tokenizer = EmbeddingTokenizer(_profile(), CharacterTokenizer())

    windows = tokenizer.window_offsets(source_text, max_tokens=12, overlap_tokens=2)

    assert len(windows) > 1
    assert windows[0].start_character == 0
    assert windows[-1].end_character == len(source_text)
    assert all(
        window.text_from(source_text) == source_text[window.start_character : window.end_character]
        for window in windows
    )
    assert all(window.token_count == tokenizer.count_tokens(window.text_from(source_text)) for window in windows)
    assert all(window.token_count <= 12 for window in windows)


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens", "message"),
    [
        (33, 0, "exceeds profile limit"),
        (2, 0, "cannot contain tokenizer special tokens"),
        (12, -1, "Token overlap must be"),
        (12, 10, "Token overlap must be"),
    ],
)
def test_window_offsets_reject_invalid_token_budgets(
    max_tokens: int,
    overlap_tokens: int,
    message: str,
) -> None:
    """Invalid external token settings fail before tokenizer windowing."""
    tokenizer = EmbeddingTokenizer(_profile(), CharacterTokenizer())

    with pytest.raises(ValueError, match=message):
        tokenizer.window_offsets("Nội dung", max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def test_from_profile_loads_pinned_fast_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile loading uses the declared tokenizer ID and immutable revision."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(tokenizer_id: str, **options: Any) -> CharacterTokenizer:
        calls.append((tokenizer_id, options))
        return CharacterTokenizer()

    monkeypatch.setattr("legal_rag.embedding.tokenizer.AutoTokenizer.from_pretrained", from_pretrained)

    tokenizer = EmbeddingTokenizer.from_profile(_profile())

    assert tokenizer.count_tokens("abc") == 5
    assert calls == [("test/tokenizer", {"revision": "tokenizer-revision", "use_fast": True})]
