"""Tokenizer adapter for exact embedding token counts and source-text windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from transformers import AutoTokenizer

from legal_rag.embedding.profiles import EmbeddingProfile


class _TokenizerBackend(Protocol):
    """Tokenizer behavior required by the embedding adapter."""

    is_fast: bool

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode text into model token IDs."""

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return the number of model wrapper tokens for one input."""

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
        """Encode text into bounded windows with source character offsets."""


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """One token-bounded slice of the original text."""

    start_character: int
    end_character: int

    def text_from(self, source_text: str) -> str:
        """Return this window as an exact slice of its source text."""
        return source_text[self.start_character : self.end_character]


class EmbeddingTokenizer:
    """Count and window text with the tokenizer pinned by an embedding profile."""

    def __init__(self, profile: EmbeddingProfile, tokenizer: _TokenizerBackend) -> None:
        """Bind one fast tokenizer to its immutable embedding profile."""
        if not tokenizer.is_fast:
            msg = f"Tokenizer does not support character offsets: {profile.tokenizer_id}"
            raise ValueError(msg)
        self.profile = profile
        self._tokenizer = tokenizer

    @classmethod
    def from_profile(cls, profile: EmbeddingProfile) -> EmbeddingTokenizer:
        """Load the exact tokenizer revision declared by an embedding profile."""
        tokenizer = AutoTokenizer.from_pretrained(
            profile.tokenizer_id,
            revision=profile.tokenizer_revision,
            use_fast=True,
        )
        return cls(profile, cast(_TokenizerBackend, tokenizer))

    def count_tokens(self, text: str) -> int:
        """Count the complete model input, including required special tokens."""
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def fits_within(self, text: str, *, max_tokens: int) -> bool:
        """Return whether model input fits without encoding beyond the decision boundary."""
        self.validate_window_settings(max_tokens=max_tokens, overlap_tokens=0)
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens + 1,
        )
        return len(token_ids) <= max_tokens

    def validate_window_settings(self, *, max_tokens: int, overlap_tokens: int) -> None:
        """Validate external token-window settings against the active profile."""
        special_token_count = self._tokenizer.num_special_tokens_to_add(pair=False)
        content_token_budget = max_tokens - special_token_count
        if max_tokens > self.profile.max_input_tokens:
            msg = f"Token window limit {max_tokens} exceeds profile limit {self.profile.max_input_tokens}"
            raise ValueError(msg)
        if content_token_budget <= 0:
            msg = f"Token window limit {max_tokens} cannot contain tokenizer special tokens"
            raise ValueError(msg)
        if overlap_tokens < 0 or overlap_tokens >= content_token_budget:
            msg = f"Token overlap must be between 0 and {content_token_budget - 1}"
            raise ValueError(msg)

    def window_offsets(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> tuple[TokenWindow, ...]:
        """Return token-bounded character windows without decoding or changing source text."""
        self.validate_window_settings(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens,
            stride=overlap_tokens,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        token_windows = cast(list[list[int]], encoded["input_ids"])
        offset_windows = cast(list[list[tuple[int, int]]], encoded["offset_mapping"])

        spans: list[list[int]] = []
        for _token_ids, offsets in zip(token_windows, offset_windows, strict=True):
            content_offsets = [(start, end) for start, end in offsets if end > start]
            if content_offsets:
                spans.append([content_offsets[0][0], content_offsets[-1][1]])
        if not spans:
            return ()

        spans[0][0] = 0
        spans[-1][1] = len(text)
        for index in range(len(spans) - 1):
            spans[index][1] = max(spans[index][1], spans[index + 1][0])

        windows: list[TokenWindow] = []
        for start_character, end_character in spans:
            window_text = text[start_character:end_character]
            token_count = self.count_tokens(window_text)
            if token_count > max_tokens:
                msg = f"Tokenizer produced a {token_count}-token source slice for a {max_tokens}-token window"
                raise ValueError(msg)
            windows.append(TokenWindow(start_character, end_character))
        return tuple(windows)
