"""Vietnamese lexical tokenization for BM25 retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from unicodedata import normalize

UNICODE_NORMALIZATION_FORM = "NFC"


def lexical_units(text: str) -> list[str]:
    """Split normalized text on Unicode punctuation and separators while preserving letters and digits."""
    normalized = normalize(UNICODE_NORMALIZATION_FORM, text).casefold()
    return "".join(character if character.isalnum() else " " for character in normalized).split()


def vietnamese_legal_tokens(segmented_tokens: Sequence[str]) -> list[str]:
    """Return namespaced words and adjacent word bigrams for Vietnamese legal BM25."""
    word_tokens: list[str] = []
    bigram_tokens: list[str] = []
    previous_word: str | None = None

    for segmented_token in segmented_tokens:
        units = lexical_units(segmented_token)
        if not units:
            previous_word = None
            continue
        word = "_".join(units)
        word_tokens.append(f"w:{word}")
        if previous_word is not None:
            bigram_tokens.append(f"b:{previous_word}|{word}")
        previous_word = word

    return [*word_tokens, *bigram_tokens]
