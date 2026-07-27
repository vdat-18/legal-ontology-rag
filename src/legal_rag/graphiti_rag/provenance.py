"""Flag legal document citations that reference documents outside the indexed corpus.

Graphiti can correctly extract a real cross-reference from a document's own text (e.g. "168/2024/
NĐ-CP sửa đổi một số điều của Nghị định số 100/2019/NĐ-CP") into a graph fact or node — this is
not a hallucination, but it does mean the *referenced* document (100/2019/NĐ-CP here) may not
itself be indexed, so any of its un-amended provisions are invisible to this graph. Surfacing
this distinction lets an answer be labeled "may be incomplete" rather than silently presented
as fully grounded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_DOCUMENT_NUMBER_PATTERN = re.compile(r"\d{1,5}/\d{4}/[A-ZĐ][A-Za-zĐ\-]*")


def find_unindexed_document_citations(texts: Sequence[str], known_so_hieu: frozenset[str]) -> list[str]:
    """Return legal document numbers mentioned in *texts* that aren't in *known_so_hieu*.

    Args:
        texts: Fact or answer text to scan for document-number-shaped substrings.
        known_so_hieu: The ``so_hieu`` values of documents actually indexed into this corpus.

    Returns:
        Sorted, deduplicated document numbers found in *texts* but absent from *known_so_hieu*.
    """
    mentioned: set[str] = set()
    for text in texts:
        mentioned.update(_DOCUMENT_NUMBER_PATTERN.findall(text))
    return sorted(mentioned - known_so_hieu)
