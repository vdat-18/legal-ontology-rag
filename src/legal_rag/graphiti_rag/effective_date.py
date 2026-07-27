"""Extract a legal document's own commencement date from its chunk text.

Vietnamese legal documents (decrees, circulars, decisions, laws) almost always state their
own effective date in a dedicated article titled "Hiệu lực thi hành" ("effective enforcement"),
using a stock self-referential phrase such as "Nghị định này có hiệu lực thi hành từ ngày 01
tháng 01 năm 2025" or "... kể từ ngày 01/01/2025". Other, unrelated clauses can also contain
the words "hiệu lực" (a license still being valid, a specific sub-provision's own delayed
phase-in date) — anchoring the date search on chunks that contain the numbered article heading
("Điều N. Hiệu lực thi hành") avoids picking up one of those instead of the document's actual
commencement date. Some document types (e.g. "Quyết định") omit the numbered heading, so a
looser fallback pattern is tried across the whole document if no heading is found.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from legal_rag.chunker.models import Chunk

_ARTICLE_HEADING_PATTERN = re.compile(r"Điều\s+\d+\.\s*Hiệu\s+lực\s+thi\s+hành", re.IGNORECASE)

_SELF_REFERENCE = r"(?:Nghị định|Thông tư|Quyết định|Luật|Văn bản)\s+này\s+có\s+hiệu\s+lực(?:\s+thi\s+hành)?"
_WORD_DATE_PATTERN = re.compile(
    _SELF_REFERENCE + r"\s+(?:kể\s+)?từ\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)
_SLASH_DATE_PATTERN = re.compile(
    _SELF_REFERENCE + r"\s+(?:kể\s+)?từ\s+ngày\s+(\d{1,2})/(\d{1,2})/(\d{4})",
    re.IGNORECASE,
)


def _find_date(text: str) -> date | None:
    """Return the first self-referential commencement date found in *text*, if any."""
    for pattern in (_WORD_DATE_PATTERN, _SLASH_DATE_PATTERN):
        match = pattern.search(text)
        if match:
            day, month, year = (int(group) for group in match.groups())
            return date(year, month, day)
    return None


def extract_effective_date(document_chunks: Sequence[Chunk]) -> date | None:
    """Find one document's own commencement date from its chunks, in chunk-index order.

    Args:
        document_chunks: All chunks belonging to a single source document.

    Returns:
        The extracted commencement date, or ``None`` if no confident match was found.
    """
    ordered = sorted(document_chunks, key=lambda chunk: chunk.chunk_index)

    for index, chunk in enumerate(ordered):
        if not _ARTICLE_HEADING_PATTERN.search(chunk.text):
            continue
        combined = chunk.text
        if index + 1 < len(ordered):
            combined += "\n" + ordered[index + 1].text
        found = _find_date(combined)
        if found is not None:
            return found

    for chunk in ordered:
        found = _find_date(chunk.text)
        if found is not None:
            return found

    return None
