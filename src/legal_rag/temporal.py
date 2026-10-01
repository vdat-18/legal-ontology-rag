"""Deterministic date extraction and temporal-target selection for evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, timedelta

_EXPLICIT_DATE_PATTERNS = (
    re.compile(r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{4})(?!\d)"),
    re.compile(
        r"(?:ngày|ngay)\s+(?P<day>\d{1,2})\s+(?:tháng|thang)\s+"
        r"(?P<month>\d{1,2})\s+(?:năm|nam)\s+(?P<year>\d{4})",
        re.IGNORECASE,
    ),
)


def extract_explicit_dates(text: str) -> tuple[date, ...]:
    """Return unique complete calendar dates in textual order.

    Invalid date-shaped text is ignored because this helper reads natural-language questions;
    structured ``query_date`` fields are validated separately and fail loudly.
    """
    matches: list[tuple[int, int, tuple[date, ...]]] = []
    for pattern in _EXPLICIT_DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                parsed = date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            except ValueError:
                continue
            matches.append((match.start(), match.end(), _dates_for_context(text, match.start(), parsed)))

    dates: list[date] = []
    occupied_spans: list[tuple[int, int]] = []
    for start, end, parsed_dates in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_spans):
            continue
        occupied_spans.append((start, end))
        for parsed in parsed_dates:
            if parsed not in dates:
                dates.append(parsed)
    return tuple(dates)


def _dates_for_context(text: str, start: int, parsed: date) -> tuple[date, ...]:
    """Convert exclusive-before wording to the preceding day at day granularity."""
    prefix = text[max(0, start - 48) : start].casefold()
    marker = r"(?:(?:ngày|ngay|thời\s+điểm|thoi\s+diem)\s*)?"
    before = r"(?:trước|truoc)"
    if re.search(rf"{before}\s+(?:và|va)\s+sau\s+{marker}$", prefix):
        return (parsed - timedelta(days=1), parsed)
    if re.search(rf"(?:đến|den)?\s*{before}\s+{marker}$", prefix):
        return (parsed - timedelta(days=1),)
    return (parsed,)


def resolve_temporal_targets(
    question: str,
    *,
    query_dates: Sequence[date] = (),
    query_date: date | None = None,
    reference_date: date | None = None,
    temporal_comparison: bool = False,
) -> tuple[date, ...]:
    """Choose reproducible dates for retrieval without collapsing temporal comparisons.

    Persisted plural dates take precedence over the legacy singular field. For an explicitly
    temporal comparison that lacks structured dates, a single date mentioned in the question is
    insufficient to select every compared version, so an empty result deliberately means an
    unfiltered temporal search. Ordinary current-law questions fall back to ``reference_date``.
    """
    if query_dates:
        return tuple(query_dates)
    if query_date is not None:
        return (query_date,)

    explicit_dates = extract_explicit_dates(question)
    if temporal_comparison:
        return explicit_dates if len(explicit_dates) > 1 else ()
    if explicit_dates:
        return explicit_dates
    return (reference_date,) if reference_date is not None else ()
