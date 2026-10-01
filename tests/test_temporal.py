"""Tests for deterministic temporal target selection."""

from __future__ import annotations

from datetime import date

from legal_rag.temporal import extract_explicit_dates, resolve_temporal_targets


def test_extract_explicit_dates_supports_legal_question_formats_in_text_order() -> None:
    """Complete Vietnamese and ISO date forms are normalized deterministically."""
    question = "Tại 2025-01-01, ngày 2 tháng 3 năm 2026 và 15/08/2026 thì quy định nào áp dụng?"

    assert extract_explicit_dates(question) == (
        date(2025, 1, 1),
        date(2026, 3, 2),
        date(2026, 8, 15),
    )


def test_extract_explicit_dates_respects_exclusive_before_day_boundary() -> None:
    """An exclusive upper bound selects the preceding day rather than the new version."""
    assert extract_explicit_dates("Quy định áp dụng trước ngày 01/01/2025?") == (date(2024, 12, 31),)
    assert extract_explicit_dates("So sánh trước và sau ngày 01/01/2025") == (
        date(2024, 12, 31),
        date(2025, 1, 1),
    )


def test_resolve_temporal_targets_preserves_structured_versions_and_safe_fallbacks() -> None:
    """Structured targets win, unknown comparisons stay unfiltered, and current queries freeze."""
    structured = (date(2025, 1, 1), date(2026, 3, 1))

    assert (
        resolve_temporal_targets(
            "comparison",
            query_dates=structured,
            reference_date=date(2026, 10, 1),
            temporal_comparison=True,
        )
        == structured
    )
    assert (
        resolve_temporal_targets(
            "So sánh quy định trước và sau khi sửa đổi",
            reference_date=date(2026, 10, 1),
            temporal_comparison=True,
        )
        == ()
    )
    assert resolve_temporal_targets("Quy định hiện nay?", reference_date=date(2026, 10, 1)) == (date(2026, 10, 1),)
