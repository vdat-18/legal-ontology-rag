"""Loader tests against the real corpus zip (offline, no Neo4j/network).

Verifies that in-scope penalty documents yield time-stamped episodes and that
the core decree carries its curated effective date. Skipped when the corpus is
unavailable (e.g. a checkout without the data archive).
"""

from __future__ import annotations

import pytest

from legal_rag.config import Settings
from legal_rag.ingestion import iter_episodes


@pytest.fixture
def settings() -> Settings:
    """Settings whose corpus source (zip or extracted dir) actually exists."""
    resolved = Settings.from_env()
    if not resolved.corpus_zip.exists() and not resolved.corpus_dir.exists():
        pytest.skip("corpus not available")
    return resolved


def test_yields_timestamped_episodes(settings: Settings) -> None:
    """Every episode carries a timezone-aware reference time."""
    episodes = list(iter_episodes(settings))
    assert len(episodes) > 100
    assert all(e.reference_time.tzinfo is not None for e in episodes)
    assert all(e.body.strip() for e in episodes)


def test_core_decree_effective_date(settings: Settings) -> None:
    """NĐ 168/2024 episodes use the curated 2025-01-01 effective date."""
    episodes = [e for e in iter_episodes(settings) if e.name.startswith("168/2024")]
    assert episodes, "NĐ 168/2024 must yield episodes"
    assert all(e.reference_time.date().isoformat() == "2025-01-01" for e in episodes)
