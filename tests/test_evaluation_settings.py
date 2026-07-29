"""Tests for ground-truth synthesis environment settings."""

from __future__ import annotations

from legal_rag.settings import _EVALUATION_TUNABLES, EvaluationSettings


def test_evaluation_settings_load_from_environment() -> None:
    """Ground-truth synthesis settings parse from process environment variables, defaulted otherwise."""
    settings = EvaluationSettings.from_environment({"EMBEDDING_PROFILE": "qwen3_0_6b"})

    assert settings.embedding_profile == "qwen3_0_6b"
    assert settings.span_match_threshold == 0.85
    assert settings.bundle_overlap_threshold == 0.5
    assert settings.max_quality_retries == 2


def test_every_declared_tunable_is_overridable_from_the_environment() -> None:
    """No declared tunable can be silently ignored: each maps to a GROUNDTRUTH_<FIELD_NAME> variable."""
    base = {"EMBEDDING_PROFILE": "qwen3_0_6b"}
    defaults = EvaluationSettings.from_environment(base)

    for name, converter in _EVALUATION_TUNABLES.items():
        default = getattr(defaults, name)
        overridden = converter(str(default + 1))
        settings = EvaluationSettings.from_environment({**base, f"GROUNDTRUTH_{name.upper()}": str(overridden)})

        assert getattr(settings, name) == overridden, name
