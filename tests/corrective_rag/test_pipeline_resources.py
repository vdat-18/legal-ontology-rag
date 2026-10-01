"""Resource lifecycle tests for Corrective RAG pipeline construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from legal_rag.corrective_rag.generate import pipeline as pipeline_module
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline


def test_constructor_closes_created_resources_in_reverse_order_without_hiding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late constructor failure closes prior resources and preserves the primary error."""
    events: list[str] = []
    setup_error = ValueError("CRAG_WEB_SEARCH_ALLOWED_DOMAINS must not be empty")

    class _Resource:
        """Record creation and cleanup for one pipeline dependency."""

        def __init__(self, name: str, *, fail_close: bool = False) -> None:
            self.name = name
            self.fail_close = fail_close
            events.append(f"create:{name}")

        def close(self) -> None:
            """Record cleanup and optionally simulate a secondary failure."""
            events.append(f"close:{self.name}")
            if self.fail_close:
                msg = f"{self.name} cleanup failed"
                raise RuntimeError(msg)

    class _FailingWebSearch:
        """Simulate allowlist validation failing late in pipeline construction."""

        def __init__(self, _config: object) -> None:
            events.append("create:web-search")
            raise setup_error

    monkeypatch.setattr(pipeline_module, "EmbeddingModel", lambda _profile: _Resource("embedder"))
    monkeypatch.setattr(pipeline_module, "LegalVectorDB", lambda _config, _embedder: _Resource("database"))
    monkeypatch.setattr(pipeline_module, "CorrectiveGrader", lambda _config: _Resource("grader"))
    monkeypatch.setattr(
        pipeline_module,
        "ResponseGenerator",
        lambda _config: _Resource("generator", fail_close=True),
    )
    monkeypatch.setattr(pipeline_module, "WebSearchProvider", _FailingWebSearch)
    config = SimpleNamespace(
        embedding_profile="test-profile",
        evaluator_lower_threshold=0.35,
        evaluator_upper_threshold=0.70,
    )

    with pytest.raises(ValueError, match="CRAG_WEB_SEARCH_ALLOWED_DOMAINS") as captured:
        CorrectiveRAGPipeline(config)

    assert captured.value is setup_error
    assert events == [
        "create:embedder",
        "create:database",
        "create:grader",
        "create:generator",
        "create:web-search",
        "close:generator",
        "close:grader",
        "close:database",
        "close:embedder",
    ]
    assert setup_error.__notes__ == [
        "Corrective RAG constructor cleanup failed: RuntimeError('generator cleanup failed')"
    ]


def test_close_releases_every_resource_and_preserves_the_first_failure() -> None:
    """A failed close callback cannot prevent later resources from being released."""
    events: list[str] = []
    first_error = RuntimeError("last-created cleanup failed")

    def close(name: str, error: BaseException | None = None) -> None:
        events.append(name)
        if error is not None:
            raise error

    pipeline = object.__new__(CorrectiveRAGPipeline)
    pipeline._close_callbacks = (
        lambda: close("first-created"),
        lambda: close("middle", ValueError("middle cleanup failed")),
        lambda: close("last-created", first_error),
    )

    with pytest.raises(RuntimeError, match="last-created cleanup failed") as captured:
        pipeline.close()

    assert captured.value is first_error
    assert events == ["last-created", "middle", "first-created"]
    assert first_error.__notes__ == ["Additional Corrective RAG cleanup failure: ValueError('middle cleanup failed')"]
