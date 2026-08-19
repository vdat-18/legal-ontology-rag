"""Tests for suggesting the questions a reader is likely to ask next."""

from __future__ import annotations

from legal_rag.graphiti_rag.suggestions import BamlFollowUpClient


class _FakeParsed:
    """Stand in for the parsed BAML result."""

    def __init__(self, questions: list[str]) -> None:
        self.questions = questions


def test_blank_suggestions_are_dropped(monkeypatch) -> None:
    """A model that pads its list with empty strings must not produce empty chips."""
    client = BamlFollowUpClient.__new__(BamlFollowUpClient)
    monkeypatch.setattr(client, "_caller", _FakeCaller(), raising=False)
    monkeypatch.setattr(
        "legal_rag.graphiti_rag.suggestions.b.parse.SuggestFollowUpQuestions",
        lambda *_args, **_kwargs: _FakeParsed(["Câu một", "   ", "Câu hai"]),
    )
    monkeypatch.setattr(
        "legal_rag.graphiti_rag.suggestions.b.request.SuggestFollowUpQuestions",
        lambda *_args, **_kwargs: object(),
    )

    assert client.suggest("q", "a") == ["Câu một", "Câu hai"]


def test_at_most_three_suggestions_are_returned(monkeypatch) -> None:
    """The chat lays the chips out in one row, so a longer list is truncated."""
    client = BamlFollowUpClient.__new__(BamlFollowUpClient)
    monkeypatch.setattr(client, "_caller", _FakeCaller(), raising=False)
    monkeypatch.setattr(
        "legal_rag.graphiti_rag.suggestions.b.parse.SuggestFollowUpQuestions",
        lambda *_args, **_kwargs: _FakeParsed(["một", "hai", "ba", "bốn", "năm"]),
    )
    monkeypatch.setattr(
        "legal_rag.graphiti_rag.suggestions.b.request.SuggestFollowUpQuestions",
        lambda *_args, **_kwargs: object(),
    )

    assert client.suggest("q", "a") == ["một", "hai", "ba"]


class _FakeCaller:
    """Return raw output without contacting a provider."""

    options: dict[str, object] = {}  # noqa: RUF012

    def execute(self, _request: object) -> str:
        """Return an unused raw payload; parsing is stubbed in the tests."""
        return "{}"
