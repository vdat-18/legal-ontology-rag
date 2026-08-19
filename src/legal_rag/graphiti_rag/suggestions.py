"""Suggest what a reader is likely to ask next, from the answer they just received."""

from __future__ import annotations

from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import BamlCaller
from legal_rag.settings import GraphitiSettings, Settings

_SUGGESTION_COUNT = 3


class FollowUpClient(Protocol):
    """Boundary implemented by a follow-up-question-suggesting language-model client."""

    def suggest(self, question: str, answer: str) -> list[str]:
        """Return questions the reader is likely to ask after *answer*."""


class BamlFollowUpClient:
    """Typed BAML client, backed by the shared runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, graphiti_settings: GraphitiSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=graphiti_settings.follow_up_max_output_tokens,
            temperature=graphiti_settings.follow_up_temperature,
            max_response_bytes=graphiti_settings.max_response_bytes,
        )

    def suggest(self, question: str, answer: str) -> list[str]:
        """Run one bounded BAML call and return the suggested follow-up questions.

        The model is asked for exactly three, but the count is enforced here as well: a chat UI
        lays them out in a fixed row, so an extra suggestion would overflow it and a missing one
        would leave a gap.
        """
        request = b.request.SuggestFollowUpQuestions(question, answer, baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        parsed = b.parse.SuggestFollowUpQuestions(raw_output, baml_options=self._caller.options)
        return [suggestion.strip() for suggestion in parsed.questions if suggestion.strip()][:_SUGGESTION_COUNT]
