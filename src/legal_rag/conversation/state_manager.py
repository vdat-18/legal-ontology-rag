"""In-memory conversation state store."""

from __future__ import annotations

from datetime import UTC, datetime

from legal_rag.conversation.state import ConversationState


class ConversationStateStore:
    """In-memory store for conversation states, replaceable with Redis later."""

    def __init__(self) -> None:
        """Initialize an empty in-memory state store."""
        self._states: dict[str, ConversationState] = {}

    def get(self, conversation_id: str) -> ConversationState | None:
        """Return the conversation state if it exists."""
        return self._states.get(conversation_id)

    def get_or_create(self, conversation_id: str) -> ConversationState:
        """Return existing state or create a new one."""
        if conversation_id not in self._states:
            self._states[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                turns=(),
                created_at=datetime.now(tz=UTC),
            )
        return self._states[conversation_id]

    def update(self, state: ConversationState) -> None:
        """Update the stored conversation state."""
        self._states[state.conversation_id] = state

    def delete(self, conversation_id: str) -> None:
        """Remove a conversation state."""
        self._states.pop(conversation_id, None)
