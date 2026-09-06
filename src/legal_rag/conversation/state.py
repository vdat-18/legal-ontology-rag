"""Typed conversation state for multi-turn retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One turn in a multi-turn conversation."""

    query: str
    rewritten_query: str
    retrieved_chunk_ids: tuple[str, ...]
    answer: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable conversation turn record."""
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ConversationState:
    """State for one multi-turn conversation."""

    conversation_id: str
    turns: tuple[ConversationTurn, ...]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable conversation state record."""
        return {
            "conversation_id": self.conversation_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "created_at": self.created_at.isoformat(),
        }

    def add_turn(self, turn: ConversationTurn) -> ConversationState:
        """Return a new state with an additional turn."""
        return ConversationState(
            conversation_id=self.conversation_id,
            turns=(*self.turns, turn),
            created_at=self.created_at,
        )
