"""Tests for conversation state management."""

from __future__ import annotations

from datetime import UTC, datetime

from legal_rag.conversation.state import ConversationState, ConversationTurn
from legal_rag.conversation.state_manager import ConversationStateStore


def test_conversation_turn_to_dict() -> None:
    """ConversationTurn serializes to JSON-compatible dict."""
    turn = ConversationTurn(
        query="Mức phạt xe đạp?",
        rewritten_query="Mức phạt người điều khiển xe đạp không chấp hành hiệu lệnh?",
        retrieved_chunk_ids=("chunk1", "chunk2"),
        answer="Phạt 150.000-250.000 đồng.",
        timestamp=datetime(2026, 9, 6, 23, 0, 0, tzinfo=UTC),
    )

    result = turn.to_dict()

    assert result["query"] == "Mức phạt xe đạp?"
    assert result["rewritten_query"] == "Mức phạt người điều khiển xe đạp không chấp hành hiệu lệnh?"
    assert result["retrieved_chunk_ids"] == ["chunk1", "chunk2"]
    assert result["answer"] == "Phạt 150.000-250.000 đồng."
    assert result["timestamp"] == "2026-09-06T23:00:00+00:00"


def test_conversation_state_add_turn() -> None:
    """ConversationState can add a turn and return a new immutable state."""
    created_at = datetime(2026, 9, 6, 22, 0, 0, tzinfo=UTC)
    state = ConversationState(
        conversation_id="conv-123",
        turns=(),
        created_at=created_at,
    )

    turn = ConversationTurn(
        query="Mức phạt xe đạp?",
        rewritten_query="Mức phạt xe đạp không chấp hành hiệu lệnh?",
        retrieved_chunk_ids=("chunk1",),
        answer="Phạt 150.000 đồng.",
        timestamp=datetime(2026, 9, 6, 22, 5, 0, tzinfo=UTC),
    )

    new_state = state.add_turn(turn)

    assert len(state.turns) == 0
    assert len(new_state.turns) == 1
    assert new_state.turns[0].query == "Mức phạt xe đạp?"
    assert new_state.conversation_id == "conv-123"
    assert new_state.created_at == created_at


def test_state_store_get_or_create() -> None:
    """ConversationStateStore creates a new state on first access."""
    store = ConversationStateStore()

    state = store.get_or_create("conv-456")

    assert state.conversation_id == "conv-456"
    assert len(state.turns) == 0


def test_state_store_update() -> None:
    """ConversationStateStore updates existing state."""
    store = ConversationStateStore()
    state = store.get_or_create("conv-789")

    turn = ConversationTurn(
        query="Test query",
        rewritten_query="Test rewritten query",
        retrieved_chunk_ids=("chunk1",),
        answer="Test answer",
        timestamp=datetime.now(tz=UTC),
    )
    updated = state.add_turn(turn)

    store.update(updated)

    retrieved = store.get("conv-789")
    assert retrieved is not None
    assert len(retrieved.turns) == 1
    assert retrieved.turns[0].query == "Test query"


def test_state_store_delete() -> None:
    """ConversationStateStore can delete a state."""
    store = ConversationStateStore()
    store.get_or_create("conv-delete")

    store.delete("conv-delete")

    assert store.get("conv-delete") is None
