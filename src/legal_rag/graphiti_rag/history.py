"""Per-session conversation history, so a follow-up question keeps its context."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

import redis

from legal_rag.baml_client.types import ConversationTurn

if TYPE_CHECKING:
    from legal_rag.settings import RedisSettings


class ConversationHistory(Protocol):
    """Boundary implemented by a store of recent turns for one chat session."""

    def load(self, session_id: str) -> list[ConversationTurn]:
        """Return the session's recent turns, oldest first."""

    def append(self, session_id: str, question: str, answer: str) -> None:
        """Record one completed turn against the session."""

    def close(self) -> None:
        """Release whatever connection the store holds."""


class RedisConversationHistory:
    """Conversation history in Redis, bounded per session and expiring when a chat goes idle.

    The list is trimmed to ``max_turns`` and given a TTL on every write: a chat UI can open any
    number of sessions, and without both bounds an abandoned one would keep its turns forever and
    an active one would grow until the whole transcript was replayed into each rewrite prompt.
    """

    def __init__(self, settings: RedisSettings) -> None:
        """Connect to Redis using the shared connection settings."""
        self._client = redis.Redis.from_url(
            settings.url,
            password=settings.password or None,
            socket_timeout=settings.socket_timeout_seconds,
            socket_connect_timeout=settings.socket_timeout_seconds,
        )
        self._max_turns = settings.max_turns
        self._ttl_seconds = settings.ttl_seconds

    def _key(self, session_id: str) -> str:
        """Return the Redis key holding one session's turns."""
        return f"graphiti:history:{session_id}"

    def load(self, session_id: str) -> list[ConversationTurn]:
        """Return the session's recent turns, oldest first."""
        raw_turns: list[bytes] = self._client.lrange(self._key(session_id), 0, -1)
        return [ConversationTurn(**json.loads(raw.decode("utf-8"))) for raw in raw_turns]

    def append(self, session_id: str, question: str, answer: str) -> None:
        """Record one completed turn, trimming and refreshing the session's expiry."""
        key = self._key(session_id)
        self._client.rpush(key, json.dumps({"question": question, "answer": answer}, ensure_ascii=False))
        self._client.ltrim(key, -self._max_turns, -1)
        self._client.expire(key, self._ttl_seconds)

    def close(self) -> None:
        """Release the Redis connection."""
        self._client.close()
