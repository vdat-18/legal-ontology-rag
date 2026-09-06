"""Multi-turn retriever extending the existing single-turn Retriever."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger
from pymilvus import MilvusClient

from legal_rag.baml_client.types import ConversationTurn as BamlConversationTurn
from legal_rag.benchmark_synthesis.vietnamese_lexical import VietnameseWordSegmenter
from legal_rag.conversation.query_rewriter import BamlMultiTurnQueryRewriter
from legal_rag.conversation.state import ConversationTurn
from legal_rag.conversation.state_manager import ConversationStateStore
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever, SearchMode
from legal_rag.settings import NaiveRagSettings, Settings


class MultiTurnRetriever(Retriever):
    """Extend Retriever with conversation state management and query rewriting."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        milvus_client: MilvusClient,
        app_settings: Settings,
        rag_settings: NaiveRagSettings,
        *,
        word_segmenter: VietnameseWordSegmenter | None = None,
        max_history_turns: int = 5,
    ) -> None:
        """Initialize retriever with conversation state manager and query rewriter."""
        super().__init__(encoder, milvus_client, rag_settings, word_segmenter=word_segmenter)
        self._state_store = ConversationStateStore()
        self._query_rewriter = BamlMultiTurnQueryRewriter(
            app_settings,
            max_output_tokens=512,
            temperature=0.0,
        )
        self._max_history_turns = max_history_turns

    def retrieve_multi_turn(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 10,
        mode: SearchMode | None = None,
    ) -> tuple[list[RetrievedChunk], str]:
        """Retrieve chunks for a multi-turn query, rewriting with conversation context.

        Returns both the retrieved chunks and the rewritten query for downstream use.
        """
        state = self._state_store.get_or_create(conversation_id)
        history_turns = self._get_recent_history(state.turns)

        rewritten_query = self._query_rewriter.rewrite(query, history_turns)

        logger.debug(
            "Rewrote multi-turn query: '{}' -> '{}'",
            query,
            rewritten_query,
        )

        chunks = self.retrieve(rewritten_query, top_k=top_k, mode=mode)

        turn = ConversationTurn(
            query=query,
            rewritten_query=rewritten_query,
            retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            answer="",
            timestamp=datetime.now(tz=UTC),
        )
        self._state_store.update(state.add_turn(turn))

        return chunks, rewritten_query

    def finalize_turn(self, conversation_id: str, answer: str) -> None:
        """Update the last turn's answer after generation completes."""
        state = self._state_store.get(conversation_id)
        if not state or not state.turns:
            return

        last_turn = state.turns[-1]
        updated_turn = ConversationTurn(
            query=last_turn.query,
            rewritten_query=last_turn.rewritten_query,
            retrieved_chunk_ids=last_turn.retrieved_chunk_ids,
            answer=answer,
            timestamp=last_turn.timestamp,
        )

        new_turns = (*state.turns[:-1], updated_turn)
        self._state_store.update(
            state.__class__(
                conversation_id=state.conversation_id,
                turns=new_turns,
                created_at=state.created_at,
            )
        )

    def _get_recent_history(self, turns: tuple[ConversationTurn, ...]) -> list[BamlConversationTurn]:
        """Return the most recent N turns as BAML conversation history."""
        recent = turns[-self._max_history_turns :] if len(turns) > self._max_history_turns else turns
        return [
            BamlConversationTurn(question=turn.query, answer=turn.answer)
            for turn in recent
            if turn.answer
        ]
