"""Tests for shared query rewriting module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from legal_rag.query_rewriting import AsyncQueryRewriter, RewrittenQuery, SyncQueryRewriter
from legal_rag.settings import Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Return local settings suitable for mocked rewrite calls."""
    return Settings.from_environment(
        {
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_TIMEOUT_SECONDS": "30",
        }
    )


class TestSyncQueryRewriter:
    """Cover the synchronous query-rewriting boundary."""

    def test_init_creates_client(self, mock_settings: Settings) -> None:
        """The synchronous wrapper constructs its transport caller."""
        rewriter = SyncQueryRewriter(mock_settings)
        assert rewriter._caller is not None

    def test_rewrite_returns_rewritten_query(self, mock_settings: Settings) -> None:
        """A structured model response becomes a rewritten query."""
        rewriter = SyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            result = rewriter.rewrite("vượt đèn đỏ bị phạt bao nhiêu?", ())

            assert isinstance(result, RewrittenQuery)
            assert result.original == "vượt đèn đỏ bị phạt bao nhiêu?"
            assert "đèn tín hiệu giao thông" in result.rewritten

    def test_rewrite_with_history(self, mock_settings: Settings) -> None:
        """Conversation history is accepted by the synchronous wrapper."""
        rewriter = SyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            history = (("Ô tô vượt đèn đỏ phạt bao nhiêu?", "Phạt 18-20 triệu đồng."),)
            result = rewriter.rewrite("còn xe máy thì sao?", history)

            assert isinstance(result, RewrittenQuery)
            assert result.original == "còn xe máy thì sao?"


class TestAsyncQueryRewriter:
    """Cover the asynchronous query-rewriting boundary."""

    def test_init_creates_client(self, mock_settings: Settings) -> None:
        """The asynchronous wrapper constructs its transport caller."""
        rewriter = AsyncQueryRewriter(mock_settings)
        assert rewriter._caller is not None

    @pytest.mark.anyio
    async def test_rewrite_returns_rewritten_query(self, mock_settings: Settings) -> None:
        """A structured async response becomes a rewritten query."""
        rewriter = AsyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute_async") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            result = await rewriter.rewrite("vượt đèn đỏ bị phạt bao nhiêu?", ())

            assert isinstance(result, RewrittenQuery)
            assert result.original == "vượt đèn đỏ bị phạt bao nhiêu?"
            assert "đèn tín hiệu giao thông" in result.rewritten

    @pytest.mark.anyio
    async def test_rewrite_with_history(self, mock_settings: Settings) -> None:
        """Conversation history is accepted by the asynchronous wrapper."""
        rewriter = AsyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute_async") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            history = (("Ô tô vượt đèn đỏ phạt bao nhiêu?", "Phạt 18-20 triệu đồng."),)
            result = await rewriter.rewrite("còn xe máy thì sao?", history)

            assert isinstance(result, RewrittenQuery)
            assert result.original == "còn xe máy thì sao?"


class TestRewrittenQuery:
    """Cover the immutable rewritten-query value object."""

    def test_dataclass_is_frozen(self) -> None:
        """Rewritten query fields cannot be reassigned."""
        result = RewrittenQuery(original="test", rewritten="test rewritten")
        with pytest.raises(AttributeError):
            result.original = "changed"

    def test_dataclass_has_slots(self) -> None:
        """The value object uses slots."""
        result = RewrittenQuery(original="test", rewritten="test rewritten")
        assert hasattr(result, "__slots__")
