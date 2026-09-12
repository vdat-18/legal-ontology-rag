"""Tests for shared query rewriting module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from legal_rag.query_rewriting import AsyncQueryRewriter, RewrittenQuery, SyncQueryRewriter
from legal_rag.settings import Settings


@pytest.fixture
def mock_settings() -> Settings:
    return Settings.from_environment(
        {
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_TIMEOUT_SECONDS": "30",
        }
    )


class TestSyncQueryRewriter:
    def test_init_creates_client(self, mock_settings: Settings) -> None:
        rewriter = SyncQueryRewriter(mock_settings)
        assert rewriter._caller is not None

    def test_rewrite_returns_rewritten_query(self, mock_settings: Settings) -> None:
        rewriter = SyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            result = rewriter.rewrite("vượt đèn đỏ bị phạt bao nhiêu?", ())

            assert isinstance(result, RewrittenQuery)
            assert result.original == "vượt đèn đỏ bị phạt bao nhiêu?"
            assert "đèn tín hiệu giao thông" in result.rewritten

    def test_rewrite_with_history(self, mock_settings: Settings) -> None:
        rewriter = SyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            history = (("Ô tô vượt đèn đỏ phạt bao nhiêu?", "Phạt 18-20 triệu đồng."),)
            result = rewriter.rewrite("còn xe máy thì sao?", history)

            assert isinstance(result, RewrittenQuery)
            assert result.original == "còn xe máy thì sao?"


class TestAsyncQueryRewriter:
    def test_init_creates_client(self, mock_settings: Settings) -> None:
        rewriter = AsyncQueryRewriter(mock_settings)
        assert rewriter._caller is not None

    @pytest.mark.anyio
    async def test_rewrite_returns_rewritten_query(self, mock_settings: Settings) -> None:
        rewriter = AsyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute_async") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            result = await rewriter.rewrite("vượt đèn đỏ bị phạt bao nhiêu?", ())

            assert isinstance(result, RewrittenQuery)
            assert result.original == "vượt đèn đỏ bị phạt bao nhiêu?"
            assert "đèn tín hiệu giao thông" in result.rewritten

    @pytest.mark.anyio
    async def test_rewrite_with_history(self, mock_settings: Settings) -> None:
        rewriter = AsyncQueryRewriter(mock_settings)
        with patch.object(rewriter._caller, "execute_async") as mock_execute:
            mock_execute.return_value = '{"query_vi": "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu?"}'

            history = (("Ô tô vượt đèn đỏ phạt bao nhiêu?", "Phạt 18-20 triệu đồng."),)
            result = await rewriter.rewrite("còn xe máy thì sao?", history)

            assert isinstance(result, RewrittenQuery)
            assert result.original == "còn xe máy thì sao?"


class TestRewrittenQuery:
    def test_dataclass_is_frozen(self) -> None:
        result = RewrittenQuery(original="test", rewritten="test rewritten")
        with pytest.raises(AttributeError):
            result.original = "changed"

    def test_dataclass_has_slots(self) -> None:
        result = RewrittenQuery(original="test", rewritten="test rewritten")
        assert hasattr(result, "__slots__")
