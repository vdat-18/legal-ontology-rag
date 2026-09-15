"""Tests for generic Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from legal_rag.retrieval.rrf import reciprocal_rank_fusion


def test_fuses_overlapping_rankings():
    """An item present in both legs receives both reciprocal-rank contributions."""
    results = reciprocal_rank_fusion(
        (("shared", "semantic"), ("shared", "keyword")),
        key=lambda item: item,
        rrf_k=60,
    )

    assert [result.item for result in results] == ["shared", "semantic", "keyword"]
    assert [result.score for result in results] == pytest.approx([2 / 61, 1 / 62, 1 / 62])


def test_keeps_item_present_in_only_one_leg():
    """Items do not need to occur in multiple rankings to be returned."""
    results = reciprocal_rank_fusion((("only",), ()), key=lambda item: item, rrf_k=10)

    assert len(results) == 1
    assert results[0].item == "only"
    assert results[0].score == pytest.approx(1 / 11)


def test_duplicate_within_leg_contributes_once_and_does_not_consume_rank():
    """Per-leg duplicates are removed before reciprocal ranks are assigned."""
    results = reciprocal_rank_fusion(
        (("first", "first", "second"),),
        key=lambda item: item,
        rrf_k=10,
    )

    assert [result.item for result in results] == ["first", "second"]
    assert [result.score for result in results] == pytest.approx([1 / 11, 1 / 12])


def test_empty_rankings_return_empty_results():
    """No rankings or only empty rankings produce no fused results."""
    assert reciprocal_rank_fusion((), key=lambda item: item, rrf_k=60) == []
    assert reciprocal_rank_fusion(((), ()), key=lambda item: item, rrf_k=60) == []


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_rejects_invalid_rrf_k(rrf_k):
    """The smoothing constant must be positive."""
    with pytest.raises(ValueError, match="rrf_k must be at least 1"):
        reciprocal_rank_fusion((("item",),), key=lambda item: item, rrf_k=rrf_k)


def test_ties_use_first_seen_order_and_keep_first_representative():
    """Equal scores are stable across legs and retain the first matching item."""
    first = {"id": "shared", "source": "semantic"}
    duplicate = {"id": "shared", "source": "keyword"}
    results = reciprocal_rank_fusion(
        ((first, {"id": "semantic", "source": "semantic"}), (duplicate, {"id": "keyword", "source": "keyword"})),
        key=lambda item: item["id"],
        rrf_k=60,
    )

    assert [result.item["id"] for result in results] == ["shared", "semantic", "keyword"]
    assert results[0].item is first
    assert [result.first_seen_order for result in results] == [0, 1, 2]
