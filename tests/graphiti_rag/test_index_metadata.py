"""Tests for persisted Graphiti embedding compatibility metadata."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag.index_metadata import (
    GRAPHITI_EMBEDDING_PREPROCESSING_VERSION,
    GraphitiIndexCompatibilityError,
    ensure_graphiti_index_metadata,
    validate_graphiti_index_metadata,
)


def _profile(*, revision: str = "abc123") -> EmbeddingProfile:
    """Build one deterministic embedding profile."""
    return EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision=revision,
        tokenizer_id="test/model",
        tokenizer_revision=revision,
        dimensions=2,
        max_input_tokens=512,
        query_instruction="Instruct: ",
    )


class _FakeDriver:
    """Persist one metadata record and expose whether Graphiti data exists."""

    def __init__(self, *, metadata: Mapping[str, Any] | None = None, has_group_data: bool = False) -> None:
        self.metadata = dict(metadata) if metadata is not None else None
        self.has_group_data = has_group_data
        self.calls: list[tuple[str, dict[str, object]]] = []

    def session(self) -> _FakeSession:
        """Open one fake transaction session."""
        return _FakeSession(self)

    async def execute_query(self, query: str, **parameters: object) -> tuple[list[dict[str, object]], None, None]:
        """Execute the metadata operation identified by its stable query marker."""
        self.calls.append((query, parameters))
        if "index_metadata:read" in query:
            return ([dict(self.metadata)] if self.metadata is not None else []), None, None
        if "index_metadata:create_constraint" in query:
            return [], None, None
        msg = "unexpected query"
        raise AssertionError(msg)


class _FakeResult:
    """Expose the two result operations used by the transaction callback."""

    def __init__(self, record: dict[str, object] | None = None) -> None:
        self.record = record

    async def single(self, *, strict: bool) -> dict[str, object] | None:
        """Return one record."""
        assert strict is True
        return self.record

    async def consume(self) -> None:
        """Consume a write result."""


class _FakeTransaction:
    """Apply marker operations to the fake driver's transaction-local state."""

    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    async def run(self, query: str, **parameters: object) -> _FakeResult:
        """Execute one transaction query by its stable marker."""
        self.driver.calls.append((query, parameters))
        if "index_metadata:create" in query:
            created = self.driver.metadata is None
            if created:
                self.driver.metadata = {
                    name: value
                    for name, value in parameters.items()
                    if name not in {"group_id", "initialization_nonce"}
                }
                self.driver.metadata["initialization_nonce"] = parameters["initialization_nonce"]
            return _FakeResult({"created": created, **self.driver.metadata})
        if "index_metadata:legacy_group_data" in query:
            return _FakeResult({"has_legacy_data": self.driver.has_group_data})
        if "index_metadata:remove_initialization_nonce" in query:
            assert self.driver.metadata is not None
            self.driver.metadata.pop("initialization_nonce", None)
            return _FakeResult()
        msg = "unexpected transaction query"
        raise AssertionError(msg)


class _FakeSession:
    """Rollback fake metadata when the transaction callback raises."""

    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    async def __aenter__(self) -> _FakeSession:
        """Enter the session context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the session context."""

    async def execute_write(self, callback: object, *args: object) -> object:
        """Run a callback and restore state when it fails."""
        previous = dict(self.driver.metadata) if self.driver.metadata is not None else None
        try:
            return await callback(_FakeTransaction(self.driver), *args)
        except BaseException:
            self.driver.metadata = previous
            raise


def _expected_metadata(*, revision: str = "abc123") -> dict[str, object]:
    """Return the exact persisted fields for the test profile."""
    return {
        "metadata_schema_version": "1.0.0",
        "embedding_preprocessing_version": GRAPHITI_EMBEDDING_PREPROCESSING_VERSION,
        "embedding_profile_id": "test-profile",
        "embedding_model_id": "test/model",
        "embedding_model_revision": revision,
        "embedding_dimensions": 2,
    }


def test_ensure_creates_metadata_before_indexing_an_empty_group() -> None:
    """A new group records the exact embedding contract before any Graphiti writes."""
    driver = _FakeDriver()

    metadata = asyncio.run(ensure_graphiti_index_metadata(driver, "group-1", _profile()))

    assert driver.metadata == _expected_metadata()
    assert metadata.embedding_model_revision == "abc123"
    create_query, parameters = driver.calls[1]
    assert "group-1" not in create_query
    assert parameters["group_id"] == "group-1"
    assert {name: parameters[name] for name in _expected_metadata()} == _expected_metadata()
    assert isinstance(parameters["initialization_nonce"], str)


def test_ensure_accepts_matching_existing_metadata_without_writing() -> None:
    """Incremental indexing reuses a group only when its embedding identity matches."""
    driver = _FakeDriver(metadata=_expected_metadata(), has_group_data=True)

    asyncio.run(ensure_graphiti_index_metadata(driver, "group-1", _profile()))

    assert len(driver.calls) == 2
    assert "create_constraint" in driver.calls[0][0]
    assert "index_metadata:create" in driver.calls[1][0]


def test_ensure_rejects_a_legacy_group_without_metadata() -> None:
    """Old vectors cannot be silently treated as compatible after preprocessing changed."""
    driver = _FakeDriver(has_group_data=True)

    with pytest.raises(GraphitiIndexCompatibilityError, match="contains data without embedding metadata"):
        asyncio.run(ensure_graphiti_index_metadata(driver, "group-1", _profile()))

    assert driver.metadata is None


@pytest.mark.parametrize("field", list(_expected_metadata()))
def test_ensure_rejects_changed_or_missing_embedding_identity(field: str) -> None:
    """Every persisted contract field is required before a group can be reused."""
    persisted = _expected_metadata()
    persisted.pop(field)
    driver = _FakeDriver(metadata=persisted, has_group_data=True)

    with pytest.raises(GraphitiIndexCompatibilityError, match=field):
        asyncio.run(ensure_graphiti_index_metadata(driver, "group-1", _profile()))


def test_validate_requires_metadata_before_querying() -> None:
    """Retrieval fails clearly when a group predates the embedding contract."""
    driver = _FakeDriver(has_group_data=True)

    with pytest.raises(GraphitiIndexCompatibilityError, match="has no embedding metadata"):
        asyncio.run(validate_graphiti_index_metadata(driver, "group-1", _profile()))


def test_validate_accepts_matching_metadata() -> None:
    """A current group is safe to query without any metadata mutation."""
    driver = _FakeDriver(metadata=_expected_metadata(), has_group_data=True)

    metadata = asyncio.run(validate_graphiti_index_metadata(driver, "group-1", _profile()))

    assert metadata.embedding_dimensions == 2
    assert len(driver.calls) == 1


def test_metadata_rejects_blank_group_ids_before_querying() -> None:
    """A blank group cannot share one global compatibility record."""
    driver = _FakeDriver()

    with pytest.raises(ValueError, match="group_id must not be blank"):
        asyncio.run(ensure_graphiti_index_metadata(driver, " ", _profile()))

    assert driver.calls == []
