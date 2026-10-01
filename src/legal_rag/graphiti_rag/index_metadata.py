"""Persist and validate the embedding contract for one Graphiti group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from uuid import uuid4

from neo4j import AsyncDriver, AsyncManagedTransaction

from legal_rag.embedding.profiles import EmbeddingProfile

GRAPHITI_EMBEDDING_PREPROCESSING_VERSION = "1.0.0"
_METADATA_SCHEMA_VERSION = "1.0.0"

_READ_METADATA_QUERY = """
// legal_rag_graphiti_index_metadata:read
MATCH (metadata:LegalRagGraphitiIndexMetadata {group_id: $group_id})
RETURN metadata.metadata_schema_version AS metadata_schema_version,
       metadata.embedding_preprocessing_version AS embedding_preprocessing_version,
       metadata.embedding_profile_id AS embedding_profile_id,
       metadata.embedding_model_id AS embedding_model_id,
       metadata.embedding_model_revision AS embedding_model_revision,
       metadata.embedding_dimensions AS embedding_dimensions
"""

_CREATE_CONSTRAINT_QUERY = """
// legal_rag_graphiti_index_metadata:create_constraint
CREATE CONSTRAINT legal_rag_graphiti_index_metadata_group IF NOT EXISTS
FOR (metadata:LegalRagGraphitiIndexMetadata)
REQUIRE metadata.group_id IS UNIQUE
"""

_CREATE_METADATA_QUERY = """
// legal_rag_graphiti_index_metadata:create
MERGE (metadata:LegalRagGraphitiIndexMetadata {group_id: $group_id})
ON CREATE SET metadata.metadata_schema_version = $metadata_schema_version,
              metadata.embedding_preprocessing_version = $embedding_preprocessing_version,
              metadata.embedding_profile_id = $embedding_profile_id,
              metadata.embedding_model_id = $embedding_model_id,
              metadata.embedding_model_revision = $embedding_model_revision,
              metadata.embedding_dimensions = $embedding_dimensions,
              metadata.initialization_nonce = $initialization_nonce
RETURN metadata.initialization_nonce = $initialization_nonce AS created,
       metadata.metadata_schema_version AS metadata_schema_version,
       metadata.embedding_preprocessing_version AS embedding_preprocessing_version,
       metadata.embedding_profile_id AS embedding_profile_id,
       metadata.embedding_model_id AS embedding_model_id,
       metadata.embedding_model_revision AS embedding_model_revision,
       metadata.embedding_dimensions AS embedding_dimensions
"""

_LEGACY_GROUP_DATA_QUERY = """
// legal_rag_graphiti_index_metadata:legacy_group_data
RETURN EXISTS {
           MATCH (:Episodic {group_id: $group_id})
       } OR EXISTS {
           MATCH (:Entity {group_id: $group_id})
       } OR EXISTS {
           MATCH (:Community {group_id: $group_id})
       } OR EXISTS {
           MATCH ()-[relation:RELATES_TO {group_id: $group_id}]-()
       } OR EXISTS {
           MATCH ()-[mention:MENTIONS {group_id: $group_id}]-()
       } AS has_legacy_data
"""

_REMOVE_INITIALIZATION_NONCE_QUERY = """
// legal_rag_graphiti_index_metadata:remove_initialization_nonce
MATCH (metadata:LegalRagGraphitiIndexMetadata {group_id: $group_id})
WHERE metadata.initialization_nonce = $initialization_nonce
REMOVE metadata.initialization_nonce
"""


class _QueryExecutor(Protocol):
    """Neo4j or Graphiti driver boundary used by metadata checks."""

    async def execute_query(self, query: str, **parameters: object) -> tuple[Sequence[Mapping[str, Any]], Any, Any]:
        """Execute one parameterized Cypher query."""
        ...


class GraphitiIndexCompatibilityError(ValueError):
    """Raised when a Graphiti group was built with another embedding contract."""


@dataclass(frozen=True, slots=True)
class GraphitiIndexMetadata:
    """Embedding identity required by all vectors persisted in one Graphiti group."""

    metadata_schema_version: str
    embedding_preprocessing_version: str
    embedding_profile_id: str
    embedding_model_id: str
    embedding_model_revision: str
    embedding_dimensions: int

    @classmethod
    def from_profile(cls, profile: EmbeddingProfile) -> GraphitiIndexMetadata:
        """Build the expected persisted identity for an embedding profile."""
        return cls(
            metadata_schema_version=_METADATA_SCHEMA_VERSION,
            embedding_preprocessing_version=GRAPHITI_EMBEDDING_PREPROCESSING_VERSION,
            embedding_profile_id=profile.profile_id,
            embedding_model_id=profile.model_id,
            embedding_model_revision=profile.model_revision,
            embedding_dimensions=profile.dimensions,
        )


async def ensure_graphiti_index_metadata(
    driver: AsyncDriver,
    group_id: str,
    profile: EmbeddingProfile,
) -> GraphitiIndexMetadata:
    """Create metadata for an empty group or reject an incompatible existing group.

    A group containing Graphiti data but no metadata predates this contract. Such a group may
    contain vectors produced from the Python representation of one-item text lists, so it must
    be rebuilt instead of being adopted silently.
    """
    _validate_group_id(group_id)
    expected = GraphitiIndexMetadata.from_profile(profile)
    await driver.execute_query(_CREATE_CONSTRAINT_QUERY)
    initialization_nonce = uuid4().hex
    async with driver.session() as session:
        return await session.execute_write(
            _ensure_metadata_transaction,
            group_id,
            expected,
            initialization_nonce,
        )


async def validate_graphiti_index_metadata(
    driver: _QueryExecutor,
    group_id: str,
    profile: EmbeddingProfile,
) -> GraphitiIndexMetadata:
    """Validate the persisted embedding identity before querying a Graphiti group."""
    _validate_group_id(group_id)
    expected = GraphitiIndexMetadata.from_profile(profile)
    persisted = await _read_metadata(driver, group_id)
    if persisted is None:
        msg = (
            f"Graphiti group {group_id!r} has no embedding metadata. "
            "Rebuild the group with the current indexer before querying it."
        )
        raise GraphitiIndexCompatibilityError(msg)
    _validate_metadata(group_id, persisted, expected)
    return expected


async def _read_metadata(driver: _QueryExecutor, group_id: str) -> Mapping[str, Any] | None:
    """Read the single metadata record for a group."""
    records, _, _ = await driver.execute_query(_READ_METADATA_QUERY, group_id=group_id)
    if not records:
        return None
    if len(records) != 1:
        msg = f"Graphiti group {group_id!r} has duplicate embedding metadata records."
        raise GraphitiIndexCompatibilityError(msg)
    return records[0]


async def _ensure_metadata_transaction(
    transaction: AsyncManagedTransaction,
    group_id: str,
    expected: GraphitiIndexMetadata,
    initialization_nonce: str,
) -> GraphitiIndexMetadata:
    """Create or validate one marker atomically with legacy-data detection."""
    parameters = {
        "group_id": group_id,
        "initialization_nonce": initialization_nonce,
        **asdict(expected),
    }
    result = await transaction.run(_CREATE_METADATA_QUERY, **parameters)
    persisted = await result.single(strict=True)
    if persisted is None:
        msg = f"Graphiti group {group_id!r} did not produce embedding metadata."
        raise GraphitiIndexCompatibilityError(msg)

    if persisted["created"] is True:
        legacy_result = await transaction.run(_LEGACY_GROUP_DATA_QUERY, group_id=group_id)
        legacy_record = await legacy_result.single(strict=True)
        if legacy_record is None or legacy_record["has_legacy_data"] is True:
            msg = (
                f"Graphiti group {group_id!r} contains data without embedding metadata. "
                "Rebuild the group with the current indexer or use a new group_id."
            )
            raise GraphitiIndexCompatibilityError(msg)
        cleanup = await transaction.run(
            _REMOVE_INITIALIZATION_NONCE_QUERY,
            group_id=group_id,
            initialization_nonce=initialization_nonce,
        )
        await cleanup.consume()

    _validate_metadata(group_id, persisted, expected)
    return expected


def _validate_group_id(group_id: str) -> None:
    """Reject group ids that cannot identify a persisted metadata record."""
    if not group_id.strip():
        msg = "Graphiti group_id must not be blank."
        raise ValueError(msg)


def _validate_metadata(
    group_id: str,
    persisted: Mapping[str, Any],
    expected: GraphitiIndexMetadata,
) -> None:
    """Reject every persisted embedding identity mismatch."""
    expected_values = asdict(expected)
    mismatched_fields = [name for name, value in expected_values.items() if persisted.get(name) != value]
    if mismatched_fields:
        fields = ", ".join(mismatched_fields)
        msg = (
            f"Graphiti group {group_id!r} has incompatible embedding metadata ({fields}). "
            "Rebuild the group or use a new group_id."
        )
        raise GraphitiIndexCompatibilityError(msg)
