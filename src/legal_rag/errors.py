"""Domain-specific exceptions for the legal-RAG pipeline.

Messages live inside the exception classes (not at the raise site) so callers
raise them with structured arguments only.
"""

from __future__ import annotations

from pathlib import Path


class MissingCredentialError(RuntimeError):
    """A required credential or configuration value is absent."""

    def __init__(self, name: str) -> None:
        """Record which credential/config key was missing."""
        super().__init__(f"{name} is not set")


class ReferenceDateNotFoundError(ValueError):
    """No effective/reference date could be determined for a document."""

    def __init__(self, filename: str) -> None:
        """Record the document whose reference date could not be resolved."""
        super().__init__(f"Cannot determine a reference date for {filename}")


class DuplicateReferenceContextError(ValueError):
    """A source batch contains an ambiguous context identifier or text."""

    def __init__(self, first_context_id: str, second_context_id: str | None = None) -> None:
        """Record the context identifiers that cannot be mapped uniquely."""
        context_ids = first_context_id if second_context_id is None else f"{first_context_id}, {second_context_id}"
        super().__init__(f"Reference context must be unique: {context_ids}")


class InvalidSyntheticSampleError(ValueError):
    """A generated sample is missing a required field or supported shape."""

    def __init__(self, field_name: str) -> None:
        """Record the invalid field without exposing generated content."""
        super().__init__(f"Synthetic sample has an invalid {field_name}")


class UnknownReferenceContextError(ValueError):
    """A generated sample references content outside the source batch."""

    def __init__(self) -> None:
        """Report provenance loss without logging the source text."""
        super().__init__("Synthetic sample references an unknown source context")


class InvalidTestsetSizeError(ValueError):
    """The requested synthetic testset size is not positive."""

    def __init__(self) -> None:
        """Report the invalid size without embedding policy in the caller."""
        super().__init__("testset_size must be greater than zero")


class UnsupportedRagasProviderError(ValueError):
    """A configured Ragas provider is not supported by this integration."""

    def __init__(self, setting_name: str, provider: str, supported: tuple[str, ...]) -> None:
        """Record the unsupported provider and accepted values."""
        super().__init__(f"{setting_name}={provider} is not supported; expected one of: {', '.join(supported)}")


class InvalidRagasComponentError(ValueError):
    """The caller provided an incomplete Ragas component override."""

    def __init__(self) -> None:
        """Report that generation components must be injected together."""
        super().__init__("generator and query_synthesizer must be provided together")


class EmptyReferenceContextBatchError(ValueError):
    """No legal source contexts were provided for synthesis."""

    def __init__(self) -> None:
        """Report the empty source boundary before invoking Ragas."""
        super().__init__("At least one reference context is required")


class MissingCorpusArtifactError(FileNotFoundError):
    """A required corpus archive or selection file does not exist."""

    def __init__(self, path: Path) -> None:
        """Record the missing path supplied by the caller."""
        super().__init__(f"Corpus artifact does not exist: {path}")


class DuplicateSelectedDocumentError(ValueError):
    """A corpus selection contains the same document more than once."""

    def __init__(self, filename: str) -> None:
        """Record the duplicate filename."""
        super().__init__(f"Selected document is duplicated: {filename}")


class SelectedDocumentNotFoundError(ValueError):
    """A selected document is absent from the canonical archive."""

    def __init__(self, filename: str, archive_name: str) -> None:
        """Record the missing document and archive name."""
        super().__init__(f"Selected document {filename} is not present in {archive_name}")
