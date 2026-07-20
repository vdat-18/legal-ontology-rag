"""Domain-specific exceptions for the legal-RAG pipeline.

Messages live inside the exception classes (not at the raise site) so callers
raise them with structured arguments only.
"""

from __future__ import annotations


class MissingCredentialError(RuntimeError):
    """A required credential or configuration value is absent."""

    def __init__(self, name: str) -> None:
        """Record which credential/config key was missing."""
        super().__init__(f"{name} is not set")


class UnsupportedDocumentError(ValueError):
    """A source document type is not supported by the loader."""

    def __init__(self, filename: str) -> None:
        """Record the offending file name."""
        super().__init__(f"Unsupported document type: {filename}")


class ReferenceDateNotFoundError(ValueError):
    """No effective/reference date could be determined for a document."""

    def __init__(self, filename: str) -> None:
        """Record the document whose reference date could not be resolved."""
        super().__init__(f"Cannot determine a reference date for {filename}")
