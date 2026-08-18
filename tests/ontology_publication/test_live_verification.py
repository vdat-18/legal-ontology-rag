"""Tests for bounded deployment and W3ID live verification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar
from urllib.parse import urlsplit

from legal_rag.ontology_validation import publication_verification
from legal_rag.ontology_validation.publication import landing_page_content, publication_endpoints
from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings
from legal_rag.ontology_validation.publication_verification import verify_namespace_publication

CHECKED_AT = datetime(2026, 8, 18, 16, 0, tzinfo=UTC)


class _Response:
    """Minimal bounded HTTP response used by the fake connection."""

    def __init__(self, status: int, headers: dict[str, str], content: bytes = b"") -> None:
        self.status = status
        self._headers = headers
        self._content = content

    def read(self, amount: int) -> bytes:
        """Return at most the requested response bytes."""
        return self._content[:amount]

    def getheader(self, name: str) -> str | None:
        """Return one case-sensitive fake header."""
        return self._headers.get(name)


class _Connection:
    """Resolve requests from an exact precomputed response table."""

    responses: ClassVar[dict[tuple[str, str, str], _Response]] = {}
    timeouts: ClassVar[list[float]] = []

    def __init__(self, host: str, _port: int | None, timeout: float) -> None:
        self._host = host
        self._request: tuple[str, str] | None = None
        type(self).timeouts.append(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def request(self, _method: str, target: str, headers: dict[str, str]) -> None:
        """Capture the target and requested representation."""
        self._request = (target, headers["Accept"])

    def getresponse(self) -> _Response:
        """Return the exact configured response."""
        target, accept = self._request or ("", "")
        return type(self).responses[(self._host, target, accept)]

    def close(self) -> None:
        """Match the managed-resource boundary of HTTPSConnection."""


def test_deployment_and_w3id_verification_cover_every_representation(
    monkeypatch,
    publication_settings: NamespacePublicationSettings,
) -> None:
    """All 28 HTML/Turtle requests validate content, media types, redirects, and timeouts."""
    _Connection.responses = _responses(publication_settings)
    _Connection.timeouts = []
    monkeypatch.setattr(publication_verification.http.client, "HTTPSConnection", _Connection)

    deployment = verify_namespace_publication(publication_settings, "deployment", CHECKED_AT)
    w3id = verify_namespace_publication(publication_settings, "w3id", CHECKED_AT)

    assert deployment.overall_status == "passed"
    assert w3id.overall_status == "passed"
    assert len(deployment.results) == 28
    assert len(w3id.results) == 28
    assert all(timeout == 15 for timeout in _Connection.timeouts)


def test_live_mismatch_produces_a_failed_report(
    monkeypatch,
    publication_settings: NamespacePublicationSettings,
) -> None:
    """A deployed representation with changed bytes fails without a semantic heuristic."""
    responses = _responses(publication_settings)
    first_key = next(key for key in responses if key[0] == "vdat-18.github.io")
    response = responses[first_key]
    responses[first_key] = _Response(response.status, response._headers, b"changed")
    _Connection.responses = responses
    _Connection.timeouts = []
    monkeypatch.setattr(publication_verification.http.client, "HTTPSConnection", _Connection)

    report = verify_namespace_publication(publication_settings, "deployment", CHECKED_AT)

    assert report.overall_status == "failed"
    assert sum(bool(result.issues) for result in report.results) == 1
    assert "deployed-content-mismatch" in next(result.issues for result in report.results if result.issues)


def _responses(settings: NamespacePublicationSettings) -> dict[tuple[str, str, str], _Response]:
    """Build exact deployment and redirect responses from the production contract."""
    responses = {}
    publication_prefix = urlsplit(settings.publication_base_url).path.rstrip("/")
    namespace_prefix = urlsplit(settings.namespace_root).path.rstrip("/")
    for resource, path, endpoint_kind in publication_endpoints(settings):
        for representation, accept in (("html", "text/html"), ("turtle", "text/turtle")):
            filename = "index.html" if representation == "html" else "ontology.ttl"
            target = f"{publication_prefix}/{path}/{filename}"
            content = (
                landing_page_content(settings, resource, path, endpoint_kind).encode("utf-8")
                if representation == "html"
                else resource.source_path.read_bytes()
            )
            media_type = "text/html; charset=utf-8" if representation == "html" else "text/turtle"
            responses[("vdat-18.github.io", target, accept)] = _Response(200, {"Content-Type": media_type}, content)
            w3id_target = f"{namespace_prefix}/{path}"
            location = f"{settings.publication_base_url}{path}/{filename}"
            responses[("w3id.org", w3id_target, accept)] = _Response(303, {"Location": location})
    return responses
