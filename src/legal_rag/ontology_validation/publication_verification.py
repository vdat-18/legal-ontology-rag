"""Verify deployed ontology representations and merged W3ID redirects."""

from __future__ import annotations

import hashlib
import http.client
from contextlib import closing
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from legal_rag.ontology_validation.publication import landing_page_content, publication_endpoints
from legal_rag.ontology_validation.publication_config import NamespacePublicationSettings, PublicationResource
from legal_rag.ontology_validation.publication_verification_models import (
    EndpointVerificationResult,
    PublicationVerificationReport,
)

VerificationMode = Literal["deployment", "w3id"]


def verify_namespace_publication(
    settings: NamespacePublicationSettings,
    mode: VerificationMode,
    checked_at: datetime,
) -> PublicationVerificationReport:
    """Verify all 28 live HTML and Turtle requests with bounded sequential I/O."""
    results = tuple(
        _verify_representation(settings, resource, path, endpoint_kind, representation, mode)
        for resource, path, endpoint_kind in publication_endpoints(settings)
        for representation in ("html", "turtle")
    )
    overall_status = "passed" if all(not result.issues for result in results) else "failed"
    return PublicationVerificationReport(checked_at, mode, overall_status, results)


def _verify_representation(
    settings: NamespacePublicationSettings,
    resource: PublicationResource,
    path: str,
    endpoint_kind: str,
    representation: Literal["html", "turtle"],
    mode: VerificationMode,
) -> EndpointVerificationResult:
    """Verify one direct deployment response or one W3ID redirect response."""
    accept = "text/html" if representation == "html" else "text/turtle"
    expected_target = (
        f"{settings.publication_base_url}{path}/{'index.html' if representation == 'html' else 'ontology.ttl'}"
    )
    requested_url = expected_target if mode == "deployment" else f"{settings.namespace_root}{path}"
    status, media_type, location, content = _request(settings, requested_url, accept)
    issues = []
    if mode == "deployment":
        expected_content = (
            landing_page_content(settings, resource, path, endpoint_kind).encode("utf-8")
            if representation == "html"
            else resource.source_path.read_bytes()
        )
        expected_media_type = "text/html" if representation == "html" else "text/turtle"
        if status != 200:
            issues.append(f"unexpected-status:{status}:expected-200")
        if media_type != expected_media_type:
            issues.append(f"unexpected-media-type:{media_type}:expected-{expected_media_type}")
        if content != expected_content:
            issues.append("deployed-content-mismatch")
    else:
        if status != settings.redirect_status:
            issues.append(f"unexpected-status:{status}:expected-{settings.redirect_status}")
        if location != expected_target:
            issues.append(f"unexpected-location:{location}:expected-{expected_target}")
    return EndpointVerificationResult(
        resource_id=resource.resource_id,
        endpoint_kind=endpoint_kind,
        representation=representation,
        requested_url=requested_url,
        actual_status=status,
        actual_media_type=media_type,
        actual_location=location,
        content_sha256=hashlib.sha256(content).hexdigest(),
        issues=tuple(issues),
    )


def _request(
    settings: NamespacePublicationSettings,
    url: str,
    accept: str,
) -> tuple[int, str | None, str | None, bytes]:
    """Perform one bounded HTTPS request without following redirects."""
    parsed = urlsplit(url)
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=settings.timeout_seconds)
    with closing(connection):
        connection.request("GET", target, headers={"Accept": accept, "User-Agent": settings.user_agent})
        response = connection.getresponse()
        content = response.read(settings.max_response_bytes + 1)
    if len(content) > settings.max_response_bytes:
        msg = f"Response exceeded {settings.max_response_bytes} bytes: {url}"
        raise ValueError(msg)
    media_type_header = response.getheader("Content-Type")
    media_type = media_type_header.split(";", maxsplit=1)[0].strip().lower() if media_type_header else None
    return response.status, media_type, response.getheader("Location"), content
