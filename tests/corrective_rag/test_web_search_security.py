"""Tests for CRAG web-search SSRF protections."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import httpx
import pytest

from legal_rag.corrective_rag.generate.web_search import (
    UnsafeWebSearchUrlError,
    WebSearchProvider,
    _is_allowed_domain,
    _validate_fetch_url,
)


def test_web_search_enabled_requires_domain_allowlist() -> None:
    """Enabled production web search must not treat an empty allowlist as allow-all."""
    with pytest.raises(ValueError, match="CRAG_WEB_SEARCH_ALLOWED_DOMAINS"):
        WebSearchProvider(_config(allowed_domains=()))


def test_domain_allowlist_does_not_allow_everything_when_empty() -> None:
    """An empty domain allowlist never authorizes a result URL."""
    assert not _is_allowed_domain("https://example.test/page", ())
    assert _is_allowed_domain("https://sub.example.test/page", ("example.test",))
    assert not _is_allowed_domain("https://badexample.test/page", ("example.test",))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://10.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/admin",
    ],
)
def test_validate_fetch_url_rejects_unsafe_schemes_and_addresses(url: str) -> None:
    """Only HTTP(S) public addresses may be fetched."""
    with pytest.raises(UnsafeWebSearchUrlError):
        _validate_fetch_url(url, allowed_domains=("example.test",), enforce_allowlist=False)


def test_validate_fetch_url_rejects_hosts_resolving_to_private_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS names resolving to private addresses are blocked."""
    monkeypatch.setattr(socket, "getaddrinfo", _resolver({"example.test": "192.168.1.10"}))

    with pytest.raises(UnsafeWebSearchUrlError, match="non-public address"):
        _validate_fetch_url("https://example.test/page", allowed_domains=("example.test",), enforce_allowlist=True)


def test_search_skips_result_redirecting_outside_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result whose redirect target leaves the allowlist is not returned."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _resolver({"duckduckgo.test": "93.184.216.34", "allowed.test": "93.184.216.35"}),
    )
    provider = WebSearchProvider(_config(allowed_domains=("allowed.test",)))
    provider.client = _FakeHttpClient(
        [
            _FakeResponse(200, text='<a class="result__a" href="https://allowed.test/page">Allowed</a>'),
            _FakeResponse(302, headers={"location": "https://not-allowed.test/private"}),
        ]
    )

    documents = provider.search("query")

    assert documents == []
    assert [str(request.url) for request in provider.client.requests] == [
        "https://duckduckgo.test/html/?q=query",
        "https://allowed.test/page",
    ]


def test_search_skips_result_redirecting_to_private_final_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """A final redirect URL is checked for public address ranges before fetching."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _resolver(
            {
                "duckduckgo.test": "93.184.216.34",
                "allowed.test": "93.184.216.35",
                "sub.allowed.test": "169.254.169.254",
            }
        ),
    )
    provider = WebSearchProvider(_config(allowed_domains=("allowed.test",)))
    provider.client = _FakeHttpClient(
        [
            _FakeResponse(200, text='<a class="result__a" href="https://allowed.test/page">Allowed</a>'),
            _FakeResponse(302, headers={"location": "https://sub.allowed.test/private"}),
        ]
    )

    documents = provider.search("query")

    assert documents == []
    assert [str(request.url) for request in provider.client.requests] == [
        "https://duckduckgo.test/html/?q=query",
        "https://allowed.test/page",
    ]


def _config(*, allowed_domains: tuple[str, ...]) -> SimpleNamespace:
    """Return config fields used by the web-search provider."""
    return SimpleNamespace(
        web_search_enabled=True,
        web_search_url="https://duckduckgo.test/html/",
        web_search_max_results=3,
        web_search_timeout_seconds=10.0,
        web_search_allowed_domains=allowed_domains,
    )


def _resolver(mapping: dict[str, str]):
    """Return a fake getaddrinfo resolver from host to one IPv4 address."""

    def fake_getaddrinfo(
        host: str,
        port: int | None,
        **kwargs: object,
    ) -> list[tuple[object, object, int, str, tuple[str, int]]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (mapping[host], port or 443))]

    return fake_getaddrinfo


class _FakeResponse:
    """Configured fake HTTP response."""

    def __init__(self, status_code: int, *, text: str = "", headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class _FakeHttpClient:
    """HTTP client fake returning configured responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def get(self, url: str, *, params: dict[str, str] | None = None) -> httpx.Response:
        """Return the next configured response."""
        self.requests.append(httpx.Request("GET", url, params=params))
        response = self.responses.pop(0)
        return httpx.Response(
            response.status_code,
            text=response.text,
            headers=response.headers,
            request=self.requests[-1],
        )

    def close(self) -> None:
        """Close fake resources."""
