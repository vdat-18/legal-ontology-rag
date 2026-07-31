"""External knowledge search for CRAG."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from loguru import logger

from legal_rag.corrective_rag.generate.config import RAGConfig

_ALLOWED_SCHEMES = frozenset(("http", "https"))
_MAX_REDIRECTS = 5


class UnsafeWebSearchUrlError(ValueError):
    """Raised when a web-search URL violates SSRF protections."""


class WebSearchProvider:
    """Search external web pages for corrective knowledge."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the web search provider."""
        self.config = config
        if self.config.web_search_enabled and not self.config.web_search_allowed_domains:
            msg = "CRAG_WEB_SEARCH_ALLOWED_DOMAINS must not be empty when CRAG web search is enabled."
            raise ValueError(msg)
        self.client = httpx.Client(
            timeout=self.config.web_search_timeout_seconds,
            follow_redirects=False,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
                )
            },
        )

    def search(self, query: str) -> list[Mapping[str, str]]:
        """Search the web and return external text documents."""
        if not self.config.web_search_enabled:
            logger.info("CRAG web search is disabled.")
            return []

        try:
            response = self._get_checked_response(
                self.config.web_search_url,
                params={"q": query},
                allowed_domains=(),
                enforce_allowlist=False,
            )
            response.raise_for_status()
        except UnsafeWebSearchUrlError:
            raise
        except httpx.HTTPError as error:
            logger.warning(f"Web search request failed: {error}")
            return []

        parser = _DuckDuckGoResultsParser(self.config.web_search_url)
        parser.feed(response.text)
        documents: list[Mapping[str, str]] = []
        for result in parser.results:
            if len(documents) >= self.config.web_search_max_results:
                break
            url = _unwrap_duckduckgo_url(result["url"])
            if not _is_allowed_result_url(url, self.config.web_search_allowed_domains):
                continue
            try:
                page = self._fetch_page(url)
            except UnsafeWebSearchUrlError as error:
                logger.debug(f"Blocked unsafe web result {url}: {error}")
                continue
            text = page.text or result.get("snippet", "") or result.get("title", "")
            if not text:
                continue
            documents.append(
                {
                    "source_name": result.get("title", page.url),
                    "url": page.url,
                    "text": text[:5000],
                }
            )
        logger.info(f"Web search returned {len(documents)} external documents.")
        return documents

    def close(self) -> None:
        """Close HTTP resources."""
        self.client.close()

    def _fetch_page(self, url: str) -> _FetchedPage:
        try:
            response = self._get_checked_response(
                url,
                params=None,
                allowed_domains=self.config.web_search_allowed_domains,
                enforce_allowlist=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.debug(f"Failed to fetch web result {url}: {error}")
            return _FetchedPage(url=url, text="")

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return _FetchedPage(url=str(response.url), text=response.text[:5000])

        parser = _VisibleTextParser()
        parser.feed(response.text)
        return _FetchedPage(url=str(response.url), text=parser.text())

    def _get_checked_response(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None,
        allowed_domains: tuple[str, ...],
        enforce_allowlist: bool,
    ) -> httpx.Response:
        """GET a URL while checking each redirect hop and final URL."""
        current_url = url
        current_params = params
        for _ in range(_MAX_REDIRECTS + 1):
            _validate_fetch_url(current_url, allowed_domains=allowed_domains, enforce_allowlist=enforce_allowlist)
            response = self.client.get(current_url, params=current_params)
            _validate_fetch_url(
                str(response.url),
                allowed_domains=allowed_domains,
                enforce_allowlist=enforce_allowlist,
            )
            if not response.is_redirect:
                return response
            location = response.headers.get("location")
            if not location:
                return response
            current_url = urljoin(str(response.url), location)
            current_params = None
        msg = f"Web search redirect limit exceeded for {url}"
        raise UnsafeWebSearchUrlError(msg)


@dataclass(frozen=True, slots=True)
class _FetchedPage:
    """Fetched page URL and extracted text."""

    url: str
    text: str


class _DuckDuckGoResultsParser(HTMLParser):
    """Extract result links from DuckDuckGo HTML output."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.results: list[dict[str, str]] = []
        self._active_link: dict[str, str] | None = None
        self._active_snippet_index: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        classes = attributes.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._active_link = {"url": urljoin(self.base_url, attributes.get("href", "")), "title": ""}
        elif "result__snippet" in classes and self.results:
            self._active_snippet_index = len(self.results) - 1

    def handle_data(self, data: str) -> None:
        if self._active_link is not None:
            self._active_link["title"] += data
        elif self._active_snippet_index is not None:
            result = self.results[self._active_snippet_index]
            result["snippet"] = f"{result.get('snippet', '')} {data}".strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_link is not None:
            self._active_link["title"] = " ".join(self._active_link["title"].split())
            if self._active_link["url"]:
                self.results.append(self._active_link)
            self._active_link = None
        elif self._active_snippet_index is not None and tag in {"a", "div"}:
            self._active_snippet_index = None


class _VisibleTextParser(HTMLParser):
    """Extract visible-ish text from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        """Return extracted text."""
        return " ".join(self._parts)


def _unwrap_duckduckgo_url(url: str) -> str:
    """Extract the target URL from a DuckDuckGo redirect URL."""
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("uddg")
    if values:
        return unquote(values[0])
    return url


def _is_allowed_domain(url: str, allowed_domains: tuple[str, ...]) -> bool:
    """Return whether a URL passes the required domain allowlist."""
    if not allowed_domains:
        return False
    host = _normalized_host(url)
    if host is None:
        return False
    domains = tuple(_normalize_domain(domain) for domain in allowed_domains if domain.strip())
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _is_allowed_result_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
    """Return whether a search result URL is allowed to be fetched."""
    try:
        _validate_fetch_url(url, allowed_domains=allowed_domains, enforce_allowlist=True)
    except UnsafeWebSearchUrlError as error:
        logger.debug(f"Blocked unsafe web result {url}: {error}")
        return False
    return True


def _validate_fetch_url(url: str, *, allowed_domains: tuple[str, ...], enforce_allowlist: bool) -> None:
    """Reject URLs that fail SSRF protections."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        msg = f"Web search URL must use HTTP(S): {url}"
        raise UnsafeWebSearchUrlError(msg)
    host = _normalized_host(url)
    if host is None:
        msg = f"Web search URL must include a host: {url}"
        raise UnsafeWebSearchUrlError(msg)
    if enforce_allowlist and not _is_allowed_domain(url, allowed_domains):
        msg = f"Web search URL host is not allowlisted: {host}"
        raise UnsafeWebSearchUrlError(msg)
    _validate_public_host(host, _port(parsed))


def _validate_public_host(host: str, port: int | None) -> None:
    """Reject loopback, private, link-local, and otherwise non-public hosts."""
    for address in _host_addresses(host, port):
        if _is_blocked_address(address):
            msg = f"Web search URL resolves to a non-public address: {address}"
            raise UnsafeWebSearchUrlError(msg)


def _host_addresses(host: str, port: int | None) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    """Resolve a host into IP addresses for SSRF checks."""
    try:
        return (ipaddress.ip_address(host),)
    except ValueError:
        pass

    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        msg = f"Web search URL host could not be resolved: {host}"
        raise UnsafeWebSearchUrlError(msg) from error
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for item in resolved:
        sockaddr = item[4]
        addresses.append(ipaddress.ip_address(sockaddr[0]))
    return tuple(addresses)


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether an address is unsafe for server-side web search."""
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
        or address.is_reserved
        or address.is_multicast
    )


def _normalized_host(url: str) -> str | None:
    """Return a normalized hostname from one URL."""
    host = urlparse(url).hostname
    if host is None:
        return None
    return _normalize_domain(host)


def _normalize_domain(value: str) -> str:
    """Normalize a domain for allowlist comparison."""
    return value.strip().rstrip(".").lower().encode("idna").decode("ascii")


def _port(parsed_url) -> int | None:
    """Return a parsed URL port, surfacing invalid port values as unsafe URLs."""
    try:
        return parsed_url.port
    except ValueError as error:
        msg = f"Web search URL has an invalid port: {parsed_url.geturl()}"
        raise UnsafeWebSearchUrlError(msg) from error
