"""External knowledge search for CRAG."""

from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from loguru import logger

from legal_rag.corrective_rag.generate.config import RAGConfig


class WebSearchProvider:
    """Search external web pages for corrective knowledge."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the web search provider."""
        self.config = config
        self.client = httpx.Client(
            timeout=self.config.web_search_timeout_seconds,
            follow_redirects=True,
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
            response = self.client.get(self.config.web_search_url, params={"q": query})
            response.raise_for_status()
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
            if not _is_allowed_domain(url, self.config.web_search_allowed_domains):
                continue
            text = self._fetch_page_text(url) or result.get("snippet", "") or result.get("title", "")
            if not text:
                continue
            documents.append(
                {
                    "source_name": result.get("title", url),
                    "url": url,
                    "text": text[:5000],
                }
            )
        logger.info(f"Web search returned {len(documents)} external documents.")
        return documents

    def close(self) -> None:
        """Close HTTP resources."""
        self.client.close()

    def _fetch_page_text(self, url: str) -> str:
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.debug(f"Failed to fetch web result {url}: {error}")
            return ""

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return response.text[:5000]

        parser = _VisibleTextParser()
        parser.feed(response.text)
        return parser.text()


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
    """Return whether a URL passes the optional domain allowlist."""
    if not allowed_domains:
        return True
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)
