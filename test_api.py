"""Run a minimal OpenAI-compatible API connectivity check."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.settings import Settings


def main() -> None:
    """Send one bounded request without logging credentials or response content."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    body = json.dumps(
        {
            "model": settings.llm_model,
            "messages": [{"role": "user", "content": "Say hello in Vietnamese"}],
            "max_tokens": 50,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = Request(  # noqa: S310 - Settings validates the configured provider URL.
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=settings.llm_timeout_seconds) as response:  # noqa: S310
        logger.info("LLM connectivity check completed with HTTP {}", response.status)


if __name__ == "__main__":
    main()
