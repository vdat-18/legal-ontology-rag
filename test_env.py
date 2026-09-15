"""Check whether the local LLM environment is configured without exposing credentials."""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from loguru import logger


def main() -> None:
    """Load the local dotenv file and report only configuration presence."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    logger.info("Dotenv file found: {}", bool(dotenv_path))
    logger.info("LLM API key configured: {}", bool(os.getenv("LLM_API_KEY")))
    logger.info("LLM base URL configured: {}", bool(os.getenv("LLM_BASE_URL")))
    logger.info("LLM model configured: {}", bool(os.getenv("LLM_MODEL")))


if __name__ == "__main__":
    main()
