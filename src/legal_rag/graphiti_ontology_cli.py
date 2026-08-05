"""Command-line interface for exporting the traffic-penalty ontology as OWL."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology export command-line interface."""
    parser = argparse.ArgumentParser(description="Export the traffic-penalty ontology as OWL 2 Turtle.")
    parser.add_argument("output", type=Path, help="Path to write the .ttl file to")
    return parser


def main() -> None:
    """Render the ontology to OWL and write it to the requested path."""
    from legal_rag.graphiti_rag.owl import build_owl

    arguments = _parser().parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(build_owl(), encoding="utf-8")
    logger.info("Wrote ontology to {}", arguments.output)


if __name__ == "__main__":
    main()
