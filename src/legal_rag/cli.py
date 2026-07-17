"""Command-line entry points for the legal-RAG pipeline.

legal-rag ingest [--reset]
legal-rag query "vượt đèn đỏ xe máy" 2025-02-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv
from loguru import logger

from legal_rag.config import Settings
from legal_rag.graph.ingest import ingest
from legal_rag.retrieval.temporal import query_at


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Run corpus ingestion into the graph."""
    asyncio.run(ingest(Settings.from_env(), reset=args.reset))


def _cmd_query(args: argparse.Namespace) -> None:
    """Run a point-in-time query and print valid facts."""
    moment = datetime.fromisoformat(args.date).replace(tzinfo=UTC)
    facts = asyncio.run(query_at(Settings.from_env(), args.query, moment))
    logger.info("'{}' tại {}: {} fact còn hiệu lực", args.query, moment.date(), len(facts))
    for f in facts:
        start = f.valid_from.date() if f.valid_from else "—"
        end = f.valid_to.date() if f.valid_to else "nay"
        sys.stdout.write(f"• {f.fact}  [{start} → {end}]\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="legal-rag", description="Temporal legal-RAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="ingest the penalty corpus into Graphiti")
    ingest_p.add_argument("--reset", action="store_true", help="clear the graph first")
    ingest_p.set_defaults(func=_cmd_ingest)

    query_p = sub.add_parser("query", help="point-in-time penalty query")
    query_p.add_argument("query", help="câu hỏi, ví dụ 'vượt đèn đỏ xe máy'")
    query_p.add_argument("date", help="thời điểm tham chiếu ISO, ví dụ 2025-02-01")
    query_p.set_defaults(func=_cmd_query)
    return parser


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (RuntimeError, ValueError) as exc:
        logger.error("Lỗi: {}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
