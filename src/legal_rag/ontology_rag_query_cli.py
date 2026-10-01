"""Command-line interface for querying the Ontology-enhanced RAG pipeline."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ontology RAG query CLI."""
    parser = argparse.ArgumentParser(description="Answer one question using the ontology-enhanced RAG pipeline.")
    parser.add_argument("query", type=str, help="Natural-language question in Vietnamese")
    parser.add_argument(
        "--top-k", type=_positive_int, default=None, help="Override the configured number of retrieved rules"
    )
    parser.add_argument(
        "--query-date", type=_iso_date, default=None, help="Query date in YYYY-MM-DD format (defaults to today)"
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        action="append",
        required=True,
        help="Ontology TTL path; repeat the option to provide multiple files",
    )
    return parser


def main() -> None:
    """Run the ontology RAG pipeline and log answer and penalty rules."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
    from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    settings = Settings.from_environment()
    milvus_settings = MilvusSettings.from_environment()
    naive_rag_settings = NaiveRagSettings.from_environment()
    ontology_paths = arguments.ontology

    pipeline = OntologyRAGPipeline(settings, milvus_settings, naive_rag_settings, ontology_paths)
    try:
        result = pipeline.run(arguments.query, query_date=arguments.query_date, top_k=arguments.top_k)

        logger.info("Answer: {}", result.answer)
        logger.info("Valid rules: {}", len(result.rules))
        logger.info(
            "Temporal mode: {} dates={} status={} candidates={} invalid={} indeterminate={}",
            getattr(result, "temporal_mode", None),
            [item.isoformat() for item in getattr(result, "query_dates", ())],
            getattr(result, "temporal_status", {}),
            getattr(result, "candidate_count", None),
            getattr(result, "invalid_rule_count", None),
            getattr(result, "indeterminate_rule_count", None),
        )

        for rule in result.rules:
            parts = []
            if rule.monetary_penalty_min is not None or rule.monetary_penalty_max is not None:
                parts.append(f"Phạt {rule.monetary_penalty_min}-{rule.monetary_penalty_max} {rule.currency or 'VND'}")
            if rule.point_value is not None:
                parts.append(f"trừ {rule.point_value} điểm")
            consequence = ", ".join(parts) if parts else "Không có thông tin phạt"

            logger.info("Rule: {} - {}", rule.conduct_label_vi, consequence)
            logger.info("  Source: {} ({})", rule.provision_path, rule.document_number)
            logger.info("  Evidence chunks: {}", list(getattr(rule, "source_chunk_ids", ())))
            logger.info(
                "  Applicability: subject={} roles={} vehicles={}",
                getattr(rule, "subject_kind", None),
                list(getattr(rule, "roles", ())),
                list(getattr(rule, "vehicle_categories", ())),
            )
            logger.info(
                "  Components: consequences={} conditions={} exceptions={}",
                len(getattr(rule, "consequences", ())),
                len(getattr(rule, "condition_expressions", ())),
                len(getattr(rule, "exceptions", ())),
            )

        for source in result.sources[:5]:
            logger.info("Source: {} ({}, score={:.4f})", source.chunk_id, source.source_name, source.score)
    finally:
        pipeline.close()


def _iso_date(value: str) -> date:
    """Parse one canonical ISO calendar date for argparse."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        msg = f"Invalid query date {value!r}; expected YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed.isoformat() != value:
        msg = f"Invalid query date {value!r}; expected YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_int(value: str) -> int:
    """Parse one positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        msg = f"Expected a positive integer, got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


if __name__ == "__main__":
    main()
