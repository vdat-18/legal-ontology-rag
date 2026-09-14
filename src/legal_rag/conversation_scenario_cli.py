"""Command-line interface for multi-turn conversation scenario sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the conversation-scenario sampling command-line interface."""
    parser = argparse.ArgumentParser(
        description="Sample target-first conversations from reviewed single-turn scenarios."
    )
    parser.add_argument("scenarios_path", type=Path, help="Reviewed single-turn scenarios JSONL path")
    parser.add_argument("relations_path", type=Path, help="Evidence-relations JSONL path")
    parser.add_argument("conversations_output_path", type=Path, help="Destination conversation scenarios JSONL path")
    parser.add_argument(
        "--conversation-distribution",
        type=Path,
        required=True,
        help="Conversation-distribution config JSON path",
    )
    parser.add_argument(
        "--total-conversations",
        type=int,
        required=True,
        help="Target number of conversation scenarios",
    )
    return parser


def main() -> None:
    """Sample deterministic conversations around retrieval-required targets."""
    from legal_rag.benchmark_synthesis.conversation_scenario_sampling import (
        load_conversation_distribution,
        sample_conversation_scenarios,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_evidence_relations,
        read_scenarios,
        write_conversation_scenarios,
    )

    arguments = _parser().parse_args()
    scenarios = read_scenarios(arguments.scenarios_path)
    relations = read_evidence_relations(arguments.relations_path)
    distribution = load_conversation_distribution(arguments.conversation_distribution)
    conversations = sample_conversation_scenarios(
        scenarios,
        relations,
        distribution,
        total_conversations=arguments.total_conversations,
    )
    write_conversation_scenarios(arguments.conversations_output_path, conversations)
    logger.info(
        "Sampled {} multi-turn conversations from {} scenarios. Wrote {}",
        len(conversations),
        len(scenarios),
        arguments.conversations_output_path,
    )


if __name__ == "__main__":
    main()
