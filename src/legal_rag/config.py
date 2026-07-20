"""Environment-driven configuration for the legal-RAG pipeline.

No secrets or environment-specific values are hardcoded; everything is read
from environment variables with safe defaults suitable for a local Neo4j.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    anthropic_api_key: str | None
    llm_model: str
    small_llm_model: str
    embedder: str  # 'openai' | 'bge'
    corpus_dir: Path
    corpus_zip: Path
    penalty_config: Path
    groundtruth_config: Path
    ragas_llm_provider: str | None
    ragas_llm_model: str | None
    ragas_embedding_provider: str | None
    ragas_embedding_model: str | None
    ragas_candidates_output: Path
    group_id: str

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables."""
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_model=os.getenv("GRAPHITI_LLM_MODEL", "claude-sonnet-4-5"),
            small_llm_model=os.getenv("GRAPHITI_SMALL_LLM_MODEL", "claude-haiku-4-5-latest"),
            embedder=os.getenv("GRAPHITI_EMBEDDER", "openai"),
            corpus_dir=Path(os.getenv("CORPUS_DIR", str(_REPO_ROOT / "data" / "raw"))),
            corpus_zip=Path(
                os.getenv(
                    "CORPUS_ZIP",
                    str(_REPO_ROOT / "data" / "raw" / "road_traffic_legal_documents.zip"),
                )
            ),
            penalty_config=Path(
                os.getenv("PENALTY_CONFIG", str(_REPO_ROOT / "data" / "config" / "penalty_corpus.json"))
            ),
            groundtruth_config=Path(
                os.getenv(
                    "GROUNDTRUTH_CONFIG",
                    str(_REPO_ROOT / "data" / "config" / "groundtruth_corpus_v1.json"),
                )
            ),
            ragas_llm_provider=os.getenv("RAGAS_LLM_PROVIDER"),
            ragas_llm_model=os.getenv("RAGAS_LLM_MODEL"),
            ragas_embedding_provider=os.getenv("RAGAS_EMBEDDING_PROVIDER"),
            ragas_embedding_model=os.getenv("RAGAS_EMBEDDING_MODEL"),
            ragas_candidates_output=Path(
                os.getenv(
                    "RAGAS_CANDIDATES_OUTPUT",
                    str(_REPO_ROOT / "data" / "evaluation" / "candidates.jsonl"),
                )
            ),
            group_id=os.getenv("GRAPHITI_GROUP_ID", "traffic_penalty_vn"),
        )
