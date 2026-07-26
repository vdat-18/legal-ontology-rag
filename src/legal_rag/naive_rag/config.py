"""Naive RAG configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class EmbedderConfig:
    """BGE-M3 embedding model settings."""

    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 8
    device: str = "cpu"


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus vector store settings."""

    uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "./milvus_naive_rag.db"))
    collection_name: str = "naive_rag_chunks"
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128
    nprobe: int = 16
    insert_batch_size: int = 200


@dataclass(frozen=True)
class GeneratorConfig:
    """Gemini LLM generator settings."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    temperature: float = 0.3
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class NaiveRAGConfig:
    """Top-level configuration for the Naive RAG pipeline."""

    chunks_dir: Path = field(default_factory=lambda: Path(os.getenv("CHUNKS_DIR", "data/processed/groundtruth")))
    top_k: int = 10
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
