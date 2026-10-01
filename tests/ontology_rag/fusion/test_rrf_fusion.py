"""Tests for RRF fusion module."""

from __future__ import annotations

from rdflib import URIRef

from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.ontology_rag.fusion.rrf_fusion import RRFFusion
from legal_rag.ontology_rag.models import OntologyInstance


def test_rrf_fusion_merges_sources() -> None:
    """RRF fusion merges three sources into ranked list."""
    semantic_chunks = [
        RetrievedChunk(chunk_id="chunk_001", text="text1", source_name="doc1", kind="paragraph", score=0.95),
        RetrievedChunk(chunk_id="chunk_002", text="text2", source_name="doc1", kind="paragraph", score=0.90),
    ]
    keyword_chunks = [
        RetrievedChunk(chunk_id="chunk_001", text="text1", source_name="doc1", kind="paragraph", score=3.5),
        RetrievedChunk(chunk_id="chunk_003", text="text3", source_name="doc1", kind="paragraph", score=2.8),
    ]
    ontology_instances = [
        OntologyInstance(uri=URIRef("rule_001"), type="PenaltyRule", source_chunk_ids=("chunk_001",)),
        OntologyInstance(uri=URIRef("rule_002"), type="PenaltyRule", source_chunk_ids=("chunk_002",)),
    ]

    fusion = RRFFusion(k=60, ontology_weight=2.0)
    ranked = fusion.fuse(semantic_chunks, keyword_chunks, ontology_instances)

    assert len(ranked) >= 1
    assert all(isinstance(inst, OntologyInstance) for inst in ranked)


def test_rrf_ontology_boosting() -> None:
    """Ontology instances get boosted scores."""
    semantic_chunks = [
        RetrievedChunk(chunk_id="chunk_001", text="text1", source_name="doc1", kind="paragraph", score=0.95),
    ]
    keyword_chunks = []
    ontology_instances = [
        OntologyInstance(uri=URIRef("rule_001"), type="PenaltyRule", source_chunk_ids=("chunk_001",)),
    ]

    fusion = RRFFusion(k=60, ontology_weight=2.0)
    ranked = fusion.fuse(semantic_chunks, keyword_chunks, ontology_instances)

    assert len(ranked) > 0


def test_rrf_empty_sources() -> None:
    """RRF fusion handles empty sources."""
    fusion = RRFFusion()
    ranked = fusion.fuse([], [], [])
    assert len(ranked) == 0


def test_rrf_scores_distinct_channels_and_merges_provenance() -> None:
    """Only direct matches receive ontology weight and repeated URIs merge."""
    semantic_only = OntologyInstance(
        uri=URIRef("urn:semantic"), type="PenaltyRule", source_chunk_ids=("semantic-source",)
    )
    keyword_only = OntologyInstance(uri=URIRef("urn:keyword"), type="PenaltyRule", source_chunk_ids=("keyword-source",))
    direct_only = OntologyInstance(uri=URIRef("urn:direct"), type="PenaltyRule", source_chunk_ids=("direct-source",))
    shared_semantic = OntologyInstance(
        uri=URIRef("urn:shared"), type="PenaltyRule", source_chunk_ids=("shared-semantic",)
    )
    shared_keyword = OntologyInstance(
        uri=URIRef("urn:shared"), type="PenaltyRule", source_chunk_ids=("shared-keyword",)
    )
    shared_direct = OntologyInstance(uri=URIRef("urn:shared"), type="PenaltyRule", source_chunk_ids=("shared-direct",))

    ranked = RRFFusion(k=60, ontology_weight=2.0).fuse(
        [],
        [],
        [direct_only, shared_direct],
        semantic_instances=[semantic_only, shared_semantic],
        keyword_instances=[keyword_only, shared_keyword],
    )

    assert [str(instance.uri) for instance in ranked] == [
        "urn:shared",
        "urn:direct",
        "urn:keyword",
        "urn:semantic",
    ]
    assert ranked[0].source_chunk_ids == ("shared-semantic", "shared-keyword", "shared-direct")
