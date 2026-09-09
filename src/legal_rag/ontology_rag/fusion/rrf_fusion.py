"""RRF fusion for hybrid search."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from loguru import logger

from legal_rag.naive_rag.retriever import RetrievedChunk
from legal_rag.ontology_rag.models import OntologyInstance


class RRFFusion:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60, ontology_weight: float = 2.0) -> None:
        """Initialize RRF parameters."""
        self._k = k
        self._ontology_weight = ontology_weight

    def fuse(
        self,
        semantic_chunks: Sequence[RetrievedChunk],
        keyword_chunks: Sequence[RetrievedChunk],
        ontology_instances: Sequence[OntologyInstance],
    ) -> list[OntologyInstance]:
        """Merge three sources into ranked instances."""
        chunk_scores = self._compute_chunk_scores(semantic_chunks, keyword_chunks, ontology_instances)
        all_instances = self._collect_all_instances(semantic_chunks, ontology_instances)
        instance_scores = self._aggregate_instance_scores(all_instances, chunk_scores)
        ranked_instances = sorted(instance_scores.items(), key=lambda x: -x[1])
        result = [inst for inst, _ in ranked_instances]
        logger.info("RRF fused {} instances from {} sources", len(result), 3)
        return result

    def _compute_chunk_scores(
        self,
        semantic_chunks: Sequence[RetrievedChunk],
        keyword_chunks: Sequence[RetrievedChunk],
        ontology_instances: Sequence[OntologyInstance],
    ) -> dict[str, float]:
        """Compute RRF scores for chunk IDs."""
        scores: dict[str, float] = defaultdict(float)

        for rank, chunk in enumerate(semantic_chunks, start=1):
            scores[chunk.chunk_id] += 1.0 / (self._k + rank)

        for rank, chunk in enumerate(keyword_chunks, start=1):
            scores[chunk.chunk_id] += 1.0 / (self._k + rank)

        for rank, instance in enumerate(ontology_instances, start=1):
            for chunk_id in instance.source_chunk_ids:
                scores[chunk_id] += self._ontology_weight / (self._k + rank)

        return dict(scores)

    def _collect_all_instances(
        self, chunks: Sequence[RetrievedChunk], ontology_instances: Sequence[OntologyInstance]
    ) -> set[OntologyInstance]:
        """Collect all unique instances from chunks and direct ontology search."""
        all_instances: set[OntologyInstance] = set(ontology_instances)
        chunk_instance_map: dict[str, OntologyInstance] = {}

        for instance in ontology_instances:
            for chunk_id in instance.source_chunk_ids:
                chunk_instance_map[chunk_id] = instance

        for chunk in chunks:
            if chunk.chunk_id in chunk_instance_map:
                all_instances.add(chunk_instance_map[chunk.chunk_id])

        return all_instances

    def _aggregate_instance_scores(
        self, instances: set[OntologyInstance], chunk_scores: dict[str, float]
    ) -> dict[OntologyInstance, float]:
        """Aggregate chunk scores to instance scores."""
        instance_scores: dict[OntologyInstance, float] = {}
        for instance in instances:
            score = sum(chunk_scores.get(cid, 0.0) for cid in instance.source_chunk_ids)
            instance_scores[instance] = score
        return instance_scores
