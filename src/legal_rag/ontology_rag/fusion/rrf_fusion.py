"""RRF fusion for hybrid search."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from loguru import logger
from rdflib import URIRef

from legal_rag.naive_rag.models import RetrievedChunk
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
        *,
        semantic_instances: Sequence[OntologyInstance] | None = None,
        keyword_instances: Sequence[OntologyInstance] | None = None,
    ) -> list[OntologyInstance]:
        """Merge chunk retrieval and ontology matches into ranked instances.

        ``ontology_instances`` contains direct ontology-search results. Callers can pass
        each chunk channel's independently linked instances to preserve channel rank and
        ensure only direct matches receive the ontology weight. The optional arguments
        retain compatibility with callers that only have chunk lists.
        """
        semantic_channel = semantic_instances
        if semantic_channel is None:
            semantic_channel = self._instances_for_chunks(semantic_chunks, ontology_instances)
        keyword_channel = keyword_instances
        if keyword_channel is None:
            keyword_channel = self._instances_for_chunks(keyword_chunks, ontology_instances)

        channels = (
            (semantic_channel, 1.0),
            (keyword_channel, 1.0),
            (ontology_instances, self._ontology_weight),
        )
        instances_by_uri: dict[URIRef, OntologyInstance] = {}
        instance_scores: dict[URIRef, float] = defaultdict(float)
        for instances, weight in channels:
            self._score_channel(instances, weight, instances_by_uri, instance_scores)

        ranked_uris = sorted(instance_scores, key=lambda uri: (-instance_scores[uri], str(uri)))
        result = [instances_by_uri[uri] for uri in ranked_uris]
        logger.info("RRF fused {} instances from {} sources", len(result), 3)
        return result

    def _instances_for_chunks(
        self,
        chunks: Sequence[RetrievedChunk],
        instances: Sequence[OntologyInstance],
    ) -> list[OntologyInstance]:
        """Infer one ranked instance channel from chunk evidence IDs."""
        instances_by_chunk: dict[str, list[OntologyInstance]] = defaultdict(list)
        for instance in instances:
            for chunk_id in instance.source_chunk_ids:
                instances_by_chunk[chunk_id].append(instance)
        ranked: list[OntologyInstance] = []
        for chunk in chunks:
            for chunk_id in chunk.effective_evidence_chunk_ids:
                ranked.extend(instances_by_chunk.get(chunk_id, ()))
        return ranked

    def _score_channel(
        self,
        instances: Sequence[OntologyInstance],
        weight: float,
        instances_by_uri: dict[URIRef, OntologyInstance],
        instance_scores: dict[URIRef, float],
    ) -> None:
        """Add one RRF term per unique URI and merge its source provenance."""
        seen: set[URIRef] = set()
        rank = 0
        for instance in instances:
            previous = instances_by_uri.get(instance.uri)
            if previous is None:
                instances_by_uri[instance.uri] = instance
            else:
                source_chunk_ids = tuple(dict.fromkeys((*previous.source_chunk_ids, *instance.source_chunk_ids)))
                instances_by_uri[instance.uri] = OntologyInstance(
                    uri=instance.uri,
                    type=previous.type,
                    source_chunk_ids=source_chunk_ids,
                )
            if instance.uri in seen:
                continue
            seen.add(instance.uri)
            rank += 1
            instance_scores[instance.uri] += weight / (self._k + rank)
