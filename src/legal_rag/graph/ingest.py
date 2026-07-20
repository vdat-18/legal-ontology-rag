"""Ingestion orchestration: add time-stamped episodes to the Graphiti graph.

Each episode's ``reference_time`` is the source document's effective date, so
Graphiti's bi-temporal model supersedes provisions changed by later amendments
without deleting history.
"""

from __future__ import annotations

from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from loguru import logger

from legal_rag.config import Settings
from legal_rag.graph.client import build_graphiti
from legal_rag.ingestion import iter_episodes
from legal_rag.ontology import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

_EXTRACTION_INSTRUCTIONS = (
    "Văn bản là quy phạm pháp luật xử phạt vi phạm giao thông đường bộ Việt Nam. "
    "Trích các thực thể: hành vi vi phạm (TrafficViolation), loại phương tiện (Vehicle), "
    "chế tài/mức phạt (Penalty: mức tiền, tước GPLX, trừ điểm), văn bản (LegalDocument). "
    "Với mỗi hành vi, nối tới chế tài (HAS_PENALTY) và ghi trích dẫn Điều/Khoản/Điểm. "
    "Giữ nguyên số tiền và số điểm trừ chính xác; không suy diễn ngoài văn bản."
)


async def ingest(settings: Settings, *, reset: bool = False) -> int:
    """Build indices then add every in-scope episode. Returns the count added."""
    graphiti = build_graphiti(settings)
    try:
        if reset:
            logger.warning("Clearing existing graph data")
            await clear_data(graphiti.driver)
        await graphiti.build_indices_and_constraints()
        added = 0
        for episode in iter_episodes(settings):
            await graphiti.add_episode(
                name=episode.name,
                episode_body=episode.body,
                source=EpisodeType.text,
                source_description=episode.source_description,
                reference_time=episode.reference_time,
                group_id=settings.group_id,
                entity_types=ENTITY_TYPES,
                edge_types=EDGE_TYPES,
                edge_type_map=EDGE_TYPE_MAP,
                custom_extraction_instructions=_EXTRACTION_INSTRUCTIONS,
            )
            added += 1
        logger.success("Ingested {} episode(s) into group '{}'", added, settings.group_id)
        return added
    finally:
        await graphiti.close()
