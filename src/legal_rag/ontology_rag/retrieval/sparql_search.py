"""Search ontology directly by keywords via SPARQL."""

from __future__ import annotations

import re
from collections import defaultdict

from loguru import logger
from rdflib import Graph

from legal_rag.ontology_rag.models import OntologyInstance

_TOKEN_PATTERN = re.compile(r"[\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", re.UNICODE)
_MAX_INSTANCES = 50

# Vietnamese function words plus corpus-generic terms that appear in almost every rule label;
# matching on them only adds noise, so they are dropped before SPARQL label matching.
_STOPWORDS = frozenset(
    {
        "khong",
        "không",
        "khi",
        "khiến",
        "cac",
        "các",
        "cua",
        "của",
        "cho",
        "va",
        "và",
        "hoac",
        "hoặc",
        "la",
        "là",
        "bi",
        "bị",
        "voi",
        "với",
        "nhung",
        "những",
        "thi",
        "thì",
        "do",
        "đó",
        "nay",
        "này",
        "muc",
        "mức",
        "phat",
        "phạt",
        "bao",
        "nhieu",
        "nhiêu",
        "xe",
        "dieu",
        "điều",
        "khien",
        "khiển",
        "nguoi",
        "người",
        "tai",
        "tại",
        "tren",
        "trên",
        "duong",
        "đường",
        "bo",
        "bộ",
        "giao",
        "thong",
        "thông",
        "tham",
        "gia",
    }
)


class SPARQLSearch:
    """Query ontology by keywords."""

    def __init__(self, graph: Graph) -> None:
        """Initialize with ontology graph."""
        self._graph = graph

    def search(self, keywords: list[str]) -> list[OntologyInstance]:
        """Find instances whose labels contain query tokens, best matches first.

        Whole questions never substring-match a short label, so each keyword is tokenized
        into lowercase terms matched with OR semantics; results are then ordered by how
        many tokens each label contains.
        """
        tokens = sorted({token for keyword in keywords for token in self._tokenize(keyword)})
        if not tokens:
            return []

        filters = " || ".join([f'CONTAINS(LCASE(STR(?label)), "{token}")' for token in tokens])
        query = f"""
        PREFIX lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#>
        PREFIX lorsp: <https://w3id.org/legal-ontology-rag/ontology/source-provenance#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?instance ?label ?chunk_id WHERE {{
            ?instance a lornr:PenaltyRule ;
                      rdfs:label ?label .
            OPTIONAL {{
                ?instance lorsp:sourcedFromChunk ?chunk .
                ?chunk dcterms:identifier ?chunk_id .
            }}
            FILTER({filters})
        }}
        LIMIT {_MAX_INSTANCES * 2}
        """
        labels: dict[object, str] = {}
        chunk_ids: dict[object, list[str]] = defaultdict(list)
        for row in self._graph.query(query):
            labels[row.instance] = str(row.label)
            if row.chunk_id is not None:
                chunk_ids[row.instance].append(str(row.chunk_id))

        scored = [
            (sum(1 for token in tokens if token in label.lower()), str(uri), uri) for uri, label in labels.items()
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        instances = [
            OntologyInstance(uri=uri, type="PenaltyRule", source_chunk_ids=tuple(chunk_ids.get(uri, ())))
            for _, _, uri in scored[:_MAX_INSTANCES]
        ]
        logger.info("SPARQL search found {} instances for keywords: {}", len(instances), keywords)
        return instances

    @staticmethod
    def _tokenize(keyword: str) -> list[str]:
        """Split one keyword into lowercase content tokens, dropping stopwords."""
        tokens = [token.lower() for token in _TOKEN_PATTERN.findall(keyword) if len(token) > 1]
        return [token for token in tokens if token not in _STOPWORDS]
