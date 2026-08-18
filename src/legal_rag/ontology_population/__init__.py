"""Source-grounded population of ontology V0.1 instance graphs."""

from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import materialize_population_graph
from legal_rag.ontology_population.models import OntologyPopulationRecord

__all__ = ["OntologyPopulationRecord", "PopulationSettings", "materialize_population_graph"]
