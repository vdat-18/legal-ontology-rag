# Ontology-Enhanced RAG Module

Ontology-based RAG system for Vietnamese traffic law, implementing a 4-phase hybrid retrieval architecture.

## Architecture Overview

```
Query → Triple Retrieval (Vector/BM25L/SPARQL) → RRF Fusion
     → ChunkLinker → InstanceRetriever → TemporalResolver
     → ContextBuilder → Generate Answer
```

## Components

### Phase 1: Index Modules

- **OntologyIndex**: Loads RDF/OWL ontologies from TTL files
- **ChunkIndex**: Maps chunks to ontology instances
- **TemporalIndex**: Extracts valid-from/valid-to periods

### Phase 2: Retrieval Modules

- **ChunkLinker**: Links retrieved chunks to ontology instances
- **InstanceRetriever**: Fetches full PenaltyRule objects via SPARQL
- **SPARQLSearch**: Keyword search against ontology
- **TemporalResolver**: Filters rules by query date

### Phase 3: Fusion

- **RRFFusion**: Reciprocal Rank Fusion with ontology_weight=2.0, k=60

### Phase 4: Pipeline

- **ContextBuilder**: Builds structured JSON from rules and chunks
- **OntologyRAGPipeline**: End-to-end pipeline orchestrating all modules

## Usage

### CLI

```bash
legal-ontology-rag-query --ontology ontology/rules.ttl "Mức phạt không đội mũ bảo hiểm?"
```

Options:
- `--top-k N`: Override number of retrieved chunks
- `--query-date YYYY-MM-DD`: Query date for temporal filtering

### API

```python
from legal_rag.ontology_rag import OntologyRAGPipeline
from legal_rag.settings import Settings, MilvusSettings, NaiveRagSettings
from pathlib import Path

pipeline = OntologyRAGPipeline(
    settings=Settings.from_environment(),
    milvus_settings=MilvusSettings.from_environment(),
    naive_rag_settings=NaiveRagSettings.from_environment(),
    ontology_paths=[Path("ontology/rules.ttl")],
)

result = pipeline.run("Không đội mũ bảo hiểm khi lái xe máy?")
print(f"Answer: {result.answer}")
for rule in result.rules:
    print(f"Rule: {rule.conduct_label_vi}")
```

### FastAPI

Start the API server:

```bash
export ONTOLOGY_PATHS="ontology/rules.ttl"
export RAG_API_KEY="your-key"
legal-ontology-rag-api
```

Query:

```bash
curl -X POST http://localhost:8000/query \
  -H "X-Api-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Mức phạt vượt đèn đỏ?"}'
```

## Data Flow

1. **Query Input**: Vietnamese legal question
2. **Triple Retrieval**:
   - Vector search via Milvus (semantic)
   - BM25L search (keyword)
   - SPARQL search (ontology)
3. **RRF Fusion**: Combine and rank results
4. **Linking**: Map chunks → ontology instances
5. **Instance Retrieval**: Fetch full PenaltyRule objects
6. **Temporal Filtering**: Filter by query_date
7. **Context Building**: Structure as JSON
8. **Answer Generation**: LLM generates answer

## Configuration

Environment variables:

- `ONTOLOGY_PATHS`: Comma-separated paths to ontology TTL files
- `RAG_API_KEY`: API key for FastAPI endpoint
- Standard LLM/Milvus settings from `Settings`, `MilvusSettings`, `NaiveRagSettings`

## Testing

```bash
pytest tests/ontology_rag/ -v
```

13 tests covering index, retrieval, fusion, context, and pipeline components.
