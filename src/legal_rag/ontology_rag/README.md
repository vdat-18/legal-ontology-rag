# Ontology-Enhanced RAG Module

Ontology-based RAG system for Vietnamese traffic law, implementing a 4-phase hybrid retrieval architecture.

## Architecture Overview

```
Query → typed rewrite and temporal plan
     → semantic/BM25L chunk retrieval + direct SPARQL rule search
     → per-channel RRF → complete PenaltyRule retrieval
     → certainty-aware temporal resolution → exact source-chunk lookup
     → structured rule + verbatim source context → answer
```

## Components

### Phase 1: Index Modules

- **OntologyIndex**: Loads RDF/OWL ontologies from TTL files
- **ChunkIndex**: Maps chunks to ontology instances
- **TemporalIndex**: Extracts valid-from/valid-to periods

### Phase 2: Retrieval Modules

- **ChunkLinker**: Links retrieved chunk evidence only to explicitly typed penalty rules
- **InstanceRetriever**: Reads complete multi-valued PenaltyRule objects without flattening RDF joins
- **SPARQLSearch**: Keyword search against ontology
- **TemporalResolver**: Classifies rules as valid, invalid, or indeterminate for each query date

### Phase 3: Fusion

- **RRFFusion**: Reciprocal Rank Fusion with ontology_weight=2.0, k=60

### Phase 4: Pipeline

- **ContextBuilder**: Keeps structured applicability, consequences, exceptions, temporal state, and verbatim source text
- **OntologyRAGPipeline**: End-to-end pipeline orchestrating all modules

## Usage

Install the retrieval dependencies (and the API dependencies when serving HTTP):

```bash
uv sync --extra naive-rag --extra api
```

### CLI

```bash
legal-ontology-rag-query --ontology path/to/canonical-ontology.ttl "Mức phạt không đội mũ bảo hiểm?"
```

Options:
- `--top-k N`: Override the number of selected rule units
- `--query-date YYYY-MM-DD`: Explicit query date; this overrides a date inferred from the question

### API

```python
from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
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
export ONTOLOGY_PATHS="path/to/canonical-ontology.ttl"
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

1. **Query Planning**: Rewrite a follow-up as a standalone legal question and identify current, at-date, or comparison intent.
2. **Triple Retrieval**:
   - Vector search via Milvus (semantic)
   - BM25L search (keyword)
   - SPARQL search (ontology)
3. **RRF Fusion**: Score semantic, keyword, and direct ontology channels independently.
4. **Instance Retrieval**: Read applicability, all consequences, conditions, exceptions, dates, and component provenance.
5. **Temporal Resolution**: Resolve all requested dates before applying `top_k`. Validity intervals use `[valid_from, valid_to)`.
6. **Source Resolution**: Fetch every selected rule's exact source chunks by primary key from the same Milvus collection.
7. **Context Building**: Exclude fallback evidence for rejected candidate rules, then separate normalized ontology
   facts from source wording that may be quoted. The character budget reserves one available exact source chunk per
   selected rule before adding further sources and unrelated fallback chunks in retrieval order.
8. **Answer Generation**: Generate from selected rules and fallback retrieved chunks.

An absent validity bound remains unknown. Such a rule is marked `indeterminate`; it may be supplied as qualified
evidence, but the answer must not claim that it is currently effective solely because its end date is absent.

## Configuration

Environment variables:

- `ONTOLOGY_PATHS`: Comma-separated paths to ontology TTL files
- `RAG_API_KEY`: API key for FastAPI endpoint
- `NAIVE_RAG_ANSWER_MAX_CONTEXT_CHARACTERS`: Maximum combined chunk ID, source name, and text characters supplied
  to answer generation (default `120000`)
- Standard LLM/Milvus settings from `Settings`, `MilvusSettings`, `NaiveRagSettings`

## Testing

```bash
pytest tests/ontology_rag/ -v
```

The tests cover index, retrieval, fusion, temporal boundaries, context, pipeline, API, CLI, and evaluation contracts.
