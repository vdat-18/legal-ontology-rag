# Legal Ontology RAG

This repository provides multiple retrieval backends over the same versioned legal-document chunks:

- Graphiti uses Neo4j for temporal entities, relationships, and source provenance.
- Naive RAG uses Milvus for dense-vector retrieval.
- Corrective RAG uses Milvus retrieval plus CRAG grading, refinement, optional web search, and answer generation.
- Attu is an optional Milvus administration UI; it is not part of the application data path.

## Local Infrastructure

Copy `.env.example` to `.env`, replace every credential placeholder, and keep the file uncommitted. Services are grouped by Compose profile so each pipeline starts only its own dependencies:

```powershell
docker compose --profile neo4j up -d neo4j
docker compose --profile milvus up -d milvus
docker compose --profile attu up -d attu
```

Compose waits for etcd and MinIO before starting Milvus, and waits for Milvus before starting Attu. Published ports bind to `127.0.0.1` by default. Override `INFRA_HOST_BIND` only when remote access is intentionally protected by a firewall or private network.

The `app` image is a batch CLI image, not a daemon. Start the required backend first, then run a command explicitly:

```powershell
docker compose --profile milvus run --rm app legal-naive-rag-index data/chunks
docker compose --profile neo4j run --rm app legal-graphiti-index data/chunks --corpus-config data/corpus.json
```

The image contains LibreOffice for legacy DOC/XLS attachment conversion. Pass `--renderer-executable /usr/bin/soffice` to parsing commands that need it. Source code is built into the image so the generated BAML client is not hidden by a host bind mount; rebuild the image after code changes.

## Milvus Data Compatibility

Milvus uses one shared database configured by `MILVUS_DATABASE`. Each retrieval method owns a separate collection inside that database, such as `naive_rag_chunks` and `legal_chunks`, so methods share infrastructure without mixing incompatible schemas or embeddings.

When upgrading an existing environment, rename `NAIVE_RAG_MILVUS_URI` to `MILVUS_URI`, `NAIVE_RAG_MILVUS_TOKEN` to `MILVUS_TOKEN`, and `NAIVE_RAG_MILVUS_DATABASE` to `MILVUS_DATABASE`. No automatic data migration runs: keep the previous database value to reuse existing data, or select `legal_rag` and rebuild the method-specific collections there.

The naive index stores content, chunker, and embedding-model identity metadata with each chunk. Re-running the indexer resumes missing chunks after a partial run. It fails instead of mixing data when the collection schema, vector dimension, index metric, corpus IDs, chunk content, chunker configuration, or embedding revision differs.

Collections created before this metadata was introduced are intentionally incompatible. Back them up, then configure the shared `MILVUS_DATABASE` and a method-specific collection name, or explicitly drop and rebuild the old collection.

Server-backed indexing flushes inserts before final verification. Milvus Lite on Windows currently has a broken flush RPC; set `NAIVE_RAG_MILVUS_FLUSH_AFTER_INSERT=false` only for that local backend.

## Corrective RAG

Corrective RAG uses the shared Docker Milvus instance, not Milvus Lite. Start Milvus before indexing, retrieval, or evaluation:

```powershell
docker compose --profile milvus up -d milvus
```

Use these settings in `.env`:

```env
EMBEDDING_LOCAL_FILES_ONLY=true
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:<MILVUS_ROOT_PASSWORD>
MILVUS_DATABASE=legal_rag
CRAG_MILVUS_COLLECTION_NAME=legal_chunks
MILVUS_INDEX_BATCH_SIZE=32
```

Index the processed chunks once after starting a fresh Milvus database:

```powershell
python -m legal_rag.corrective_rag.api.cli index --data-dir data/processed/groundtruth
```

Start the FastAPI server after Milvus is running and chunks are indexed:

```powershell
python -m legal_rag.corrective_rag.api --host 127.0.0.1 --port 8000
```

Open the interactive API docs at <http://127.0.0.1:8000/docs>.

Run the full CRAG pipeline:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/crag/query `
  -ContentType "application/json" `
  -Body '{"query":"Vuot den do xe may bi phat bao nhieu?","limit":5}'
```

Run vector-only retrieval from Milvus:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/crag/search `
  -ContentType "application/json" `
  -Body '{"query":"muc phat vuot den do xe may","limit":5}'
```

Test direct web search without loading Milvus or the embedding model:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/crag/web-search `
  -ContentType "application/json" `
  -Body '{"query":"muc phat vuot den do xe may nghi dinh 168","limit":3}'
```

Inside the full CRAG pipeline, web search is used only for `INCORRECT` and `AMBIGUOUS` retrieval actions. A `CORRECT` retrieval intentionally keeps `web_search_count` at `0`.

## Operations

- Health: Neo4j exposes its browser health endpoint on port 7474; Milvus exposes health and Prometheus metrics on port 9091. Container logs use bounded rotation.
- Secrets: Neo4j, MinIO, and Milvus have no source-controlled fallback credentials. Milvus authentication is enabled and clients use `MILVUS_TOKEN`.
- Backup: use Neo4j's supported dump/backup tooling for `neo4j_data`. Use a Milvus Backup release compatible with Milvus 2.4 for collections, or stop Milvus and snapshot `etcd_data`, `minio_data`, and `milvus_data` together. A partial snapshot of only one Milvus volume is not a valid backup.
- Restore: restore into an isolated environment first, start the matching database version, run the integration smoke tests, and only then replace the active deployment. Keep the original volumes until the restored corpus row IDs and representative queries are verified.

Authentication environment variables initialize fresh datastores; they do not reliably rotate credentials already persisted in old volumes. Before reusing an existing Neo4j or Milvus volume, rotate its credential through the database's supported administration API and make `.env` match it, or restore the data into freshly initialized secured volumes.

Run opt-in datastore smoke tests against running services with:

```powershell
$env:RUN_INFRA_INTEGRATION="1"
.venv\Scripts\python.exe -m pytest -q tests/integration/test_datastores.py
```

## Checks

Regenerate the local BAML client after changing or pulling files under `baml_src/`:

```powershell
.venv\Scripts\baml-cli.exe generate
```

```powershell
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m pytest -q
docker compose config --quiet
```
