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

Install the CRAG API, Milvus, embedding, and Streamlit runtime dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[corrective-rag]"
```

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
CRAG_MILVUS_INDEX_TYPE=IVF_FLAT
CRAG_MILVUS_METRIC_TYPE=COSINE
CRAG_MILVUS_NLIST=128
CRAG_MILVUS_NPROBE=16
CRAG_MILVUS_FLUSH_AFTER_INSERT=true
MILVUS_INDEX_BATCH_SIZE=32
CRAG_WEB_SEARCH_ALLOWED_DOMAINS=chinhphu.vn,vanban.chinhphu.vn,moj.gov.vn
```

Corrective RAG validates its Milvus collection schema, vector dimension, vector
metric, chunk content hashes, chunker config hash, and embedding
model/revision metadata before reusing indexed rows. Collections created by an
older CRAG indexer without these metadata fields are intentionally rejected;
drop and rebuild `CRAG_MILVUS_COLLECTION_NAME` rather than mixing stale chunks.

When `CRAG_WEB_SEARCH_ENABLED=true`, web search requires a non-empty domain
allowlist. Only HTTP(S) result URLs are fetched, redirects are checked again at
their final URL, and loopback, private, link-local, reserved, or multicast
addresses are blocked.

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

## Ontology Population

Populate the ontology from screened legal-document chunks before any GraphRAG indexing:

```powershell
uv run --extra dev legal-populate-ontology `
  data/chunks/document.jsonl `
  data/benchmark_synthesis/screening/document.jsonl `
  data/benchmark_synthesis/metadata/document.jsonl `
  ontology/population_contract.yaml `
  data/ontology/population/document.jsonl `
  data/ontology/progress/document.jsonl `
  data/ontology/graphs/document.ttl
```

The command persists reviewable typed records and writes the Turtle graph only after SHACL conformance. See
`docs/ontology_population_pipeline.md` for extraction boundaries, evidence requirements, resume behavior, and semantic
limitations.

After chunk-local population, assemble rule components that span nearby chunks with the reviewed three-command flow:

```powershell
uv run --extra dev legal-build-assembly-candidates <chunks.jsonl> <population.jsonl> `
  ontology/assembly_contract.yaml <candidates.json>
uv run --extra dev legal-propose-rule-assemblies <chunks.jsonl> <metadata.jsonl> <population.jsonl> `
  <candidates.json> ontology/assembly_contract.yaml <proposals.jsonl> <progress.jsonl> <reviews.jsonl>
uv run --extra dev legal-finalize-rule-assemblies <chunks.jsonl> <metadata.jsonl> <population.jsonl> `
  <proposals.jsonl> <reviews.jsonl> ontology/population_contract.yaml ontology/assembly_contract.yaml `
  <canonical.jsonl> <canonical.ttl>
```

Only `legal-propose-rule-assemblies` calls the LLM. Every proposal must be explicitly accepted or rejected by a human
reviewer before the offline finalizer can write SHACL-validated canonical outputs.

After every document has canonical within-document rules, resolve explicit cross-document legal changes:

```powershell
uv run --extra dev legal-build-change-candidates <corpus_manifest.json> `
  ontology/change_resolution_contract.yaml <candidates.json>
uv run --extra dev legal-propose-legal-changes <corpus_manifest.json> <candidates.json> `
  ontology/change_resolution_contract.yaml <proposals.jsonl> <progress.jsonl> <reviews.jsonl>
uv run --extra dev legal-finalize-legal-changes <corpus_manifest.json> <proposals.jsonl> <reviews.jsonl> `
  ontology/population_contract.yaml ontology/change_resolution_contract.yaml `
  <canonical_population.jsonl> <legal_change_events.jsonl> <canonical_graph.ttl>
```

Only the proposal command calls the LLM. Candidate generation is exhaustive and bounded rather than similarity-based;
human acceptance owns change type, effective date, and rule-version linkage. See
`docs/ontology_population_pipeline.md` for the manifest schema and review workflow.

Resolve legal exceptions and grouped provision citations after canonical rule validity is available:

```powershell
uv run --extra dev legal-build-exception-candidates <corpus_manifest.json> <canonical_population.jsonl> `
  ontology/exception_resolution_contract.yaml <candidates.json>
uv run --extra dev legal-propose-legal-exceptions <corpus_manifest.json> <canonical_population.jsonl> `
  <candidates.json> ontology/exception_resolution_contract.yaml <proposals.jsonl> <progress.jsonl> <reviews.jsonl>
uv run --extra dev legal-finalize-legal-exceptions <corpus_manifest.json> <canonical_population.jsonl> `
  <legal_change_events.jsonl> <proposals.jsonl> <reviews.jsonl> ontology/population_contract.yaml `
  ontology/exception_resolution_contract.yaml <legal_exceptions.jsonl> <canonical_graph.ttl>
```

Only proposal generation calls the LLM. Human review owns exception scope, grouped citation resolution, and
version-specific redirect targets; finalization writes outputs only after SHACL conformance.

Normalize reusable conduct, role, and vehicle concepts after exception resolution:

```powershell
uv run --extra dev legal-build-concept-candidates <canonical_population.jsonl> `
  ontology/concept_resolution_contract.yaml <candidates.json>
uv run --extra dev legal-propose-concept-relations <corpus_manifest.json> <candidates.json> `
  ontology/concept_resolution_contract.yaml <proposals.jsonl> <progress.jsonl> <reviews.jsonl>
uv run --extra dev legal-finalize-concept-resolution <corpus_manifest.json> <canonical_population.jsonl> `
  <legal_change_events.jsonl> <legal_exceptions.jsonl> <candidates.json> <proposals.jsonl> <reviews.jsonl> `
  ontology/population_contract.yaml ontology/concept_resolution_contract.yaml `
  <canonical_concepts.jsonl> <concept_relations.jsonl> <canonical_graph.ttl>
```

Candidate generation exhaustively pairs concepts only within the same kind and makes no semantic decision. Only the
proposal command calls the LLM. Accepted `SameConcept` decisions remap rules to a shared canonical identity; reviewed
broader, narrower, and related decisions become SKOS relations without becoming OWL subclass or equivalence axioms.

Review the ontology's external vocabulary boundary locally with:

```powershell
uv run --extra dev legal-review-ontology-reuse `
  ontology/external_reuse_registry.yaml `
  data/ontology/qa/external_reuse_review.json
```

The registry records source, version, license, attribution, compatibility, boundaries, and exact term IRIs for Dublin
Core, PROV-O, SKOS, and OWL-Time. It also records ELI as candidate-only and the remaining legal resources as reviewed
references or patterns. The command rejects unregistered semantic IRIs, candidate/reference namespace use, missing
foundational declarations, and external OWL imports. Technical checks pass; project-owner acceptance remains explicit.

Run the ontology semantic release gates with pinned ROBOT and HermiT tooling:

```powershell
uv run --extra dev legal-validate-ontology-semantics `
  ontology/semantic_qa_contract.yaml `
  data/ontology/qa/artifacts `
  data/ontology/qa/semantic_qa_report.json
```

This command uses the pinned `obolibrary/robot:v1.9.10` Docker image with networking disabled. It validates the complete
OWL 2 DL import closure, runs HermiT, audits unsatisfiable classes and unintended inferred equivalences, executes the
ROBOT quality report, and records traceability/CQ coverage. The report is written before the command fails an open gate.
The report also binds its result to a SHA-256 digest of the complete semantic input closure, so a report becomes stale
when an ontology module, SHACL graph, CQ query, fixture, registry, catalog, policy, or semantic contract changes.
The current ontology passes every semantic gate, including ROBOT's ontology-license rule. Public release remains
blocked by namespace resolution, project-owner acceptance of the completed external-reuse review, and independent legal
review.

After every semantic gate passes, build the immutable V0.1 distribution with:

```powershell
uv run --extra dev legal-package-ontology-release `
  ontology/release_contract.yaml `
  data/ontology/qa/semantic_qa_report.json `
  data/ontology/releases/v0.1.0 `
  data/ontology/releases/v0.1.0/release_report.json
```

The release command verifies the semantic-input digest, ontology/version IRIs, license metadata and file, the complete
external-reuse registry against the current graph, explicit human approvals, and the complete source inventory. It
always writes a readiness report
before failing an open gate. Once all gates pass, it emits a reproducible ZIP containing a machine-readable manifest and
a separate `.sha256` checksum. Runtime data, populated instances, retrieval indexes, and GraphRAG artifacts are excluded.

Build the proposed canonical namespace publication before requesting W3ID registration:

```powershell
uv run --extra dev legal-build-ontology-publication `
  ontology/namespace_publication.yaml `
  data/ontology/publication/site `
  data/ontology/publication/report.json
```

This creates 14 canonical/versioned HTML and byte-identical Turtle endpoints plus a SHA-256 publication manifest. It
also validates the copy-ready W3ID `.htaccess` and `README.md` bundle. The proposed GitHub Pages target must be deployed
and externally verified before the W3ID pull request is submitted. See `docs/ontology_namespace_publication.md`.

The manual `Deploy ontology publication` GitHub Actions workflow publishes the generated site with least-privilege Pages
permissions and commit-pinned actions. After deployment, run the bounded live verifier in `deployment` mode; after the
W3ID pull request is merged, run it again in `w3id` mode to verify all 28 content-negotiated requests and redirects.

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
uv run --extra dev legal-validate-ontology-semantics ontology/semantic_qa_contract.yaml `
  data/ontology/qa/artifacts data/ontology/qa/semantic_qa_report.json
uv run --extra dev legal-package-ontology-release ontology/release_contract.yaml `
  data/ontology/qa/semantic_qa_report.json data/ontology/releases/v0.1.0 `
  data/ontology/releases/v0.1.0/release_report.json
uv run --extra dev legal-build-ontology-publication ontology/namespace_publication.yaml `
  data/ontology/publication/site data/ontology/publication/report.json
uv run --extra dev legal-verify-ontology-publication ontology/namespace_publication.yaml `
  deployment data/ontology/publication/deployment_verification.json
docker compose config --quiet
```
