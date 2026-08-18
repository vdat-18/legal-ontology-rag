# Ontology Namespace Publication

This workflow prepares the canonical `https://w3id.org/legal-ontology-rag/` namespace without claiming that the
identifier is live before its redirect is merged by the W3ID maintainers. It covers ontology construction artifacts
only; populated data, retrieval indexes, GraphRAG, and Graphiti are outside the publication site.

## Versioned Contract

`ontology/namespace_publication.yaml` is the source of truth for:

- the W3ID namespace root and proposed GitHub Pages deployment base;
- the root ontology, five modules, and SHACL graph;
- canonical and immutable version IRIs;
- source Turtle files and publication paths;
- the maintainer identity, redirect status, license, and W3ID bundle paths.

The current deployment base is `https://vdat-18.github.io/legal-ontology-rag/`. It is a proposed target until the
repository is public, GitHub Pages is configured, and every generated URL is externally verified.

## Build The Static Site

Run:

```powershell
uv run --extra dev legal-build-ontology-publication `
  ontology/namespace_publication.yaml `
  data/ontology/publication/site `
  data/ontology/publication/report.json
```

The output directory must not already exist. This prevents stale files from surviving a publication build. The command
validates the Turtle ontology/version IRIs, XML catalog mappings, endpoint uniqueness, CC BY 4.0 metadata, and exact W3ID
pull-request bundle before writing anything.

A successful build contains:

```text
site/
|-- publication-manifest.json
|-- ontology/
|   |-- index.html
|   |-- ontology.ttl
|   |-- 0.1.0/
|   `-- <module paths>/
`-- shapes/
    |-- index.html
    |-- ontology.ttl
    `-- 0.5.0/
```

Canonical and versioned Turtle files are byte-identical copies of their reviewed repository sources. Landing pages are
self-contained and link the canonical IRI, version IRI, Turtle representation, project, and CC BY 4.0 license. The JSON
manifest records all 14 endpoints, expected content types, source paths, and SHA-256 digests.

## Deploy Before W3ID

Publish the generated `site/` directory at the exact base URL in the contract. For the proposed GitHub Pages target:

1. Make the repository and publication target accessible to the intended audience.
2. In repository settings, configure Pages to use **GitHub Actions** as its source.
3. Run the manually triggered `Deploy ontology publication` workflow from `.github/workflows/ontology-pages.yml`.
4. Confirm that the workflow reports `https://vdat-18.github.io/legal-ontology-rag/` as its environment URL.
5. Verify the deployed representations with:

```powershell
uv run --extra dev legal-verify-ontology-publication `
  ontology/namespace_publication.yaml `
  deployment `
  data/ontology/publication/deployment_verification.json
```

The workflow has no push trigger. It uses least-privilege Pages permissions and commit-pinned official actions. The live
verifier makes 28 sequential bounded HTTPS requests, compares every HTML and Turtle response byte-for-byte with the
current reviewed sources, and checks the expected media type. It uses the contract timeout and response-size limit.

6. Confirm that every manifest `html_url` returns HTML and every `turtle_url` returns the recorded Turtle bytes.
7. Confirm versioned paths cannot be replaced by a later release deployment.
8. Change the contract base URL and regenerate everything if a different hosting target is selected.

Do not submit W3ID rules that redirect to missing, private, or unverified targets.

## Prepare The W3ID Pull Request

The copy-ready directory is:

```text
ontology/publication/w3id/legal-ontology-rag/
|-- .htaccess
`-- README.md
```

Copy that directory to the root of a fork of `perma-id/w3id.org`, preserving the directory name. The bundle follows the
official W3ID requirements: `.htaccess` contains 303 redirect rules and content negotiation, while `README.md` documents
the identifier purpose, maintainer GitHub account, project, and license.

Before opening the pull request:

1. Re-run the local publication command from a clean output path.
2. Verify all deployment targets from `publication-manifest.json` over HTTPS.
3. Test browser requests and `Accept: text/turtle` requests separately.
4. Confirm unknown paths return 404 rather than an unrelated ontology.
5. Use a descriptive W3ID commit and pull-request title such as `Add legal-ontology-rag identifiers`.

After the W3ID pull request is merged, verify the actual 303 responses and exact `Location` headers:

```powershell
uv run --extra dev legal-verify-ontology-publication `
  ontology/namespace_publication.yaml `
  w3id `
  data/ontology/publication/w3id_verification.json
```

## Approval Boundary

Local generation proves only that repository sources, catalog mappings, publication paths, and the proposed W3ID bundle
are internally consistent. Set `namespace_resolution` to approved only after:

- the static deployment is public and verified;
- the W3ID pull request is merged;
- canonical and version IRIs return 303 redirects to the expected representations;
- the verification evidence is recorded in `ontology/release_contract.yaml`.
