# third_party

Vendored external source, committed into this repository.

## graphiti

- **Source:** https://github.com/getzep/graphiti
- **Version:** 0.29.2
- **Vendored commit:** `5e2be0faf7038a5b40e700d757b2c337e96b3a05`
- **License:** Apache-2.0 (see `graphiti/LICENSE`)

The upstream `.git` history and the large `tests/evals/` evaluation datasets
were removed to keep this repository light; the library source
(`graphiti_core/`) and packaging files are intact.

Install the vendored copy for development:

```bash
pip install -e third_party/graphiti
```

To update, re-clone upstream at the desired tag, remove its `.git` and
`tests/evals/`, and replace this directory; then bump the version/commit above.
