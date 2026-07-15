# legal-ontology-rag

See [AGENTS.md](AGENTS.md) for the repository coding standards.

## Development

Install the local checks once:

```bash
python -m pip install pre-commit ruff==0.15.21 pytest
pre-commit install
```

Run quality checks before committing:

```bash
ruff format .
ruff check .
pytest
pre-commit run --all-files
```

Keep changes focused, include tests for changed behavior, and explain any
intentional exception to `AGENTS.md` or Ruff. Never commit secrets, unintended
generated artifacts, or local environment files.
