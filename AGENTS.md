# Contributor Guide

This file is the main coding guide for this repository.

## Workflow

- Read the related code, tests, config, and docs before editing.
- Keep each change small and focused.
- State important assumptions and tradeoffs.
- Add or update tests for changed behavior.
- Run the relevant checks before handoff.

## Python

- Use Ruff for formatting and linting.
- Use type hints at public APIs and system boundaries.
- Add short docstrings for modules, classes, functions, and methods.
- Use `pathlib` for filesystem paths.
- Use context managers for managed resources.
- Use the project logging style. Do not use `print()` for diagnostics.
- Keep functions small and easy to understand.

## Design

- Prefer simple code over flexible code that is not needed yet.
- Prefer explicit data flow over hidden global state.
- Do not add dependencies unless the standard library or current dependencies are not enough.
- Do not break public APIs, file formats, or persisted schemas without a migration plan.
- Do not add fallback paths, caches, retries, feature flags, or compatibility shims unless the task needs them.

## Data And Security

- Validate untrusted input at system boundaries.
- Keep secrets out of code, docs, tests, logs, and committed config.
- Do not log credentials, personal data, or full document contents.
- Do not build shell commands or SQL queries with string interpolation.
- Use explicit timeouts and bounded concurrency for external services.
- Treat model and tool output as untrusted. Validate it before use.

## Heuristics

- Do not use keyword matching, handcrafted scores, or rule engines for semantic decisions.
- Use regex only for small lexical or syntax parsing tasks.
- Put changeable values in typed config, environment variables, or versioned data files.

## Checks

Run these before committing when possible:

```bash
ruff format --check .
ruff check .
pytest
pre-commit run --all-files
```

Do not weaken tests or checks only to make them pass.
