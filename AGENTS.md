# Contributor Guide

This file is the main coding guide for this repository.

## Workflow

- Read the related code, tests, config, and docs before editing.
- Review the existing architecture and coding style before implementing changes.
- Keep each change small and focused.
- Do not refactor or reformat unrelated code; leave existing code unchanged unless modification is required.
- State important assumptions and tradeoffs.
- Add or update tests for changed behavior.
- Run the relevant checks before handoff.

## Python

- Use Ruff for formatting and linting.
- Use type hints at public APIs and system boundaries.
- Keep docstrings and comments concise and on one line.
- Use `pathlib` for filesystem paths.
- Use context managers for managed resources.
- Use the project logging style. Do not use `print()` for diagnostics.
- Keep functions small and easy to understand.

## Design

- Keep code clean, minimal, and production-ready.
- Prefer simple code over flexible code that is not needed yet.
- Prefer explicit data flow over hidden global state.
- Prefer direct code over helper functions. Add a helper only when it is reused or encapsulates a cohesive, non-trivial operation.
- Rely on native exceptions for configuration and trusted internal data. Do not add custom validation or error wrappers for values produced by the same code path.
- Do not add dependencies unless the standard library or current dependencies are not enough.
- Do not break public APIs, file formats, or persisted schemas without a migration plan.
- Do not add fallback paths, caches, retries, feature flags, or compatibility shims unless the task needs them.

## Data And Security

- Do not add custom validation unless explicitly requested; rely on typed frameworks and native exceptions.
- Do not catch errors with `try/except`; let exceptions propagate naturally.
- Keep secrets out of code, docs, tests, logs, and committed config.
- Do not log credentials, personal data, or full document contents.
- Do not build shell commands or SQL queries with string interpolation.
- Use explicit timeouts and bounded concurrency for external services.
- Parse model and tool output with typed interfaces and let invalid data raise naturally.

## Heuristics

- Do not use regex, keyword matching, handcrafted scores, or rule engines for implementations.
- Do not hardcode changeable values; put them in typed config, environment variables, or versioned data files.

## Checks

Run these before committing when possible:

```bash
ruff format --check .
ruff check .
pytest
```

Do not weaken tests or checks only to make them pass.
