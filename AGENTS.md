# Python Contributor Guide

This file defines the default working agreements for Python repositories. More specific repository or module instructions take precedence.

## Workflow

- Read the relevant code, tests, configuration, and documentation before editing.
- Identify the repository's Python version, package manager, architecture, and configured tools before implementing changes.
- State material assumptions and tradeoffs. Clarify ambiguity only when different interpretations would materially change the result.
- For non-trivial multi-step work, use a brief plan with a verification step for each stage.
- Keep changes small and surgical. Every changed line should trace directly to the requested outcome.
- Do not refactor, reformat, rename, or improve unrelated code; match the existing style.
- Remove only imports, variables, or functions made unused by your own change; mention unrelated dead code instead of deleting it.

## Python

- Follow the Python version and package manager configured by the repository.
- Use Ruff and pytest unless the repository explicitly defines alternatives.
- Use type hints at public APIs, system boundaries, and places where types are not obvious.
- Prefer `pathlib` for filesystem paths and context managers for managed resources.
- Use `loguru.logger` for logging and diagnostics; never use `print()`.
- Keep docstrings and comments concise and explain only what the code does not make clear.
- Do not edit generated code directly; use the repository's generation workflow.

## Design And Reliability

- Write the minimum code that solves the current problem. Do not add speculative features, abstractions, flexibility, or configurability.
- Prefer explicit data flow over hidden global state and side effects.
- Add a helper only when it is reused or encapsulates a cohesive, non-trivial operation.
- Let unexpected internal errors propagate; handle expected failures at system boundaries and catch only exceptions that can be handled meaningfully.
- Do not duplicate validation guaranteed by trusted internal data or typed frameworks. Validate untrusted input at system boundaries using the repository's existing approach.
- Do not add dependencies when the standard library or current dependencies are sufficient.
- Do not break public APIs, file formats, or persisted schemas without a migration plan.
- Do not add fallbacks, caches, retries, feature flags, or compatibility shims unless the task requires them.
- Keep changeable values in typed configuration, environment variables, or versioned data files.

## Data And Security

- Keep secrets, credentials, tokens, and personal data out of code, tests, documentation, logs, and committed configuration.
- Do not log credentials, personal data, or full sensitive document contents.
- Do not build shell commands or SQL queries by interpolating untrusted data; use structured APIs and parameterized queries.
- Use explicit timeouts and bounded concurrency for external services.
- Parse external, model, and tool output through typed interfaces or schemas and let invalid data fail clearly.
- Do not perform destructive operations until the exact scope and target are known.

## Verification And Handoff

- Add or update tests for changed behavior. Reproduce bugs with tests when practical and keep tests passing before and after refactors.
- Discover expected commands from the README, `pyproject.toml`, task runner, and CI configuration.
- Run the smallest relevant checks first, then broader formatting, linting, typing, build, and test checks when appropriate.
- Do not remove, skip, or weaken tests and checks merely to make them pass.
- Preserve the user's uncommitted and unrelated changes. Do not revert or overwrite work you did not create.
- Do not use destructive Git or filesystem commands, and do not amend commits, unless explicitly requested.
- Review the final diff, summarize what changed and why, list the checks run, and identify anything that could not be verified.
- Do not claim success without evidence from the relevant checks.
