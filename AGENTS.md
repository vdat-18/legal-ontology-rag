# Repository Instructions

These rules apply to every contributor and coding agent in this repository.

## Required Workflow

- Treat this file as the canonical coding standard for the repository.
- Inspect the relevant code, tests, configuration, and documentation before changing anything.
- Surface material assumptions, ambiguities, and tradeoffs instead of silently guessing.
- Define observable success criteria and verify them before declaring work complete.
- Make the smallest change that solves the requested problem; avoid unrelated refactors.

## Design and Maintainability

- Prefer separation of concerns, KISS, and YAGNI.
- Apply DRY only after real, stable duplication exists; do not create speculative abstractions.
- Keep modules, classes, and functions focused, cohesive, and easy to delete or replace.
- Prefer explicit data flow and dependency injection over mutable globals and hidden side effects.
- Match established project patterns unless a change is explicitly intended to replace them.
- Remove only unused code introduced by the current change. Report unrelated dead code instead of deleting it.
- Add a dependency only when the standard library and existing dependencies are insufficient; explain the tradeoff.
- Do not break public APIs, persisted schemas, or file formats without an explicit migration plan and compatibility tests.
- Do not add compatibility shims, fallback paths, caches, retries, or feature flags unless the task requires them.

## Python Standards

- Use Ruff for formatting and linting. Do not hand-format around Ruff.
- Add concise docstrings at the top of every Python module, class, function, and method.
- Describe contracts, intent, invariants, or non-obvious behavior; do not restate the implementation.
- Keep comments short and limited to reasoning that the code cannot express clearly.
- Use type hints for public APIs, data models, and system boundaries.
- Use `pathlib` for filesystem paths and context managers for managed resources.
- Use the repository's established logging abstraction. Prefer Loguru for new Python applications and never use `print()` for diagnostics.
- Use explicit user-output APIs for intentional CLI output; keep diagnostics in the logger.
- Log useful structured context without secrets, credentials, personal data, or full document contents.

## Hardcoding and Heuristics

- Do not use regular expressions, keyword matching, handcrafted scores, or rule-based decision engines for business decisions or semantic interpretation.
- Use regex only for bounded lexical or syntax validation when a simpler parser or string API is insufficient; keep it small, documented, and tested.
- Do not hardcode configuration, paths, URLs, model names, prompts, schemas, mappings, thresholds, identifiers, credentials, or business values in source code.
- Put changeable values in typed configuration, environment variables, or versioned data files.
- Use named constants only for genuine language, protocol, or mathematical invariants.
- Prefer proper parsers, typed models, schema validation, and data-driven components over pattern matching or nested conditional rules.
- Treat external, tool-generated, and model-generated output as untrusted. Validate it against an explicit schema before use.

## Security and Reliability

- Validate untrusted input at system boundaries and reject invalid data with specific errors.
- Keep secrets out of source code, logs, fixtures, notebooks, and committed configuration.
- Do not use exceptions for normal control flow.
- Catch only specific exceptions at true system boundaries. Keep `try` blocks minimal, preserve exception causes, and never swallow failures.
- Fail fast with actionable errors. Never return fake success or silently fall back to degraded behavior.
- Use context managers for deterministic resource cleanup.
- Avoid unsafe deserialization, shell injection, unchecked path traversal, and unbounded network or file operations.
- Never construct shell commands or SQL queries through string interpolation; use structured APIs and parameterized queries.
- Do not perform blocking I/O in async code. Bound concurrency and support cancellation and explicit timeouts.
- Use least privilege, bounded retries with backoff, explicit timeouts, and deterministic cleanup for external resources.
- Pin direct and automation dependencies, commit the lockfile, remove unused dependencies, and audit known vulnerabilities before release.
- Pin third-party CI actions to reviewed full commit SHAs.

## Tests and Verification

- Test externally observable behavior rather than implementation details.
- For a bug fix, add a regression test that fails before the fix and passes after it.
- Keep tests deterministic and isolated from live network services by default.
- Inject clocks and random generators when behavior depends on time or randomness; use fixed seeds in tests.
- Cover failure paths, malformed external data, and compatibility-sensitive behavior when relevant.
- Run `ruff format --check .`, `ruff check .`, and the relevant test suite before handoff.
- Run `pre-commit run --all-files` before committing when pre-commit is installed.
- Do not weaken tests, lint rules, or security checks merely to make a build pass.
