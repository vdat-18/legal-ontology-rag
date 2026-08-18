"""Keep the generated BAML client in step with ``baml_src`` before tests import it.

The client is generated code and is not committed, so a clean checkout has none at all and a
checkout that pulls a schema change has a stale one. Either way the failure lands during test
collection as an ImportError for a type that does exist in ``baml_src`` — which reads like a
broken test rather than a missing build step. Regenerating here makes ``pytest`` work from a
clean checkout without a separate command to remember.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_BAML_SOURCE = _ROOT / "baml_src"
_GENERATED_CLIENT = _ROOT / "src" / "legal_rag" / "baml_client"


def _generator_command() -> list[str] | None:
    """Locate the BAML generator, preferring the one beside the running interpreter.

    ``baml-cli`` is a console script rather than an importable module, so it is resolved from the
    active environment's scripts directory first; falling back to ``PATH`` covers an interpreter
    invoked from outside its own virtualenv.
    """
    local = Path(sys.executable).parent / ("baml-cli.exe" if sys.platform == "win32" else "baml-cli")
    if local.exists():
        return [str(local)]
    found = shutil.which("baml-cli")
    return [found] if found else None


def _generated_is_current() -> bool:
    """Return whether the generated client covers every current schema file."""
    marker = _GENERATED_CLIENT / "types.py"
    if not marker.exists():
        return False
    generated_at = marker.stat().st_mtime
    return all(source.stat().st_mtime <= generated_at for source in _BAML_SOURCE.rglob("*.baml"))


def pytest_configure() -> None:
    """Regenerate the BAML client when it is missing or older than the schema."""
    if not _BAML_SOURCE.is_dir() or _generated_is_current():
        return
    command = _generator_command()
    if command is None:
        message = "baml-cli is not installed; run `uv sync` before the tests"
        raise RuntimeError(message)
    subprocess.run(  # noqa: S603
        [*command, "generate", "--from", str(_BAML_SOURCE), "--no-tests"], cwd=_ROOT, check=True
    )
