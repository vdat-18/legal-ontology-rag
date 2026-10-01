"""Build-time checks for generated package sources."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_REQUIRED_BAML_CLIENT_FILES = (
    "__init__.py",
    "async_client.py",
    "inlinedbaml.py",
    "sync_client.py",
    "types.py",
)


class CustomBuildHook(BuildHookInterface):
    """Reject distributable artifacts that omit the generated BAML client."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        """Fail before packaging when generated runtime modules are unavailable."""
        del version, build_data
        client_dir = Path(self.root) / "src" / "legal_rag" / "baml_client"
        missing = [name for name in _REQUIRED_BAML_CLIENT_FILES if not (client_dir / name).is_file()]
        if missing:
            missing_files = ", ".join(missing)
            message = (
                "Generated BAML client is incomplete; missing "
                f"{missing_files}. Run `uv run baml-cli generate --from baml_src --no-tests` before building."
            )
            raise RuntimeError(message)
