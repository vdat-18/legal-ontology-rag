FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Installed as its own layer so it is only invalidated by dependency changes,
# not by every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --all-extras

# baml_client/ is git-ignored and must be generated before the project itself
# can be installed, since legal_rag imports it.
COPY src/ src/
COPY baml_src/ baml_src/
COPY README.md ./
RUN uv run baml-cli generate --from baml_src --no-tests \
    && uv sync --frozen --all-extras

CMD ["bash"]
