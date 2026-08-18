# Chat frontend

Next.js chat UI for the Graphiti knowledge-graph API, showing each answer together with the
pipeline's retrieval trace: the query the graph was searched with, the facts returned, and the
source chunk behind every fact.

## Run

Requires Node.js 18.18+ (not currently installed on this machine).

```bash
# 1. start the API (separate terminal, from the repository root)
uv run legal-graphiti-api --port 8100

# 2. start the UI
cd frontend
cp .env.local.example .env.local     # only if the API is not on 127.0.0.1:8100
npm install
npm run dev                          # http://localhost:3000
```

The API allows the `localhost:3000` origin by default; serving the UI from another origin needs
that origin passed to `create_app(cors_origins=...)`.
