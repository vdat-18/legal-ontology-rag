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

## Design review

The UI has not been reviewed with Vercel's `web-design-guidelines` / `react-best-practices`
skills, which need Node.js to install:

```bash
npx skills add vercel-labs/agent-skills -a claude-code --skill web-design-guidelines
npx skills add vercel-labs/agent-skills -a claude-code --skill react-best-practices
```

Applied by hand in the meantime: accessible name on the composer input, `aria-live` on the
transcript so answers are announced, `role="status"`/`role="alert"` on pending and error states,
`aria-expanded` on the reasoning toggle, a visible `:focus-visible` ring, and a
`prefers-reduced-motion` guard.
