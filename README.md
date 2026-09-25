# Family Kanban

A household kanban PWA with a machine-learning classifier that is versioned,
monitored for drift, and retrained from real and simulated family data.

- Product spec: [docs/PRD.md](docs/PRD.md)
- Build plan: [docs/PLAN.md](docs/PLAN.md)
- Architecture decisions: [docs/decisions/](docs/decisions/)
- ML learning notes: [docs/learning/](docs/learning/)

## Local development

```bash
cp .env.example .env
docker compose up -d                                   # Postgres + pgvector
cd backend && uv sync && uv run uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
```
