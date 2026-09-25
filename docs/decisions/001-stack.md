# ADR-001: Technology stack

**Status**: accepted · 2026-09-25

## Context
Solo, part-time portfolio project. Needs: installable mobile PWA, Python ML
(scikit-learn), multi-tenant data, email/password + Google login, cheap hosting.

## Decision
- **Backend**: FastAPI + SQLAlchemy 2 + Alembic, Python 3.12, managed with uv.
- **Database**: Postgres 17 + pgvector (one database for relational data and embeddings).
- **Frontend**: Vite + React + TypeScript PWA. Next.js was considered and rejected:
  there is no server-rendering need because FastAPI is the backend, and Vite keeps the
  PWA simpler.
- **Auth**: self-hosted `fastapi-users` (email/password, verification, reset) with
  Google OAuth via `httpx-oauth`; bearer transport + database token strategy so
  sessions are revocable and long-lived enough for a PWA.
- **Email**: Resend, isolated in one module and inert without an API key.
- **LLM**: Gemini (cheap, simple structured output), only for demo generation.
- **Hosting**: Railway (API, Postgres, volume for model artifacts, cron later).

## Consequences
- `fastapi-users` is in **maintenance mode** (security fixes only; last release
  v15.0.5, March 2026). Acceptable: the feature set we need is complete and stable.
  Auth is isolated in one module so it can be replaced if the successor library ships.
- Self-hosting auth means we own email delivery, token storage and password hashing.
- One Postgres for everything keeps ops minimal; pgvector is enough at household scale.
