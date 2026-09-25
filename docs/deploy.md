# Deploying to Railway

The backend ships as a Docker image (`backend/Dockerfile`, config in `backend/railway.json`).
Migrations run automatically on every deploy before the API starts.

## One-time setup

1. **Project**: create a Railway project and connect this GitHub repo.
2. **Database**: add Postgres. We need the `vector` extension from Phase 8 onward:
   use Railway's **pgvector** Postgres template, or confirm on the stock one that
   `CREATE EXTENSION vector;` works (via `psql` against `DATABASE_PUBLIC_URL`).
3. **API service**: from the repo, set **Root Directory = `backend`**. Railway picks up
   `railway.json` (Dockerfile build, `/health` healthcheck).
4. **Variables** on the API service (paste raw values: no quotes, no trailing newline):

   | Variable | Value |
   |---|---|
   | `APP_ENV` | `production` |
   | `DATABASE_URL` | reference the Postgres service's `DATABASE_URL` (the `postgresql://` scheme is normalised in code) |
   | `AUTH_SECRET` | `python3 -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `FRONTEND_URL` / `CORS_ORIGINS` | the frontend URL (Phase 4) |
   | `RESEND_API_KEY`, `EMAIL_FROM` | when real emails should go out |
   | `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | when Google login is enabled |

5. **Networking**: generate a public domain; check `https://<domain>/health` returns `{"status":"ok"}`.
6. **Admin**: register your account, then run against the public DB URL:
   `cd backend && DATABASE_URL=<DATABASE_PUBLIC_URL> uv run python -m app.cli make-admin you@example.com`

## Google login (when enabling it)

- Google Cloud project → OAuth consent screen (External) → **Web application** client.
- **Enable the People API** in the same project, or the callback fails after a
  successful Google sign-in.
- Authorised redirect URIs (byte-exact, no trailing slash):
  `<FRONTEND_URL>/auth/google/callback` and `<FRONTEND_URL>/auth/google/associate-callback`.
- Publish the consent screen to Production before family members use it.

## Verify locally first

```bash
cd backend && docker build -t famkanban-api .
docker run --rm -p 8080:8080 -e PORT=8080 -e APP_ENV=production -e AUTH_SECRET=<48+ chars> \
  -e DATABASE_URL=postgresql://famkanban:famkanban@host.docker.internal:5433/famkanban famkanban-api
```
