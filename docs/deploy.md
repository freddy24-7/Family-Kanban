# Deploying to Railway

The backend ships as a Docker image (`backend/Dockerfile`, config in `backend/railway.json`).
Migrations run automatically on every deploy before the API starts.

## One-time setup

1. **Project**: create a Railway project and connect this GitHub repo.
2. **Database**: + Create → Template → search **pgvector** (current setup: service
   `pgvector`, Postgres 18.6, pgvector 0.8.6). ⚠️ In this template `DATABASE_URL` is the
   **public** proxy URL and `DATABASE_URL_PRIVATE` the internal one.
3. **API service**: Settings → Source → **Root Directory = `/backend`** (without it Railpack
   fails with "could not determine how to build"). Optional Watch Paths: `/backend/**`.
   Railway then picks up `railway.json` (Dockerfile build, `/health` healthcheck).
4. **Variables** on the API service (paste raw values: no quotes, no trailing newline):

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | `${{pgvector.DATABASE_URL_PRIVATE}}` — internal network (the `postgres://` scheme is normalised in code) |
   | `APP_ENV` | `production` |
   | `AUTH_SECRET` | `${{ secret(64) }}` (Railway generates it) |
   | `FRONTEND_URL` / `CORS_ORIGINS` | the frontend URL (Phase 4) |
   | `RESEND_API_KEY`, `EMAIL_FROM`, `EMAIL_REPLY_TO` | real email (see below) |
   | `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | when Google login is enabled |

   ⚠️ Don't blindly accept Railway's "Suggested Variables": it proposed `${{ secret() }}` for
   `ACCESS_TOKEN_LIFETIME_SECONDS` (a number) and for the Google client secret. Only add
   variables you have real values for; blank/unset ones fall back to code defaults.

5. **Networking**: generate a public domain on port **8080**; check `https://<domain>/health` returns `{"status":"ok"}`.
6. **Admin**: register your account, then run against the **public** DB URL (the pgvector
   service's `DATABASE_URL`):
   `cd backend && DATABASE_URL=<public url> uv run python -m app.cli make-admin you@example.com`

## Frontend (PWA) on Railway

A second service in the same project, built from `frontend/Dockerfile` (Node build →
static files served by Caddy with an SPA fallback and cache headers).

1. **+ Create → GitHub repo** (same repo) → Settings → Source → **Root Directory `/frontend`**
   (optional Watch Paths: `/frontend/**`).
2. **Variables** on the frontend service:

   | Variable | Value |
   |---|---|
   | `VITE_API_URL` | `https://${{Family-Kanban.RAILWAY_PUBLIC_DOMAIN}}` (the API's public URL) |

   ⚠️ `VITE_*` values are baked in at **build** time (the Dockerfile passes it as a build
   arg and refuses to build without it). After changing it, redeploy; a restart is not enough.
3. **Networking → Generate Domain** (port from the deploy log, 8080 by default).
4. On the **API** service set `FRONTEND_URL` and `CORS_ORIGINS` to the frontend URL
   (`https://<frontend domain>`, no trailing slash): CORS for the browser, and the links in
   verification / reset / invite emails.

Cache policy (Caddyfile): `/assets/*` (hashed) cached for a year; everything else, including
`index.html` for every route and the service worker, `no-cache`, so phones pick up new
deploys.

## Email (Resend)

Configured 2026-09-30 for `scalarox.nl` (region eu-west-1):
- DNS: DKIM `resend._domainkey` TXT, `send` MX → `feedback-smtp.eu-west-1.amazonses.com`,
  `send` TXT SPF `include:amazonses.com`, `_dmarc` (p=reject). The domain's own MX
  (Hostnet mailboxes) is untouched: Resend uses the `send.` subdomain.
- API key: *Sending access* only, restricted to `scalarox.nl`.
- `EMAIL_FROM = Gezinsbord <gezinsbord@scalarox.nl>` (no mailbox needed on a verified domain),
  `EMAIL_REPLY_TO = info@scalarox.nl` so replies reach a real inbox.
- Check: `POST /auth/forgot-password` for your own address, then look for
  `POST https://api.resend.com/emails "200 OK"` in the API logs.

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
