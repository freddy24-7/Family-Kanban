import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config, repository
from app.auth import include_auth_routers
from app.db import SessionFactory
from app.routers import demo, households, sprints, topics

# App loggers at INFO (uvicorn only configures its own). In development the
# mailer logs verification/reset/invite links here.
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail loud: the app is useless without its database.
    async with SessionFactory() as session:
        await repository.ensure_stub_model_versions(session)
    yield


app = FastAPI(title="Family Kanban API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.head("/health", include_in_schema=False)
def health() -> dict[str, str]:
    # Static on purpose: uptime monitors must not trigger DB or model work.
    return {"status": "ok"}


include_auth_routers(app)
app.include_router(households.router)
app.include_router(topics.router)
app.include_router(sprints.router)
app.include_router(demo.router)
