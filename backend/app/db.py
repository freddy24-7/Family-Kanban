"""Database engine and session. Query code lives in repository.py."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app import config


def make_engine(pooled: bool = True) -> AsyncEngine:
    """pooled=True for the API (connections recycled before proxies drop them).
    pooled=False for long batch jobs (seed/simulate/retrain): a fresh connection
    per session, so nothing sits idle in a pool for minutes and goes stale."""
    if pooled:
        return create_async_engine(config.DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    return create_async_engine(config.DATABASE_URL, poolclass=NullPool)


engine = make_engine(pooled=config.DB_POOLED)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session
