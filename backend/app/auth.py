"""Authentication with fastapi-users: email/password (+ verification and reset)
and optional Google OAuth. The only module that configures fastapi-users, so the
library can be swapped later (it is in maintenance mode, see ADR-001)."""

import uuid

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, BearerTransport
from fastapi_users.authentication.strategy.db import AccessTokenDatabase, DatabaseStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from app import config, mailer, messages_nl
from app.db import get_session
from app.models import AccessToken, OAuthAccount, User

# --- Schemas ----------------------------------------------------------------


class UserRead(schemas.BaseUser[uuid.UUID]):
    display_name: str


class UserCreate(schemas.BaseUserCreate):
    display_name: str = Field(min_length=1, max_length=100)


class UserUpdate(schemas.BaseUserUpdate):
    display_name: str | None = Field(default=None, min_length=1, max_length=100)


# --- User manager -------------------------------------------------------------


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = config.AUTH_SECRET
    verification_token_secret = config.AUTH_SECRET

    async def validate_password(self, password: str, user) -> None:
        from fastapi_users.exceptions import InvalidPasswordException

        if len(password) < 10:
            raise InvalidPasswordException(reason="Password must be at least 10 characters")

    async def on_after_register(self, user: User, request: Request | None = None) -> None:
        if not user.is_verified:
            await self.request_verify(user, request)

    async def on_after_request_verify(
        self, user: User, token: str, request: Request | None = None
    ) -> None:
        subject, body = messages_nl.verify_email(
            f"{config.FRONTEND_URL}/verify-email?token={token}"
        )
        await mailer.send_email(user.email, subject, body)

    async def on_after_forgot_password(
        self, user: User, token: str, request: Request | None = None
    ) -> None:
        subject, body = messages_nl.reset_password(
            f"{config.FRONTEND_URL}/reset-password?token={token}"
        )
        await mailer.send_email(user.email, subject, body)


async def get_user_db(session: AsyncSession = Depends(get_session)):
    yield SQLAlchemyUserDatabase(session, User, OAuthAccount)


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


# --- Auth backend: bearer token stored in the database (revocable) -------------


async def get_access_token_db(session: AsyncSession = Depends(get_session)):
    yield SQLAlchemyAccessTokenDatabase(session, AccessToken)


def get_database_strategy(
    access_token_db: AccessTokenDatabase[AccessToken] = Depends(get_access_token_db),
) -> DatabaseStrategy:
    return DatabaseStrategy(access_token_db, lifetime_seconds=config.ACCESS_TOKEN_LIFETIME_SECONDS)


auth_backend = AuthenticationBackend(
    name="database",
    transport=BearerTransport(tokenUrl="auth/login"),
    get_strategy=get_database_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(active=True, superuser=True)


# --- Routers --------------------------------------------------------------------


def include_auth_routers(app) -> None:
    app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth", tags=["auth"])
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
    )
    app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
    app.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"]
    )

    # Google login: dormant until the OAuth client is configured.
    if config.GOOGLE_OAUTH_CLIENT_ID and config.GOOGLE_OAUTH_CLIENT_SECRET:
        google = GoogleOAuth2(config.GOOGLE_OAUTH_CLIENT_ID, config.GOOGLE_OAUTH_CLIENT_SECRET)
        redirect_url = f"{config.FRONTEND_URL}/auth/google/callback"
        # associate_by_email=False on purpose: auto-linking would let someone who
        # pre-registered your email with a password take over your Google login.
        # Existing users link Google explicitly via the associate router instead.
        app.include_router(
            fastapi_users.get_oauth_router(
                google,
                auth_backend,
                config.AUTH_SECRET,
                redirect_url=redirect_url,
                associate_by_email=False,
                is_verified_by_default=True,  # Google verifies email addresses
                csrf_token_cookie_secure=config.IS_PRODUCTION,
            ),
            prefix="/auth/google",
            tags=["auth"],
        )
        app.include_router(
            fastapi_users.get_oauth_associate_router(
                google,
                UserRead,
                config.AUTH_SECRET,
                redirect_url=f"{config.FRONTEND_URL}/auth/google/associate-callback",
                csrf_token_cookie_secure=config.IS_PRODUCTION,
            ),
            prefix="/auth/associate/google",
            tags=["auth"],
        )
