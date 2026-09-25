"""Admin commands. Usage: uv run python -m app.cli make-admin you@example.com"""

import asyncio
import sys

from sqlalchemy import func, select

from app.db import SessionFactory
from app.models import User


async def make_admin(email: str) -> None:
    async with SessionFactory() as session:
        user = await session.scalar(select(User).where(func.lower(User.email) == email.lower()))
        if user is None:
            sys.exit(f"No user with email {email}; register first.")
        user.is_superuser = True
        await session.commit()
        print(f"{user.email} is now a platform admin.")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "make-admin":
        sys.exit(__doc__)
    asyncio.run(make_admin(sys.argv[2]))


if __name__ == "__main__":
    main()
