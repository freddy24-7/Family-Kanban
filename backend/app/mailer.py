"""The only module that sends email (Resend HTTP API).

Dormant until RESEND_API_KEY is set: in development messages are logged instead,
so verification/reset/invite links can be copied from the server log.
Fails soft: a delivery problem is logged, never raised into the request.
"""

import logging

import httpx

from app import config

log = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


def _payload(to: str, subject: str, body_text: str) -> dict:
    payload = {"from": config.EMAIL_FROM, "to": [to], "subject": subject, "text": body_text}
    if config.EMAIL_REPLY_TO:
        payload["reply_to"] = config.EMAIL_REPLY_TO
    return payload


async def send_email(to: str, subject: str, body_text: str) -> bool:
    if not config.RESEND_API_KEY:
        log.info(
            "Email not sent (RESEND_API_KEY unset)\nTo: %s\nSubject: %s\n\n%s",
            to,
            subject,
            body_text,
        )
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                RESEND_URL,
                headers={"Authorization": f"Bearer {config.RESEND_API_KEY}"},
                json=_payload(to, subject, body_text),
            )
            response.raise_for_status()
        return True
    except httpx.HTTPError:
        log.exception("Email delivery failed (to=%s, subject=%s)", to, subject)
        return False
