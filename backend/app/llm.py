"""The ONLY module that talks to Gemini.

- Dormant until GEMINI_API_KEY is set (`is_configured()`).
- Model cascade: falls through on 429/5xx/404 and on network errors (dropped
  connections, timeouts), so overload or a retired model degrades gracefully.
- Returns raw JSON items; callers validate each item themselves (one malformed
  item must never discard the whole response).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app import config

log = logging.getLogger(__name__)

CASCADE_STATUS_CODES = {404, 429, 500, 502, 503, 504}
REQUEST_TIMEOUT_MS = 120_000


class LLMNotConfigured(RuntimeError):
    pass


class LLMUnavailable(RuntimeError):
    """Every model in the cascade failed."""


@dataclass
class LLMResult:
    items: list[Any]
    model: str
    tokens_in: int
    tokens_out: int
    failed_models: list[str] = field(default_factory=list)


def is_configured() -> bool:
    return bool(config.GEMINI_API_KEY)


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if not is_configured():
        raise LLMNotConfigured("GEMINI_API_KEY is not set")
    if _client is None:
        _client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
    return _client


async def generate_json_list(
    system: str, prompt: str, item_schema: dict[str, Any], temperature: float = 1.0
) -> LLMResult:
    """Ask for a JSON object {"items": [...]} whose items follow `item_schema`
    (an OpenAPI-style dict). Items are returned unvalidated."""
    client = _get_client()
    response_schema = {
        "type": "OBJECT",
        "properties": {"items": {"type": "ARRAY", "items": item_schema}},
        "required": ["items"],
    }
    generation_config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=temperature,
        # Writing short tickets needs no reasoning; thinking tokens were ~90% of cost.
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    failed: list[str] = []
    for model in config.GEMINI_MODELS:
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=prompt, config=generation_config
            )
        except errors.APIError as exc:
            if exc.code in CASCADE_STATUS_CODES:
                log.warning("Gemini model %s failed (%s); trying next", model, exc.code)
                failed.append(f"{model}:{exc.code}")
                continue
            raise
        except httpx.TransportError as exc:  # dropped connection, timeout, DNS, ...
            log.warning("Gemini model %s network error (%s); trying next", model, exc)
            failed.append(f"{model}:{type(exc).__name__}")
            continue
        usage = response.usage_metadata
        return LLMResult(
            items=_parse_items(response.text or ""),
            model=model,
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
            if usage
            else 0,
            failed_models=failed,
        )
    raise LLMUnavailable(f"All Gemini models failed: {', '.join(failed)}")


def _parse_items(text: str) -> list[Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Gemini returned invalid JSON (%d chars)", len(text))
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []
