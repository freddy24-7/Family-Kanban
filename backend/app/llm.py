"""The ONLY module that talks to Gemini.

- Dormant until GEMINI_API_KEY is set (`is_configured()`).
- Model cascade: falls through on 429/5xx/404 and on network errors (dropped
  connections, timeouts), so overload or a retired model degrades gracefully.
- Returns raw JSON items; callers validate each item themselves (one malformed
  item must never discard the whole response).
"""

import asyncio
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
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
ATTEMPTS_PER_MODEL = 3
RETRY_BASE_SECONDS = 2.0  # 2s, 4s between attempts
REQUEST_TIMEOUT_MS = 120_000
# Cheapest first. Models that reject a level are remembered for this process.
THINKING_LEVELS = ["minimal", "low"]
_thinking_level_for: dict[str, str] = {}


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
    (an OpenAPI-style dict). Items are returned unvalidated.

    Per model: brief retries with backoff on transient errors (429/5xx/network),
    then fall through to the next model."""
    client = _get_client()
    response_schema = {
        "type": "OBJECT",
        "properties": {"items": {"type": "ARRAY", "items": item_schema}},
        "required": ["items"],
    }
    failed: list[str] = []
    for model in config.GEMINI_MODELS:
        for attempt in range(ATTEMPTS_PER_MODEL):
            level = _thinking_level_for.get(model, THINKING_LEVELS[0])
            generation_config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
                # Writing short tickets needs no reasoning; thinking tokens were ~90%
                # of the cost. Not every model supports the lowest level.
                thinking_config=types.ThinkingConfig(thinking_level=level),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=prompt, config=generation_config
                )
            except errors.APIError as exc:
                if _unsupported_thinking_level(exc) and level != THINKING_LEVELS[-1]:
                    next_level = THINKING_LEVELS[THINKING_LEVELS.index(level) + 1]
                    log.info("Model %s rejects thinking=%s; using %s", model, level, next_level)
                    _thinking_level_for[model] = next_level
                    continue
                if exc.code in RETRYABLE_STATUS_CODES and attempt < ATTEMPTS_PER_MODEL - 1:
                    await asyncio.sleep(RETRY_BASE_SECONDS * 2**attempt)
                    continue
                if exc.code in CASCADE_STATUS_CODES:
                    log.warning("Gemini model %s failed (%s); trying next", model, exc.code)
                    failed.append(f"{model}:{exc.code}")
                    break
                raise
            except httpx.TransportError as exc:  # dropped connection, timeout, DNS, ...
                if attempt < ATTEMPTS_PER_MODEL - 1:
                    await asyncio.sleep(RETRY_BASE_SECONDS * 2**attempt)
                    continue
                log.warning("Gemini model %s network error (%s); trying next", model, exc)
                failed.append(f"{model}:{type(exc).__name__}")
                break
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


def _unsupported_thinking_level(exc: errors.APIError) -> bool:
    return exc.code == 400 and "thinking level" in (exc.message or "").lower()


def _parse_items(text: str) -> list[Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Gemini returned invalid JSON (%d chars)", len(text))
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []
