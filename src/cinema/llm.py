"""Ollama structured-JSON calls. Must be used inside ResourceManager.ollama_session()."""
from __future__ import annotations

import json
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return text[start:]


def chat_json(cfg: Config, model: str, system: str, user: str, schema: type[T], *, session: dict | None = None,
              think: bool | str | None = None, temperature: float = 0.7, num_ctx: int = 32768,
              num_predict: int = 16000, retries: int = 2, seed: int | None = None,
              timeout: float = 3600) -> T:
    """Call Ollama with a JSON schema; re-validate with Pydantic; feed validation errors back on retry.

    keep_alive stays positive within a session so a batch reuses the loaded model; the session
    context manager performs the verified keep_alive=0 unload.
    """
    if think is None:
        think = cfg.models.writer_think if model == cfg.models.writer else False
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_error = ""
    cloud = cfg.is_cloud(model)
    if cloud:
        # Measured: Ollama Cloud ignores `format`, so the schema goes in the prompt and Pydantic enforces it.
        messages[0]["content"] += ("\n\nOutput ONLY one JSON object (no prose, no code fences) that validates "
                                   "against this JSON Schema:\n" + json.dumps(schema.model_json_schema()))
    for attempt in range(retries + 1):
        body = {
            "model": model, "messages": messages, "stream": False, "format": schema.model_json_schema(),
            "keep_alive": "30m", "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict,
                        **({"seed": seed + attempt} if seed is not None else {})},
        }
        try:
            if cloud:
                from .cloud import CloudError, chat as cloud_chat
                try:
                    data = cloud_chat(cfg, body, timeout=timeout)
                except CloudError as exc:
                    raise LLMError(str(exc)) from exc
            else:
                r = httpx.post(f"{cfg.ollama_url}/api/chat", json=body,
                               timeout=httpx.Timeout(timeout, connect=30))
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as exc:
            last_error = f"HTTP error: {exc}"
            continue
        if session is not None:
            session["calls"] = session.get("calls", 0) + 1
            session.setdefault("eval_tokens", 0)
            session["eval_tokens"] += data.get("eval_count", 0) or 0
        content = data.get("message", {}).get("content", "")
        try:
            return schema.model_validate_json(_extract_json(content))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)[:1500]
            messages = [messages[0], messages[1]] + [
                {"role": "assistant", "content": content[:6000]},
                {"role": "user", "content": "Your JSON failed validation:\n" + last_error +
                 "\nReturn the complete corrected JSON object only."},
            ]
    raise LLMError(f"{model} failed to produce valid {schema.__name__}: {last_error}")
