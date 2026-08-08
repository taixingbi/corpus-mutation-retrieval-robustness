from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from facm.clients.http_retry import request_with_retry


@dataclass
class ChatResult:
    """Normalized chat completion result for OpenAI-compatible (incl. reasoning) APIs."""

    content: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_message: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    model: str = ""

    @property
    def text(self) -> str:
        return self.content


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # OpenAI-style content parts; skip reasoning-only parts when typed
                typ = str(item.get("type") or "")
                if typ in {"reasoning", "thinking"}:
                    continue
                if "text" in item and item["text"] is not None:
                    parts.append(str(item["text"]))
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "".join(parts)
    return str(value)


def extract_message_content(message: dict[str, Any]) -> str:
    """Prefer final answer fields; do not fall back to long reasoning traces."""
    for key in ("content", "output_text", "answer", "final_answer"):
        text = _as_text(message.get(key)).strip()
        if text:
            return text
    # Some gateways nest the answer
    for key in ("response", "output"):
        nested = message.get(key)
        if isinstance(nested, dict):
            text = extract_message_content(nested).strip()
            if text:
                return text
        text = _as_text(nested).strip()
        if text:
            return text
    return ""


def is_truncated_completion(
    *,
    finish_reason: str | None,
    usage: dict[str, Any],
    max_tokens: int,
    content: str,
) -> bool:
    """Detect length truncation (reasoning models often burn budget before JSON ends)."""
    fr = (finish_reason or "").lower()
    if fr in {"length", "max_tokens", "max_length"}:
        return True
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    if isinstance(completion_tokens, int) and max_tokens > 0 and completion_tokens >= max_tokens:
        return True
    # Incomplete JSON / cut mid-object — finish_reason may still be "stop" on some gateways
    s = content.strip()
    if s.startswith("{") and not s.endswith("}"):
        return True
    if s.startswith("[") and not s.endswith("]"):
        return True
    if s.count("{") > s.count("}"):
        return True
    return False


class ChatClient:
    """OpenAI-compatible chat client for Bedrock inference MVP."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
        *,
        retries: int = 5,
    ) -> None:
        load_dotenv()
        self.base_url = (base_url or os.getenv("FACM_INFERENCE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("FACM_INFERENCE_API_KEY") or "1234"
        self.model = model or os.getenv("FACM_INFERENCE_MODEL") or "openai.gpt-oss-120b-1:0"
        self.timeout = timeout
        self.retries = max(0, int(retries))
        if not self.base_url:
            raise RuntimeError(
                "FACM_INFERENCE_URL is not set. See example_inference.md / .env.example"
            )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> ChatResult:
        url = f"{self.base_url}/v1/chat/completions".replace("//v1/", "/v1/")
        used_model = model or self.model
        payload: dict[str, Any] = {
            "model": used_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        def _once() -> dict[str, Any]:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()

        data = request_with_retry(_once, retries=self.retries, label="chat")
        if data.get("error"):
            raise RuntimeError(f"chat error: {data.get('error')} {data.get('detail')}")
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        content = extract_message_content(message)
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        truncated = is_truncated_completion(
            finish_reason=finish_reason,
            usage=usage,
            max_tokens=max_tokens,
            content=content,
        )
        return ChatResult(
            content=content,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
            raw_message=message,
            truncated=truncated,
            model=used_model,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> str:
        """Backward-compatible helper — returns final answer text only."""
        return self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        ).content
