"""Helpers for ChatGPT Codex Responses API calls used by LongMemEval."""

from __future__ import annotations

import json

import httpx

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def parse_sse_response_text(text: str) -> dict:
    """Parse SSE text and return the final `response` object."""
    last_response: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload_raw = line[6:]
        if not payload_raw or payload_raw == "[DONE]":
            continue
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type", "")
        if event_type in {"response.created", "response.in_progress"}:
            resp = payload.get("response")
            if isinstance(resp, dict):
                last_response = resp
        elif event_type in {"response.completed", "response.failed"}:
            resp = payload.get("response")
            if isinstance(resp, dict):
                return resp
        elif event_type == "error":
            return {"error": payload.get("error", payload)}
    return last_response or {}


def codex_simple_text_completion(
    *,
    token: str,
    model: str,
    prompt: str,
    instructions: str = "You are a helpful assistant.",
    max_output_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_s: float = 300.0,
) -> dict:
    """Call ChatGPT Codex responses endpoint and return parsed response object."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "instructions": instructions,
        "store": False,
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(CODEX_RESPONSES_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI Codex API error {resp.status_code}: {resp.text[:500]}")
    return parse_sse_response_text(resp.text)


def extract_output_text(response: dict) -> str:
    """Extract assistant text from a parsed codex `response` object."""
    text_parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                text_parts.append(part.get("text", ""))
    return "".join(text_parts)


def extract_usage(response: dict) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from codex response usage."""
    usage = response.get("usage", {})
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
