"""Full-context baseline: send entire haystack to Sonnet."""

from __future__ import annotations

import json as _json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx

from .auth import resolve_provider_token
from .cost import BudgetTracker
from .dataset import LongMemEvalQuestion
from .openai_codex import codex_simple_text_completion, extract_output_text, extract_usage

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


def _format_haystack(q: LongMemEvalQuestion) -> str:
    """Format haystack using LongMemEval's exact prompt template."""
    parts = [
        "I will give you several history chats between you and a user.",
        "Please answer the question based on the relevant chat history.",
        "",
        "History Chats:",
    ]

    for i, (session, date) in enumerate(zip(q.haystack_sessions, q.haystack_dates), 1):
        parts.append("")
        parts.append(f"### Session {i}:")
        parts.append(f"Session Date: {date}")
        parts.append("Session Content:")
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")

    parts.append("")
    parts.append(f"Current Date: {q.question_date}")
    parts.append(f"Question: {q.question}")
    parts.append("Answer:")

    return "\n".join(parts)


def _run_gemini_cli(formatted: str, model: str, question_id: str) -> dict:
    """Run baseline via gemini CLI (uses Google account OAuth / Ultra subscription)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(formatted)
        prompt_file = f.name

    try:
        result = subprocess.run(
            ["gemini", "-p", f"Answer the question at the end based on the chat history provided via stdin.",
             "--model", model, "--output-format", "json"],
            stdin=open(prompt_file),
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        import os
        os.unlink(prompt_file)

    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else ""
        raise RuntimeError(f"gemini CLI failed (rc={result.returncode}): {stderr}")

    # Parse JSON output - find the JSON object in stdout (skip any non-JSON preamble)
    stdout = result.stdout.strip()
    # Find the first '{' to handle "Loaded cached credentials." preamble
    json_start = stdout.find("{")
    if json_start < 0:
        raise RuntimeError(f"gemini CLI returned no JSON: {stdout[:300]}")

    data = _json.loads(stdout[json_start:])
    hypothesis = data.get("response", "")

    # Extract token counts from stats
    stats = data.get("stats", {})
    input_tokens = 0
    output_tokens = 0
    for model_stats in stats.get("models", {}).values():
        tokens = model_stats.get("tokens", {})
        input_tokens += tokens.get("input", 0)
        output_tokens += tokens.get("candidates", 0)

    return {
        "hypothesis": hypothesis,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# gemini-oauth: Direct HTTP calls via Gemini CLI's OAuth credentials
# Uses https://cloudcode-pa.googleapis.com (Code Assist API) with Bearer auth.
# ---------------------------------------------------------------------------

_OAUTH_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"
_OAUTH_CLIENT_ID = os.environ.get("GEMINI_OAUTH_CLIENT_ID", "")
_OAUTH_CLIENT_SECRET = os.environ.get("GEMINI_OAUTH_CLIENT_SECRET", "")
_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

# Module-level caches
_cached_token: str | None = None
_cached_token_expiry: float = 0.0
_cached_project_id: str | None = None


def _get_gemini_oauth_token() -> str:
    """Get a valid OAuth access token, refreshing if needed."""
    global _cached_token, _cached_token_expiry

    now = time.time() * 1000  # expiry_date is in ms

    if _cached_token and _cached_token_expiry > now + 60_000:
        return _cached_token

    creds = _json.loads(_OAUTH_CREDS_PATH.read_text())
    expiry = creds.get("expiry_date", 0)

    if expiry > now + 60_000:
        _cached_token = creds["access_token"]
        _cached_token_expiry = expiry
        return _cached_token

    # Token expired — refresh it
    logger.info("Gemini OAuth token expired, refreshing...")
    refresh_token = creds.get("refresh_token")
    if not refresh_token:
        raise ValueError("No refresh_token in ~/.gemini/oauth_creds.json")

    resp = httpx.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": _OAUTH_CLIENT_ID,
            "client_secret": _OAUTH_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OAuth token refresh failed: {resp.status_code} {resp.text[:300]}")

    token_data = resp.json()
    new_access_token = token_data["access_token"]
    expires_in = token_data.get("expires_in", 3600)
    new_expiry = time.time() * 1000 + expires_in * 1000

    # Update the creds file so gemini CLI stays in sync
    creds["access_token"] = new_access_token
    creds["expiry_date"] = new_expiry
    _OAUTH_CREDS_PATH.write_text(_json.dumps(creds, indent=2))

    _cached_token = new_access_token
    _cached_token_expiry = new_expiry
    logger.info("Gemini OAuth token refreshed, expires in %ds", expires_in)
    return _cached_token


def _get_gemini_project_id() -> str:
    """Get the Code Assist project ID (cached after first call)."""
    global _cached_project_id
    if _cached_project_id:
        return _cached_project_id

    token = _get_gemini_oauth_token()
    resp = httpx.post(
        f"{_CODE_ASSIST_ENDPOINT}:loadCodeAssist",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={},
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to load Code Assist project: {resp.status_code} {resp.text[:300]}")
    _cached_project_id = resp.json().get("cloudaicompanionProject", "")
    logger.info("Gemini OAuth project ID: %s", _cached_project_id)
    return _cached_project_id


def _run_gemini_oauth(formatted: str, model: str, question_id: str, max_output_tokens: int = 8192, thinking_budget: int = 2048) -> dict:
    """Run via Code Assist API using OAuth token (same path as gemini CLI)."""
    token = _get_gemini_oauth_token()
    project_id = _get_gemini_project_id()

    payload = {
        "model": model,
        "project": project_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": formatted}]}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": 0.0,
                "thinkingConfig": {"thinkingBudget": thinking_budget},
            },
        },
    }

    url = f"{_CODE_ASSIST_ENDPOINT}:generateContent"

    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        # Retry on 429/503
        for _retry in range(5):
            if resp.status_code not in (429, 503):
                break
            wait = min(10 * (2 ** _retry), 120)
            logger.warning("Gemini OAuth retry %d (HTTP %d), waiting %ds", _retry + 1, resp.status_code, wait)
            time.sleep(wait)
            # Refresh token on retry in case it expired
            token = _get_gemini_oauth_token()
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

    if resp.status_code != 200:
        logger.error("Gemini OAuth error %d: %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Gemini OAuth error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    # Response is wrapped: {response: {candidates: [...], usageMetadata: {...}}}
    inner = data.get("response", data)
    candidates = inner.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    hypothesis = "".join(p.get("text", "") for p in parts if "text" in p)
    usage = inner.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    return {
        "hypothesis": hypothesis,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _dump_baseline_payload(
    *,
    question_id: str,
    question_text: str,
    gold_answer: str,
    formatted_prompt: str,
    hypothesis: str,
    input_tokens: int,
    output_tokens: int,
    provider: str,
    model: str,
    elapsed_s: float,
    cache_dir: Path | None = None,
) -> None:
    """Save the full baseline payload for autopsy analysis."""
    if not cache_dir:
        cache_dir = Path(__file__).parent / "cache"
    out_dir = cache_dir / question_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"baseline_payload_log_{ts}.json"

    payload = {
        "question_id": question_id,
        "question": question_text,
        "gold_answer": gold_answer,
        "hypothesis": hypothesis,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "formatted_prompt": formatted_prompt,
    }
    try:
        out_path.write_text(_json.dumps(payload, indent=2, default=str))
        logger.info("Baseline [%s]: payload log saved to %s", question_id, out_path)
    except Exception as e:
        logger.warning("Baseline [%s]: failed to save payload log: %s", question_id, e)


def run_baseline(
    question: LongMemEvalQuestion,
    budget: BudgetTracker,
    model: str = "claude-sonnet-4-5-20250929",
    provider: str = "anthropic",
    api_key: str | None = None,
    auth_mode: str = "auto",
    cache_dir: Path | None = None,
    verbose_reasoning: bool = False,
) -> dict:
    """Run full-context baseline for a single question.

    Returns dict with: hypothesis, input_tokens, output_tokens, cost, elapsed_s.
    """
    if provider in {"gemini-cli", "gemini-oauth"}:
        formatted = _format_haystack(question)
        logger.info(
            "Baseline [%s]: sending ~%d chars (~%d tokens) via %s (%s)",
            question.question_id, len(formatted), len(formatted) // 4, provider, model,
        )
        t0 = time.time()
        if provider == "gemini-oauth":
            result_data = _run_gemini_oauth(formatted, model, question.question_id)
        else:
            result_data = _run_gemini_cli(formatted, model, question.question_id)
        elapsed = time.time() - t0
        entry = budget.record(
            label=f"baseline:{question.question_id}",
            model=model,
            input_tokens=result_data["input_tokens"],
            output_tokens=result_data["output_tokens"],
        )
        logger.info(
            "Baseline [%s]: %d in / %d out tokens, $%.4f, %.1fs",
            question.question_id, result_data["input_tokens"], result_data["output_tokens"],
            entry.cost_usd, elapsed,
        )
        result = {
            "hypothesis": result_data["hypothesis"],
            "input_tokens": result_data["input_tokens"],
            "output_tokens": result_data["output_tokens"],
            "cost": entry.cost_usd,
            "elapsed_s": round(elapsed, 1),
        }
        _dump_baseline_payload(
            question_id=question.question_id,
            question_text=question.question,
            gold_answer=question.answer,
            formatted_prompt=formatted,
            hypothesis=result_data["hypothesis"],
            input_tokens=result_data["input_tokens"],
            output_tokens=result_data["output_tokens"],
            provider=provider,
            model=model,
            elapsed_s=result["elapsed_s"],
            cache_dir=cache_dir,
        )
        return result

    token = resolve_provider_token(provider, explicit_token=api_key, auth_mode=auth_mode)
    if not token:
        if provider in {"openai", "openai-codex", "openai_codex"}:
            raise ValueError(
                "OpenAI auth not found. Set OPENAI_API_KEY or OAuth token "
                "(OPENAI_OAUTH_ACCESS_TOKEN / OPENAI_OAUTH_TOKEN / OPENAI_ACCESS_TOKEN)."
            )
        if provider == "anthropic":
            raise ValueError("ANTHROPIC_API_KEY not set")
        if provider == "gemini":
            raise ValueError("GEMINI_API_KEY not set")
        raise ValueError(f"Unsupported provider or missing auth: {provider}")

    formatted = _format_haystack(question)
    logger.info(
        "Baseline [%s]: sending ~%d chars (~%d tokens) to %s/%s",
        question.question_id, len(formatted), len(formatted) // 4, provider, model,
    )

    if provider == "anthropic":
        headers = {
            "x-api-key": token,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": formatted}],
        }
        if verbose_reasoning:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
            payload["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            payload["temperature"] = 1
            payload["max_tokens"] = 16000
        url = API_URL
    elif provider == "openai":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # Reasoning models (gpt-5*, o3*, o4*) only support temperature=1
        _is_reasoning = model.startswith(("gpt-5", "o3", "o4"))
        payload = {
            "model": model,
            "max_completion_tokens": 1024,
            "messages": [{"role": "user", "content": formatted}],
        }
        if not _is_reasoning:
            payload["temperature"] = 0.0
        url = "https://api.openai.com/v1/chat/completions"
    elif provider in {"openai-codex", "openai_codex"}:
        headers = {}
        payload = {}
        url = ""
    elif provider == "gemini":
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": formatted}]}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.0,
                "thinkingConfig": {"thinkingBudget": 2048},
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={token}"
    else:
        raise ValueError(f"Unsupported baseline provider: {provider}")

    t0 = time.time()
    if provider in {"openai-codex", "openai_codex"}:
        data = codex_simple_text_completion(
            token=token,
            model=model,
            prompt=formatted,
            max_output_tokens=1024,
            temperature=0.0,
            timeout_s=300.0,
        )
    else:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            # Retry on 429/503 with exponential backoff
            for _retry in range(5):
                if resp.status_code not in (429, 503):
                    break
                wait = min(10 * (2 ** _retry), 120)
                logger.warning("Baseline retry %d (HTTP %d), waiting %ds", _retry + 1, resp.status_code, wait)
                time.sleep(wait)
                resp = client.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            logger.error("Baseline API error %d: %s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"Baseline API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
    elapsed = time.time() - t0
    if provider == "anthropic":
        text_parts = [
            block["text"] for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        hypothesis = "\n".join(text_parts)
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        # Save full response with thinking blocks for analysis
        if verbose_reasoning:
            _bl_payload_path = (cache_dir or Path("benchmarks/longmemeval/cache")) / question.question_id / f"baseline_payload_{time.strftime('%Y%m%d_%H%M%S')}.json"
            _bl_payload_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            _bl_payload_path.write_text(_json.dumps(data, indent=2))
    elif provider == "openai":
        choices = data.get("choices", [])
        hypothesis = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    elif provider in {"openai-codex", "openai_codex"}:
        hypothesis = extract_output_text(data)
        input_tokens, output_tokens = extract_usage(data)
    else:  # gemini
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        hypothesis = "".join(p.get("text", "") for p in parts if "text" in p)
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

    entry = budget.record(
        label=f"baseline:{question.question_id}",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    logger.info(
        "Baseline [%s]: %d in / %d out tokens, $%.4f, %.1fs",
        question.question_id, input_tokens, output_tokens, entry.cost_usd, elapsed,
    )

    result = {
        "hypothesis": hypothesis,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": entry.cost_usd,
        "elapsed_s": round(elapsed, 1),
    }

    # Dump baseline payload log for autopsy
    _dump_baseline_payload(
        question_id=question.question_id,
        question_text=question.question,
        gold_answer=question.answer,
        formatted_prompt=formatted,
        hypothesis=hypothesis,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider=provider,
        model=model,
        elapsed_s=result["elapsed_s"],
        cache_dir=cache_dir,
    )

    return result
