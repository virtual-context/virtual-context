#!/usr/bin/env python3
"""Model Evaluation Battery for VC tagger/compactor pipeline.

Tests 7 quality dimensions using real payloads from LocOMo Conv-48
and MRCR 4n_0543. ~30 seconds per model via OpenRouter.

Usage:
    python scripts/model_eval_battery.py --model google/gemini-2.0-flash-001
    python scripts/model_eval_battery.py --model thudm/glm-4-32b
    python scripts/model_eval_battery.py --model all   # run all known models
    OPENROUTER_API_KEY=sk-... python scripts/model_eval_battery.py --model ...

Tests:
    1. Contamination  — 480 MRCR tags + play scene → should NOT invent "lost-keys"
    2. Tag Reuse      — 4 sequential gaming→yoga turns → reuse tags, not synonyms
    3. Verb Accuracy  — "were given Battlefield 1" → verb != "mentioned"
    4. Pronoun Resol. — "We are planning..." + prev context → who contains "partner"
    5. Implied Facts  — Karlie "last photo, last one" → death implied
    6. Noise          — cooking segment → 0 junk facts (asked, greeted, said)
    7. JSON           — all outputs parse without stripping fences
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
LOCOMO_PATH = Path.home() / "projects" / "locomo" / "data" / "locomo10.json"
MRCR_PAYLOAD = (
    REPO / "benchmarks" / "mrcr" / "cache" / "4n_0543"
    / "payload_log_20260315_151159_842973.json"
)
MRCR_PAPER_PAYLOAD = (
    REPO / "extraneous" / "paper" / "data" / "mrcr" / "4n_0543"
)

# ---------------------------------------------------------------------------
# Prompts — imported from the actual codebase
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO))
from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_DETAILED  # noqa: E402
from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT  # noqa: E402

TAGGER_SYSTEM = TAG_GENERATOR_PROMPT_DETAILED.format(min_tags=2, max_tags=5)
COMPACTOR_SYSTEM = DEFAULT_SUMMARY_PROMPT  # Will fill {tags}, {conversation_text}, etc. per test

# ---------------------------------------------------------------------------
# Known models (OpenRouter IDs)
# ---------------------------------------------------------------------------
KNOWN_MODELS = [
    "google/gemini-2.0-flash-001",
    "thudm/glm-4-32b",
    "qwen/qwen-2.5-72b-instruct",
    "cohere/command-a-03-2025",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3-235b-a22b",
    "mistralai/mistral-large-2411",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_locomo_conv48() -> dict:
    """Load LocOMo Conv-48 (Deborah ↔ Jolene)."""
    with open(LOCOMO_PATH) as f:
        data = json.load(f)
    return data[7]  # conv-48


def load_mrcr_tags() -> list[str]:
    """Extract the 480-tag list from the MRCR enriched payload."""
    payload_file = MRCR_PAYLOAD
    if not payload_file.exists():
        # Try paper data folder
        candidates = list(MRCR_PAPER_PAYLOAD.glob("*payload_log*.json"))
        if not candidates:
            # Fallback: try extraneous paper data
            candidates = list(
                (REPO / "extraneous" / "paper" / "data" / "mrcr" / "4n_0543").glob("*vc*.json")
            )
        if candidates:
            payload_file = candidates[0]
        else:
            print("WARNING: No MRCR payload found. Contamination test will use empty tag list.")
            return []

    with open(payload_file) as f:
        data = json.load(f)

    # If this is a payload log with http_conversation
    if "http_conversation" in data:
        system = data["http_conversation"][0]["body"]["system"]
        m = re.search(r"\[all (\d+) topics\]\s*(.+?)(?:\n\n|\n\[)", system, re.DOTALL)
        if m:
            return [t.strip() for t in m.group(2).split(",")]

    # If it's the enriched VC payload directly (gemini/opus/sonnet vc json)
    if isinstance(data, dict) and "system" in data:
        system = data["system"]
        if isinstance(system, list):
            system = " ".join(s.get("text", "") if isinstance(s, dict) else str(s) for s in system)
        m = re.search(r"\[all (\d+) topics\]\s*(.+?)(?:\n\n|\n\[)", system, re.DOTALL)
        if m:
            return [t.strip() for t in m.group(2).split(",")]

    print("WARNING: Could not extract tag list from MRCR payload.")
    return []


def format_session(conv: dict, session_key: str) -> str:
    """Format a LocOMo session as conversation text."""
    messages = conv["conversation"][session_key]
    lines = []
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def format_messages(messages: list[dict]) -> str:
    """Format a list of message dicts as conversation text."""
    lines = []
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call via OpenRouter
# ---------------------------------------------------------------------------
def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float = 0.0,
) -> tuple[str, float]:
    """Call OpenRouter API, return (response_text, elapsed_seconds)."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 8192,
        "messages": messages,
    }
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=body, timeout=180)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content, elapsed


def parse_json_response(raw: str) -> tuple[dict | None, bool]:
    """Parse JSON from LLM response. Returns (parsed, clean_json).
    clean_json=True means no fence-stripping was needed."""
    # Try raw first
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        return None, False


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
class TestResult:
    def __init__(self, name: str, passed: bool, detail: str, elapsed: float, raw: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.elapsed = elapsed
        self.raw = raw

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.elapsed:.1f}s): {self.detail}"


def test_contamination(model: str, api_key: str, conv48: dict, mrcr_tags: list[str]) -> TestResult:
    """Test 1: Does the model apply 'lost-keys' to a pants/couch scene?

    The MRCR dataset has play scenes about finding keys between couch cushions.
    With 480 tags including 'keys' and various play-related tags, a weak model
    might invent 'lost-keys' for an unrelated pants scene.
    """
    # Build tagger prompt with existing tags
    tag_line = f"Existing tags (reuse when the topic genuinely matches, but create new tags for new topics): {', '.join(mrcr_tags)}"

    # The pants scene — a new conversation about finding pants under couch cushions
    # This is deliberately similar to the keys-under-couch-cushions play scenes in MRCR
    pants_scene = (
        "User: Write me a short play scene about pants.\n"
        "Assistant: Here's a short play scene:\n\n"
        "THE MISSING PANTS\n\n"
        "SARAH: (frantically searching) Has anyone seen my pants? I swear I left them "
        "right here on the chair!\n\n"
        "TOM: (lifting couch cushions) Found them! They were wedged between the couch "
        "cushions this whole time.\n\n"
        "SARAH: (relieved) Oh thank goodness! I was about to go to work in my pajamas.\n\n"
        "TOM: Maybe we should organize this apartment. Things keep disappearing.\n\n"
        "SARAH: You're right. This morning I couldn't find my socks either."
    )

    user_prompt = f"{tag_line}\n\nTag this conversation (return 2-5 tags):\n\n{pants_scene}"

    raw, elapsed = call_llm(model, TAGGER_SYSTEM, user_prompt, api_key)
    parsed, clean = parse_json_response(raw)

    if parsed is None:
        return TestResult("contamination", False, "JSON parse failed", elapsed, raw)

    tags = [t.lower() for t in parsed.get("tags", [])]

    # FAIL if model invented 'lost-keys' by merging 'keys' + 'lost-items' concepts
    bad_tags = [t for t in tags if "lost-key" in t or t == "lost-keys"]
    if bad_tags:
        return TestResult("contamination", False, f"Invented {bad_tags}", elapsed, raw)

    # Also check for 'keys' tag — this scene is about PANTS, not keys
    if "keys" in tags:
        return TestResult("contamination", False, "Applied 'keys' tag to pants scene", elapsed, raw)

    return TestResult("contamination", True, f"tags={tags}", elapsed, raw)


def test_tag_reuse(model: str, api_key: str, conv48: dict) -> TestResult:
    """Test 2: Does the model reuse existing tags on same-topic turns?

    4 sequential turns from Conv-48 Session 2:
    - Turn 1: Gaming + Detroit (creates tags)
    - Turn 2: Learning to play (should reuse gaming tags)
    - Turn 3: Walking Dead planning (should reuse gaming tags)
    - Turn 4: Running group + yoga (should create NEW tags)
    """
    session2 = conv48["conversation"]["session_2"]
    session15 = conv48["conversation"]["session_15"]

    # Extract the 4 test turns
    turns = [
        # Turn 1: Gaming intro
        'Jolene: They are very unusual pets! Here\'s me and my partner gaming last week - it\'s so fun. We played the game "Detroit" on the console. We are both crazy about this activity!',
        # Turn 2: Learning to play
        "Deborah: Did your boyfriend teach you to play?\nJolene: Even as a child I learned to play on my own.",
        # Turn 3: Walking Dead
        'Deborah: Do you only play old games or try new ones?\nJolene: We are planning to play "Walking Dead" next Saturday.',
        # Turn 4: Fitness (from Session 15)
        "Deborah: Hey Jolene! I started a running group with Anna - it's awesome connecting with people who care about fitness!\nJolene: Cool, Deb! Glad you found some people to get fit with.",
    ]

    all_tags = []
    new_tags_per_turn = []
    total_elapsed = 0
    raw_outputs = []

    accumulated_tags = []
    for i, turn in enumerate(turns):
        if accumulated_tags:
            tag_line = f"Existing tags (reuse when the topic genuinely matches, but create new tags for new topics): {', '.join(accumulated_tags)}"
            user_prompt = f"{tag_line}\n\nTag this conversation (return 2-5 tags):\n\n{turn}"
        else:
            user_prompt = f"Tag this conversation (return 2-5 tags):\n\n{turn}"

        raw, elapsed = call_llm(model, TAGGER_SYSTEM, user_prompt, api_key)
        total_elapsed += elapsed
        raw_outputs.append(raw)

        parsed, _ = parse_json_response(raw)
        if parsed is None:
            return TestResult("tag_reuse", False, f"JSON parse failed on turn {i+1}", total_elapsed, "\n---\n".join(raw_outputs))

        tags = [t.lower() for t in parsed.get("tags", [])]
        new = [t for t in tags if t not in accumulated_tags]
        new_tags_per_turn.append(new)
        all_tags.append(tags)
        accumulated_tags = list(set(accumulated_tags + tags))

    # Check: turns 2 and 3 should have 0 new tags (reuse gaming tags)
    # Turn 4 should have new tags (fitness/running/yoga)
    new_on_gaming = len(new_tags_per_turn[1]) + len(new_tags_per_turn[2])
    new_on_yoga = len(new_tags_per_turn[3])

    detail = f"new_tags=[{len(new_tags_per_turn[0])},{len(new_tags_per_turn[1])},{len(new_tags_per_turn[2])},{len(new_tags_per_turn[3])}]"

    if new_on_gaming > 2:
        return TestResult("tag_reuse", False, f"Leaked {new_on_gaming} new tags on gaming turns. {detail}", total_elapsed, "\n---\n".join(raw_outputs))

    if new_on_yoga == 0:
        return TestResult("tag_reuse", False, f"No new tags on yoga turn. {detail}", total_elapsed, "\n---\n".join(raw_outputs))

    return TestResult("tag_reuse", True, detail, total_elapsed, "\n---\n".join(raw_outputs))


def test_verb_accuracy(model: str, api_key: str, conv48: dict) -> TestResult:
    """Test 3: Does the model extract 'were given' or downgrade to 'mentioned'?

    Session 20: Jolene says "We were given a new game for the console last week,
    it is Battlefield 1." The verb should be "were given" / "received", NOT
    "mentioned" / "discussed" / "talked about".

    Uses a targeted excerpt — just the Battlefield exchange, not the full session.
    """
    # Targeted excerpt — only the Battlefield-relevant lines
    segment = (
        "Jolene: Long time no talk! We were given a new game for the console last week, "
        "it is Battlefield 1. What's been up with you?\n"
        "Deborah: Hey Jolene! Good to hear from you. That's cool!"
    )

    compactor_prompt = DEFAULT_SUMMARY_PROMPT.format(
        session_date="2023-08-21",
        tags="gaming, battlefield, console",
        target_tokens=200,
        conversation_text=segment,
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("verb_accuracy", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])

    # Find the Battlefield fact
    bf_facts = [
        f for f in facts
        if "battlefield" in f.get("object", "").lower()
        or "battlefield" in f.get("what", "").lower()
        or "game" in f.get("object", "").lower()
    ]

    if not bf_facts:
        return TestResult("verb_accuracy", False, "No Battlefield fact extracted", elapsed, raw)

    good_verbs = {"were given", "was given", "received", "got", "were gifted"}
    bad_verbs = {"mentioned", "discussed", "talked about", "said", "told", "shared"}

    for fact in bf_facts:
        verb = fact.get("verb", "").lower().strip()
        if verb in bad_verbs or verb.startswith("mention"):
            return TestResult("verb_accuracy", False, f"Bad verb: '{verb}'", elapsed, raw)
        if verb in good_verbs:
            return TestResult("verb_accuracy", True, f"verb='{verb}'", elapsed, raw)

    # Verb exists but not in either list — borderline
    actual_verb = bf_facts[0].get("verb", "")
    return TestResult("verb_accuracy", True, f"verb='{actual_verb}' (acceptable)", elapsed, raw)


def test_pronoun_resolution(model: str, api_key: str, conv48: dict) -> TestResult:
    """Test 4: Does the model resolve 'We' to Jolene + partner?

    Preceding context from Session 2 establishes that Jolene games with her partner.
    The current segment just says 'We are planning to play Walking Dead next Saturday.'
    The model must resolve 'We' using the preceding context.
    The 'who' field should contain 'partner'.
    """
    # Minimal preceding context — establishes partner gaming relationship
    prev_context = (
        'Jolene: Here\'s me and my partner gaming last week. '
        'We played the game "Detroit" on the console.'
    )

    # Current segment — just the Walking Dead line
    segment = 'Jolene: We are planning to play "Walking Dead" next Saturday.'

    compactor_prompt = DEFAULT_SUMMARY_PROMPT.format(
        session_date="2023-01-27",
        tags="gaming, walking-dead, console",
        target_tokens=200,
        conversation_text=f"[Previous context]\n{prev_context}\n\n[Current segment]\n{segment}",
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("pronoun_resolution", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])

    # Find Walking Dead / planning fact
    wd_facts = [
        f for f in facts
        if "walking dead" in f.get("object", "").lower()
        or "walking dead" in f.get("what", "").lower()
        or "planning to play" in f.get("what", "").lower()
    ]

    if not wd_facts:
        return TestResult("pronoun_resolution", False, "No Walking Dead fact extracted", elapsed, raw)

    # Check 'who' field for partner resolution
    for fact in wd_facts:
        who = fact.get("who", "").lower()
        what = fact.get("what", "").lower()
        subject = fact.get("subject", "").lower()
        # Partner might appear in who, what, or subject
        if "partner" in who or "partner" in what or "partner" in subject:
            return TestResult("pronoun_resolution", True, f"who='{fact.get('who','')}'", elapsed, raw)

    # Check if any fact mentions partner anywhere
    for fact in wd_facts:
        all_text = json.dumps(fact).lower()
        if "partner" in all_text:
            return TestResult("pronoun_resolution", True, f"partner found in fact", elapsed, raw)

    who_vals = [f.get("who", "") for f in wd_facts]
    return TestResult("pronoun_resolution", False, f"No partner resolution. who={who_vals}", elapsed, raw)


def test_implied_facts(model: str, api_key: str, conv48: dict) -> TestResult:
    """Test 5: Does the model extract 'Karlie passed away' from indirect language?

    Uses ONLY the indirect segment — no explicit 'I lost a friend' statement.
    The model must infer death from 'last photo', 'our last one', 'Memories keep
    our loved ones close', and the condolence reply.
    """
    # Narrowed segment: only the indirect references, matching original battery spec
    segment = (
        "Deborah: Memories keep our loved ones close. This is the last photo with "
        "Karlie which was taken last summer when we hiked. It was our last one. "
        "We had such a great time! Every time I see it, I can't help but smile.\n"
        "Jolene: I'm sorry for your loss. It sounds like you had wonderful times together."
    )

    compactor_prompt = DEFAULT_SUMMARY_PROMPT.format(
        session_date="2023-02-22",
        tags="loss, friendship, karlie, garden, travel",
        target_tokens=300,
        conversation_text=segment,
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("implied_facts", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])

    # Check for death/passing fact about Karlie
    death_terms = ["passed away", "died", "death", "passing", "lost", "passed"]

    for fact in facts:
        fact_text = json.dumps(fact).lower()
        if "karlie" in fact_text:
            for term in death_terms:
                if term in fact_text:
                    return TestResult("implied_facts", True, f"Found: {fact.get('what', '')[:80]}", elapsed, raw)

    # Also check summary for the implied fact
    summary = parsed.get("summary", "").lower()
    if "karlie" in summary and any(t in summary for t in death_terms):
        return TestResult("implied_facts", True, "Implied in summary (not facts)", elapsed, raw)

    karlie_facts = [f for f in facts if "karlie" in json.dumps(f).lower()]
    if karlie_facts:
        return TestResult("implied_facts", False, f"Karlie mentioned but no death: {karlie_facts[0].get('what','')[:80]}", elapsed, raw)

    return TestResult("implied_facts", False, "No Karlie fact extracted at all", elapsed, raw)


def test_noise(model: str, api_key: str, conv48: dict) -> TestResult:
    """Test 6: Does the model extract junk facts like 'asked a question'?

    Session 8: Deborah talks about stir-fry, Jolene talks about lasagna/exams.
    Good facts: food preferences, study habits, snake at park.
    Bad facts: 'Deborah asked a question', 'Jolene greeted Deborah'.
    """
    segment = format_session(conv48, "session_8")

    compactor_prompt = DEFAULT_SUMMARY_PROMPT.format(
        session_date="2023-03-02",
        tags="cooking, food, stress, exams, self-care",
        target_tokens=300,
        conversation_text=segment,
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("noise", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])

    noise_verbs = {
        "asked", "greeted", "responded", "said", "replied", "mentioned",
        "discussed", "talked", "inquired", "questioned", "noted", "expressed",
        "shared", "commented", "acknowledged", "suggested",
    }

    noise_facts = []
    for fact in facts:
        verb = fact.get("verb", "").lower().strip()
        # Check if the verb is purely conversational noise
        if verb in noise_verbs:
            noise_facts.append(f"{fact.get('subject','?')}|{verb}|{fact.get('object','?')[:40]}")

    if noise_facts:
        return TestResult("noise", False, f"{len(noise_facts)} noise facts: {noise_facts[:3]}", elapsed, raw)

    return TestResult("noise", True, f"{len(facts)} clean facts", elapsed, raw)


def test_json_compliance(results: list[TestResult]) -> TestResult:
    """Test 7: Did all outputs parse as valid JSON without stripping fences?

    Checks across all prior test results.
    """
    all_clean = True
    failures = []

    for r in results:
        if not r.raw:
            continue
        # For tag_reuse, there are multiple outputs separated by ---
        raws = r.raw.split("\n---\n") if "\n---\n" in r.raw else [r.raw]
        for raw in raws:
            raw = raw.strip()
            if not raw:
                continue
            try:
                json.loads(raw)
            except json.JSONDecodeError:
                all_clean = False
                failures.append(r.name)
                break

    if all_clean:
        return TestResult("json_compliance", True, "All outputs clean JSON", 0.0)
    else:
        return TestResult("json_compliance", False, f"Fenced/broken JSON in: {failures}", 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_battery(model: str, api_key: str, verbose: bool = False) -> list[TestResult]:
    """Run all 7 tests against a model."""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # Load data
    conv48 = load_locomo_conv48()
    mrcr_tags = load_mrcr_tags()
    print(f"Loaded Conv-48 ({conv48['sample_id']}), {len(mrcr_tags)} MRCR tags")

    results = []
    tests = [
        ("1. Contamination", lambda: test_contamination(model, api_key, conv48, mrcr_tags)),
        ("2. Tag Reuse",     lambda: test_tag_reuse(model, api_key, conv48)),
        ("3. Verb Accuracy", lambda: test_verb_accuracy(model, api_key, conv48)),
        ("4. Pronoun Resol", lambda: test_pronoun_resolution(model, api_key, conv48)),
        ("5. Implied Facts", lambda: test_implied_facts(model, api_key, conv48)),
        ("6. Noise",         lambda: test_noise(model, api_key, conv48)),
    ]

    for label, test_fn in tests:
        sys.stdout.write(f"  {label}... ")
        sys.stdout.flush()
        try:
            result = test_fn()
            results.append(result)
            status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
            print(f"{status} ({result.elapsed:.1f}s) — {result.detail}")
            if verbose and not result.passed:
                print(f"    RAW: {result.raw[:300]}")
        except Exception as e:
            results.append(TestResult(label, False, f"ERROR: {e}", 0.0))
            print(f"\033[31mERROR\033[0m — {e}")

    # Test 7: JSON compliance (meta-test across all results)
    sys.stdout.write("  7. JSON Compliance... ")
    sys.stdout.flush()
    json_result = test_json_compliance(results)
    results.append(json_result)
    status = "\033[32mPASS\033[0m" if json_result.passed else "\033[31mFAIL\033[0m"
    print(f"{status} — {json_result.detail}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.elapsed for r in results)
    median_time = sorted(r.elapsed for r in results if r.elapsed > 0)[len([r for r in results if r.elapsed > 0]) // 2] if any(r.elapsed > 0 for r in results) else 0

    print(f"\n  Score: {passed}/{total} | Total: {total_time:.1f}s | Median: {median_time:.1f}s")

    return results


def print_matrix(all_results: dict[str, list[TestResult]]):
    """Print a comparison matrix across all models."""
    test_names = ["contamination", "tag_reuse", "verb_accuracy", "pronoun_resolution", "implied_facts", "noise", "json_compliance"]
    short = ["Contam", "Reuse", "Verb", "Who", "Implied", "Noise", "JSON"]

    print(f"\n{'='*80}")
    print("RESULTS MATRIX")
    print(f"{'='*80}")

    # Header
    header = f"{'Model':<40} " + " ".join(f"{s:>7}" for s in short) + f" {'Score':>6} {'Speed':>6}"
    print(header)
    print("-" * len(header))

    for model, results in all_results.items():
        model_short = model.split("/")[-1][:38]
        cells = []
        for tn in test_names:
            r = next((r for r in results if r.name == tn), None)
            if r is None:
                cells.append("  —  ")
            elif r.passed:
                cells.append(" \033[32m PASS\033[0m ")
            else:
                cells.append(" \033[31m FAIL\033[0m ")

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        times = [r.elapsed for r in results if r.elapsed > 0]
        median = sorted(times)[len(times) // 2] if times else 0

        print(f"{model_short:<40} {''.join(cells)} {passed}/{total}   {median:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Battery for VC pipeline")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001", help="OpenRouter model ID or 'all'")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw output on failures")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    if not LOCOMO_PATH.exists():
        print(f"ERROR: LocOMo data not found at {LOCOMO_PATH}")
        print("Clone https://github.com/snap-research/locomo to ~/projects/locomo/")
        sys.exit(1)

    models = KNOWN_MODELS if args.model == "all" else [args.model]

    all_results = {}
    for model in models:
        results = run_battery(model, api_key, verbose=args.verbose)
        all_results[model] = results

    if len(models) > 1:
        print_matrix(all_results)


if __name__ == "__main__":
    main()
