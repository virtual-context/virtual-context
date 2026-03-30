#!/usr/bin/env python3
"""
Virtual Context Model Evaluation Battery
=========================================

Tests whether an LLM is suitable for use as a tagger/compactor in the
Virtual Context (VC) ingestion pipeline. VC uses an LLM to:

  1. TAG conversation turns (assign 2-5 topic labels)
  2. SUMMARIZE segments (compress into structured JSON with facts)

Different models fail in different ways. This battery tests 7 failure
modes using real conversational data. A model that passes all 7 is
viable for the VC pipeline. ~30 seconds per model via OpenRouter.

Usage:
    OPENROUTER_API_KEY=sk-... python run_battery.py --model google/gemini-2.0-flash-001
    OPENROUTER_API_KEY=sk-... python run_battery.py --model thudm/glm-4-32b
    OPENROUTER_API_KEY=sk-... python run_battery.py --model all

Tests:
    1. Contamination  -- Given 480 existing tags + a pants scene, does the model
                         hallucinate "lost-keys" by cross-contaminating from the tag list?
    2. Tag Reuse      -- Given 4 sequential turns (3 gaming, 1 yoga), does the model
                         reuse tags on same-topic turns and create new ones on topic shift?
    3. Verb Accuracy  -- "We were given Battlefield 1" -> verb should be "were given",
                         NOT "mentioned" or "discussed"
    4. Pronoun Resol. -- "We are planning to play Walking Dead" + prior context about
                         partner -> "who" field should resolve to include "partner"
    5. Implied Facts  -- "Last photo with Karlie... sadly last time together" -> model
                         should extract that Karlie passed away (implied, not stated)
    6. Noise          -- Cooking conversation -> 0 junk facts like "asked a question"
                         or "greeted" (conversational acts, not substantive facts)
    7. JSON           -- All outputs parse as valid JSON without markdown fence stripping

Requirements:
    pip install requests

Data files (included in this package):
    data/mrcr_tags.json      -- 480 topic tags from a real MRCR benchmark session
    data/test_fixtures.json  -- Conversation text for the noise test (LocOMo Conv-48)
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
# Paths -- data files shipped with this package
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
# These are the actual production prompts from the Virtual Context codebase.
# The tagger prompt tells the LLM how to assign tags and extract facts.
# The compactor prompt tells the LLM how to summarize a conversation segment.

TAGGER_SYSTEM_PROMPT = """\
You are a semantic tagger for conversation segments. Given a piece of conversation,
generate 2-5 short, lowercase tags that capture the key topics.

Rules:
- Prefer specific tags over generic ones. A good tag should narrow down WHICH
  conversation this came from, not just what broad category it falls into.
  Single-word tags are fine when already specific ("database", "teeth", "fitness").
  But when a single word is too broad and could match many unrelated conversations,
  qualify it with a hyphenated compound: "reservation-timing" not "timing",
  "cycle-tracking" not "tracking", "transit-schedule" not "schedule".
  Ask yourself: "Would this tag match conversations about completely different topics?"
  If yes, make it more specific.
- STRONGLY prefer reusing existing tags over creating new ones. Before inventing
  ANY new tag, check whether an existing tag already covers that topic -- even if
  the wording is slightly different. Use "teeth" not "dental" if "teeth" exists.
  Use "data-visualization" not "data-visualization-tools" or "visualization-techniques".
  Do NOT create variants by appending -tips, -tools, -techniques, -strategies, -options,
  -resources, -planning, -management, etc. to an existing tag stem.
  A new tag is only justified when the topic is genuinely absent from the existing list.
- When the text introduces a genuinely NEW topic not covered by any existing tag,
  create a new tag. Do NOT force-fit unrelated text into existing tags.
- Only add the tag "rule" when the user gives the assistant an explicit standing
  instruction about response style -- phrases like "always ...", "never ...",
  "from now on ...", "don't sugarcoat", "be honest with me".
  Do NOT tag as "rule": personal opinions ("I hate running"), feelings ("I'm
  overwhelmed"), desires ("I want to learn X"), questions, or topic switches.
- Set "temporal" to true when the query references a specific time position in
  the conversation -- "the first thing we discussed", "at the beginning",
  "early on", "going way back", "when we first started". False for general
  queries, even if they reference the past ("remind me about X" is NOT temporal
  -- it's looking for a topic, not a time position).
- Generate 2-5 related_tags: alternate words someone might use when referring back
  to these same concepts later (e.g. if tagging a discussion about "materialized views",
  related_tags might include "caching", "precomputed", "feed-optimization").
  These help future recall when the user uses different vocabulary.
- Messages may contain channel metadata (e.g. "[Telegram NAME ...]", "[WhatsApp ...]",
  "[Discord ...]", "[message_id: NNN]", timestamps, sender info). Ignore all metadata
  formatting -- tag only the actual conversational content within the message.
- Do NOT generate tags about the communication medium, channel, group, or server
  (e.g. "messaging", "threading", "chat", "telegram-group",
  "discord-server", "slack-channel", "texting", "communication", "conversation").
  These describe WHERE or HOW the conversation happens, not WHAT it is about. Tag only the
  substantive topics being discussed.
- For very short or trivial messages (greetings, reactions, single-word responses,
  emoji), return only the tags that genuinely apply -- it is acceptable to return
  fewer than 2 tags when the content does not warrant more.
- Tag the concrete subject being discussed, not the conversational framing.
  "What do you think of trees?" -> tag "trees" or "nature", NOT "introspection"
  or "cognition". The question format ("what do you think", "how do you feel",
  "tell me about") is framing -- the subject is what matters for retrieval.
  Even if the assistant's response is philosophical or reflective, always include
  at least one tag for the concrete noun or topic the user asked about.
- Tag both what the conversation is about AND what the user reveals about
  their own life, experiences, or situation -- even if mentioned in passing.
- Extract facts about the user's life, experiences, preferences, plans, and world.
  For each fact, classify:
  - "fact_type": "personal" (user's life/identity/preferences), "experience" (assistant-provided info the user engaged with), or "world" (facts about other people, places, things in the user's world)
  - "subject": who -- use the actual name when conversation metadata identifies the sender (e.g. if metadata shows sender "Bob", the subject is "Bob", not "user"). When no name is available, use "user". For people mentioned but not speaking, use their name.
  - "verb": the exact action verb from the conversation (e.g. "led", "ordered", "prefers", "lives in")
  - "object": what (specific noun phrase -- preserve ALL numbers, names, dates, amounts)
  - "status": one of: active, completed, planned, abandoned, recurring
  - "what": one full sentence capturing the complete fact with ALL specifics preserved.
    WRONG: "User has a personal best time." RIGHT: "User has a personal best 5K time of 27:12."
    WRONG: "User paid a parking ticket." RIGHT: "User paid a $40 parking ticket."
  Extract the FACT behind the question, not the conversational act.
  WRONG: "user asks about Cairo restaurants" RIGHT: "user wants to try authentic Egyptian food in Cairo"
  DO NOT extract: mere asks, mentions, discusses, requests for information.
  Only extract facts with genuine substance. Skip greetings and filler.
  When a pronoun refers to a named person mentioned earlier, resolve it: "Emily (user's college roommate)".
- Return JSON only: {{"tags": ["tag1", "tag2"], "primary": "tag1", "temporal": false, "related_tags": ["alt1", "alt2"], "facts": [{{"subject": "user", "verb": "...", "object": "...", "status": "...", "fact_type": "personal|experience|world", "what": "..."}}]}}
- The "primary" tag is the single most relevant tag
- No markdown fences, no extra text
"""

COMPACTOR_SYSTEM_PROMPT = """\
SESSION DATE: {session_date}

Summarize the following conversation segment (tags: {tags}).
Preserve: key decisions, action items, entities mentioned, specific data points,
and specific feature/concept names exactly as discussed (e.g. "cook mode", "dark theme", "rate limiter" --
do NOT generalize these into broader categories like "UI features" or "infrastructure").

CRITICAL -- Any text involving numbers is mandatory and absolutely essential to the summary, always include them exactly as in the conversation.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number (e.g. "2 hours" must stay "2 hours", not
"about an hour"; "$45" must stay "$45", not "around $50").

When the user states what they are doing, have done, or where they keep/store something,
preserve that as a direct assertion, not as a plan or intention.

Capture the tone and texture of the conversation -- was it casual/playful, urgent/stressed,
analytical/technical, emotional/vulnerable, collaborative/brainstorming?

When speakers are identified by name in the conversation, always use their
actual name in the summary. Never replace a named speaker with "User" or "the user".

Be concise but retain enough detail that the conversation could be resumed from this summary.
The summary should be {target_tokens} tokens or fewer.

Conversation:
{conversation_text}

Respond with JSON:
{{
  "summary": "...",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."],
  "date_references": ["..."],
  "refined_tags": ["tag1", "tag2"],
  "related_tags": ["alternate-term1", "alternate-term2"]
}}

For "related_tags", generate 3-8 alternate terms someone might use to refer to these
concepts later.

Also extract facts from the RAW CONVERSATION TEXT above (not from your summary).
The summary may omit details -- facts must capture ALL substantive information
from every speaker in the conversation, even details not included in the summary.
For each fact:
- "subject": who -- use the actual name when conversation identifies the sender. When no name is available, use "user". For people mentioned but not speaking, use their name.
- "verb": the EXACT action verb from the conversation text (e.g. "led", "built", "prefers", "lives in", "ordered")
  VERB RULE: Use the verb that matches the actual event described.
  When someone says "we were given X", the verb is "were given" -- NOT "mentioned" or "discussed".
  When someone says "I went to X", the verb is "went to" -- NOT "talked about".
  Only use "mentioned"/"discussed" when the conversation is genuinely about referencing
  something without doing it.
- "object": what (specific noun phrase -- preserve ALL numbers, names, dates, amounts exactly)
- "status": one of: active, completed, planned, abandoned, recurring
- "fact_type": classify as "personal", "experience", or "world"
- "what": one full sentence capturing the complete fact with ALL specifics preserved.
- "who": ALL people involved (populate when present, empty string if n/a)
  WHO RULE: Resolve pronouns. "We" in a speaker's message means the speaker + someone.
  Use preceding context to determine who.
- "when": the calendar date this event occurred, resolved from context.
  DATE RULES -- use SESSION DATE ({session_date}) as your reference point:
  "today" / "this morning" -> use {session_date}.
  "yesterday" -> the day before {session_date}.
  "last Saturday" -> the most recent Saturday before {session_date}.
  Always write the RESOLVED calendar date (e.g. "2023-05-20"), NOT the relative term.
  If truly unresolvable -> use "".
- "where": location (populate when present, empty string if n/a)
- "why": context or significance (populate when present, empty string if n/a)
Extract the FACT behind the question, not the conversational act.
Extract both EXPLICIT and IMPLIED facts. When someone refers to "the last photo/time
with someone", "sadly the last time together", "I'm sorry for your loss" -- extract the
implied fact (e.g. the person passed away).
Include "facts" in the JSON response.
Only extract facts with genuine substance. Skip greetings and filler."""


# ---------------------------------------------------------------------------
# Known models (OpenRouter IDs) -- add your own
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
def load_mrcr_tags() -> list[str]:
    """Load the 480 MRCR tags from the bundled data file."""
    path = DATA_DIR / "mrcr_tags.json"
    if not path.exists():
        print(f"WARNING: {path} not found. Contamination test will use empty tag list.")
        return []
    with open(path) as f:
        return json.load(f)


def load_session_8_text() -> str:
    """Load the LocOMo Conv-48 Session 8 text for the noise test."""
    path = DATA_DIR / "test_fixtures.json"
    if not path.exists():
        print(f"WARNING: {path} not found. Noise test will fail.")
        return ""
    with open(path) as f:
        fixtures = json.load(f)
    return fixtures.get("session_8_text", "")


# ---------------------------------------------------------------------------
# LLM call via OpenRouter
# ---------------------------------------------------------------------------
def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float = 0.0,
    api_url: str = "https://openrouter.ai/api/v1/chat/completions",
) -> tuple[str, float]:
    """Call an OpenAI-compatible API. Returns (response_text, elapsed_seconds).

    Override api_url for local models:
        --api-url http://localhost:11434/v1/chat/completions  (Ollama)
        --api-url http://localhost:1234/v1/chat/completions   (LM Studio)
    """
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
    resp = requests.post(api_url, headers=headers, json=body, timeout=300)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content, elapsed


def parse_json_response(raw: str) -> tuple[dict | None, bool]:
    """Parse JSON from LLM response. Returns (parsed, clean_json).
    clean_json=True means no fence-stripping was needed."""
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        return None, False


# ---------------------------------------------------------------------------
# Test results
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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "elapsed_s": round(self.elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Test 1: Contamination
# ---------------------------------------------------------------------------
def test_contamination(model: str, api_key: str, api_url: str, mrcr_tags: list[str]) -> TestResult:
    """Does the model apply 'lost-keys' to a pants/couch scene?

    WHY THIS MATTERS: VC feeds the model a list of ALL existing tags from
    the session (which can be hundreds). A weak model sees tags like 'keys',
    'lost-items', and 'couch-cushions' in the list and hallucinates that a
    scene about finding PANTS under couch cushions is about lost keys.

    PASS: Tags do NOT include 'lost-keys' or 'keys'
    FAIL: Model cross-contaminated from the tag list
    """
    tag_line = (
        "Existing tags (reuse when the topic genuinely matches, "
        "but create new tags for new topics): " + ", ".join(mrcr_tags)
    )

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

    raw, elapsed = call_llm(model, TAGGER_SYSTEM_PROMPT, user_prompt, api_key, api_url=api_url)
    parsed, clean = parse_json_response(raw)

    if parsed is None:
        return TestResult("contamination", False, "JSON parse failed", elapsed, raw)

    tags = [t.lower() for t in parsed.get("tags", [])]

    bad_tags = [t for t in tags if "lost-key" in t or t == "lost-keys"]
    if bad_tags:
        return TestResult("contamination", False, f"Invented {bad_tags}", elapsed, raw)
    if "keys" in tags:
        return TestResult("contamination", False, "Applied 'keys' tag to pants scene", elapsed, raw)

    return TestResult("contamination", True, f"tags={tags}", elapsed, raw)


# ---------------------------------------------------------------------------
# Test 2: Tag Reuse
# ---------------------------------------------------------------------------
def test_tag_reuse(model: str, api_key: str, api_url: str) -> TestResult:
    """Does the model reuse existing tags on same-topic turns?

    WHY THIS MATTERS: VC groups conversation turns by tag. If the model
    creates synonyms ("gaming", "video-games", "console-gaming") instead of
    reusing the original tag, related turns get scattered across different
    segments. This destroys the compactor's ability to build coherent summaries.

    SETUP: 4 sequential turns from a real conversation:
      Turn 1: Gaming + Detroit (creates initial tags)
      Turn 2: Learning to play (SHOULD reuse gaming tags, not invent new ones)
      Turn 3: Walking Dead planning (SHOULD reuse gaming tags)
      Turn 4: Running group + yoga (SHOULD create NEW tags -- different topic)

    PASS: 0-2 new tags on gaming turns 2+3, AND at least 1 new tag on yoga turn 4
    FAIL: >2 new tags leaked on gaming turns, OR 0 new tags on the yoga turn
    """
    turns = [
        # Turn 1: Gaming intro
        'Jolene: They are very unusual pets! Here\'s me and my partner gaming last week '
        '- it\'s so fun. We played the game "Detroit" on the console. '
        'We are both crazy about this activity!',
        # Turn 2: Learning to play
        "Deborah: Did your boyfriend teach you to play?\n"
        "Jolene: Even as a child I learned to play on my own.",
        # Turn 3: Walking Dead
        'Deborah: Do you only play old games or try new ones?\n'
        'Jolene: We are planning to play "Walking Dead" next Saturday.',
        # Turn 4: Fitness (topic shift)
        "Deborah: Hey Jolene! I started a running group with Anna - it's awesome "
        "connecting with people who care about fitness!\n"
        "Jolene: Cool, Deb! Glad you found some people to get fit with.",
    ]

    all_tags = []
    new_tags_per_turn = []
    total_elapsed = 0
    raw_outputs = []
    accumulated_tags = []

    for i, turn in enumerate(turns):
        if accumulated_tags:
            tag_line = (
                "Existing tags (reuse when the topic genuinely matches, "
                "but create new tags for new topics): " + ", ".join(accumulated_tags)
            )
            user_prompt = f"{tag_line}\n\nTag this conversation (return 2-5 tags):\n\n{turn}"
        else:
            user_prompt = f"Tag this conversation (return 2-5 tags):\n\n{turn}"

        raw, elapsed = call_llm(model, TAGGER_SYSTEM_PROMPT, user_prompt, api_key, api_url=api_url)
        total_elapsed += elapsed
        raw_outputs.append(raw)

        parsed, _ = parse_json_response(raw)
        if parsed is None:
            return TestResult(
                "tag_reuse", False,
                f"JSON parse failed on turn {i+1}",
                total_elapsed, "\n---\n".join(raw_outputs),
            )

        tags = [t.lower() for t in parsed.get("tags", [])]
        new = [t for t in tags if t not in accumulated_tags]
        new_tags_per_turn.append(new)
        all_tags.append(tags)
        accumulated_tags = list(set(accumulated_tags + tags))

    new_on_gaming = len(new_tags_per_turn[1]) + len(new_tags_per_turn[2])
    new_on_yoga = len(new_tags_per_turn[3])

    detail = (
        f"new_tags=[{len(new_tags_per_turn[0])},"
        f"{len(new_tags_per_turn[1])},"
        f"{len(new_tags_per_turn[2])},"
        f"{len(new_tags_per_turn[3])}]"
    )

    if new_on_gaming > 2:
        return TestResult(
            "tag_reuse", False,
            f"Leaked {new_on_gaming} new tags on gaming turns. {detail}",
            total_elapsed, "\n---\n".join(raw_outputs),
        )

    if new_on_yoga == 0:
        return TestResult(
            "tag_reuse", False,
            f"No new tags on yoga turn. {detail}",
            total_elapsed, "\n---\n".join(raw_outputs),
        )

    return TestResult("tag_reuse", True, detail, total_elapsed, "\n---\n".join(raw_outputs))


# ---------------------------------------------------------------------------
# Test 3: Verb Accuracy
# ---------------------------------------------------------------------------
def test_verb_accuracy(model: str, api_key: str, api_url: str) -> TestResult:
    """Does the model extract 'were given' or downgrade to 'mentioned'?

    WHY THIS MATTERS: VC stores structured facts with subject-verb-object.
    If every verb becomes "mentioned" or "discussed", facts lose their meaning.
    "Jolene were given Battlefield 1" is a real event.
    "Jolene mentioned Battlefield 1" is a conversational act.
    The difference matters for answering "What game did Jolene receive?"

    PASS: verb is "were given", "was given", "received", or "got"
    FAIL: verb is "mentioned", "discussed", "talked about", etc.
    """
    segment = (
        "Jolene: Long time no talk! We were given a new game for the console last week, "
        "it is Battlefield 1. What's been up with you?\n"
        "Deborah: Hey Jolene! Good to hear from you. That's cool!"
    )

    compactor_prompt = COMPACTOR_SYSTEM_PROMPT.format(
        session_date="2023-08-21",
        tags="gaming, battlefield, console",
        target_tokens=200,
        conversation_text=segment,
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key, api_url=api_url)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("verb_accuracy", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])

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

    actual_verb = bf_facts[0].get("verb", "")
    return TestResult("verb_accuracy", True, f"verb='{actual_verb}' (acceptable)", elapsed, raw)


# ---------------------------------------------------------------------------
# Test 4: Pronoun Resolution
# ---------------------------------------------------------------------------
def test_pronoun_resolution(model: str, api_key: str, api_url: str) -> TestResult:
    """Does the model resolve 'We' to Jolene + partner?

    WHY THIS MATTERS: VC compacts conversations into summaries with structured
    facts. If "We are planning to play Walking Dead" becomes a fact with
    who="Jolene" (missing partner), downstream queries like "Who is Jolene
    playing Walking Dead with?" get no answer.

    SETUP: Prior context establishes Jolene games with her partner. Current
    segment says "We are planning to play Walking Dead."

    PASS: "who" field contains "partner"
    FAIL: "partner" missing from all Walking Dead facts
    """
    prev_context = (
        'Jolene: Here\'s me and my partner gaming last week. '
        'We played the game "Detroit" on the console.'
    )
    segment = 'Jolene: We are planning to play "Walking Dead" next Saturday.'

    compactor_prompt = COMPACTOR_SYSTEM_PROMPT.format(
        session_date="2023-01-27",
        tags="gaming, walking-dead, console",
        target_tokens=200,
        conversation_text=f"[Previous context]\n{prev_context}\n\n[Current segment]\n{segment}",
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key, api_url=api_url)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("pronoun_resolution", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])

    wd_facts = [
        f for f in facts
        if "walking dead" in f.get("object", "").lower()
        or "walking dead" in f.get("what", "").lower()
        or "planning to play" in f.get("what", "").lower()
    ]

    if not wd_facts:
        return TestResult("pronoun_resolution", False, "No Walking Dead fact extracted", elapsed, raw)

    for fact in wd_facts:
        who = fact.get("who", "").lower()
        what = fact.get("what", "").lower()
        subject = fact.get("subject", "").lower()
        if "partner" in who or "partner" in what or "partner" in subject:
            return TestResult("pronoun_resolution", True, f"who='{fact.get('who','')}'", elapsed, raw)

    for fact in wd_facts:
        all_text = json.dumps(fact).lower()
        if "partner" in all_text:
            return TestResult("pronoun_resolution", True, "partner found in fact", elapsed, raw)

    who_vals = [f.get("who", "") for f in wd_facts]
    return TestResult("pronoun_resolution", False, f"No partner resolution. who={who_vals}", elapsed, raw)


# ---------------------------------------------------------------------------
# Test 5: Implied Facts
# ---------------------------------------------------------------------------
def test_implied_facts(model: str, api_key: str, api_url: str) -> TestResult:
    """Does the model extract 'Karlie passed away' from indirect language?

    WHY THIS MATTERS: Users don't always state facts directly. "This is the
    last photo with Karlie... sadly the last time we were together" + the
    reply "I'm sorry for your loss" implies Karlie died. VC needs to store
    the implied fact so that "What happened to Karlie?" gets an answer.

    PASS: Any fact mentioning Karlie contains "passed away", "died", or similar
    FAIL: No death-related fact for Karlie
    """
    segment = (
        "Deborah: Memories keep our loved ones close. This is the last photo with "
        "Karlie which was taken last summer when we hiked. It was our last one. "
        "We had such a great time! Every time I see it, I can't help but smile.\n"
        "Jolene: I'm sorry for your loss. It sounds like you had wonderful times together."
    )

    compactor_prompt = COMPACTOR_SYSTEM_PROMPT.format(
        session_date="2023-02-22",
        tags="loss, friendship, karlie, garden, travel",
        target_tokens=300,
        conversation_text=segment,
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key, api_url=api_url)
    parsed, _ = parse_json_response(raw)

    if parsed is None:
        return TestResult("implied_facts", False, "JSON parse failed", elapsed, raw)

    facts = parsed.get("facts", [])
    death_terms = ["passed away", "died", "death", "passing", "lost", "passed"]

    for fact in facts:
        fact_text = json.dumps(fact).lower()
        if "karlie" in fact_text:
            for term in death_terms:
                if term in fact_text:
                    return TestResult(
                        "implied_facts", True,
                        f"Found: {fact.get('what', '')[:80]}",
                        elapsed, raw,
                    )

    summary = parsed.get("summary", "").lower()
    if "karlie" in summary and any(t in summary for t in death_terms):
        return TestResult("implied_facts", True, "Implied in summary (not facts)", elapsed, raw)

    karlie_facts = [f for f in facts if "karlie" in json.dumps(f).lower()]
    if karlie_facts:
        return TestResult(
            "implied_facts", False,
            f"Karlie mentioned but no death: {karlie_facts[0].get('what','')[:80]}",
            elapsed, raw,
        )

    return TestResult("implied_facts", False, "No Karlie fact extracted at all", elapsed, raw)


# ---------------------------------------------------------------------------
# Test 6: Noise
# ---------------------------------------------------------------------------
def test_noise(model: str, api_key: str, api_url: str) -> TestResult:
    """Does the model extract junk facts like 'asked a question'?

    WHY THIS MATTERS: A cooking conversation should produce facts like
    "Deborah loves stir-fry" and "Jolene has exams coming up." Facts like
    "Deborah asked Jolene a question" or "Jolene greeted Deborah" are noise
    that waste storage and confuse retrieval. VC's fact DB would bloat with
    useless entries.

    PASS: 0 noise facts (verbs: asked, greeted, responded, said, replied, etc.)
    FAIL: Any conversational-act facts extracted
    """
    session_8_text = load_session_8_text()
    if not session_8_text:
        return TestResult("noise", False, "Missing test fixture data/test_fixtures.json", 0.0)

    compactor_prompt = COMPACTOR_SYSTEM_PROMPT.format(
        session_date="2023-03-02",
        tags="cooking, food, stress, exams, self-care",
        target_tokens=300,
        conversation_text=session_8_text,
    )

    raw, elapsed = call_llm(model, "", compactor_prompt, api_key, api_url=api_url)
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
        if verb in noise_verbs:
            noise_facts.append(
                f"{fact.get('subject','?')}|{verb}|{fact.get('object','?')[:40]}"
            )

    if noise_facts:
        return TestResult(
            "noise", False,
            f"{len(noise_facts)} noise facts: {noise_facts[:3]}",
            elapsed, raw,
        )

    return TestResult("noise", True, f"{len(facts)} clean facts", elapsed, raw)


# ---------------------------------------------------------------------------
# Test 7: JSON Compliance
# ---------------------------------------------------------------------------
def test_json_compliance(results: list[TestResult]) -> TestResult:
    """Did all outputs parse as valid JSON without stripping fences?

    WHY THIS MATTERS: VC parses LLM output as JSON. If the model wraps its
    response in ```json ... ``` markdown fences, VC has to strip them (fragile).
    Models that return clean JSON are more reliable in production.

    PASS: All outputs from tests 1-6 parse as raw JSON
    FAIL: Any output required fence stripping or failed to parse
    """
    all_clean = True
    failures = []

    for r in results:
        if not r.raw:
            continue
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
# Runner
# ---------------------------------------------------------------------------
def run_battery(
    model: str,
    api_key: str,
    api_url: str,
    verbose: bool = False,
    output_json: str | None = None,
) -> list[TestResult]:
    """Run all 7 tests against a model."""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"API:   {api_url}")
    print(f"{'='*60}")

    mrcr_tags = load_mrcr_tags()
    print(f"Loaded {len(mrcr_tags)} MRCR tags, session 8 fixture")

    results = []
    tests = [
        ("1. Contamination", lambda: test_contamination(model, api_key, api_url, mrcr_tags)),
        ("2. Tag Reuse",     lambda: test_tag_reuse(model, api_key, api_url)),
        ("3. Verb Accuracy", lambda: test_verb_accuracy(model, api_key, api_url)),
        ("4. Pronoun Resol", lambda: test_pronoun_resolution(model, api_key, api_url)),
        ("5. Implied Facts", lambda: test_implied_facts(model, api_key, api_url)),
        ("6. Noise",         lambda: test_noise(model, api_key, api_url)),
    ]

    for label, test_fn in tests:
        sys.stdout.write(f"  {label}... ")
        sys.stdout.flush()
        try:
            result = test_fn()
            results.append(result)
            status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
            print(f"{status} ({result.elapsed:.1f}s) -- {result.detail}")
            if verbose and not result.passed:
                print(f"    RAW: {result.raw[:500]}")
        except Exception as e:
            results.append(TestResult(label, False, f"ERROR: {e}", 0.0))
            print(f"\033[31mERROR\033[0m -- {e}")

    # Test 7: JSON compliance (meta-test)
    sys.stdout.write("  7. JSON Compliance... ")
    sys.stdout.flush()
    json_result = test_json_compliance(results)
    results.append(json_result)
    status = "\033[32mPASS\033[0m" if json_result.passed else "\033[31mFAIL\033[0m"
    print(f"{status} -- {json_result.detail}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    times = [r.elapsed for r in results if r.elapsed > 0]
    total_time = sum(times)
    median_time = sorted(times)[len(times) // 2] if times else 0

    print(f"\n  Score: {passed}/{total} | Total: {total_time:.1f}s | Median: {median_time:.1f}s")

    # Save JSON results if requested
    if output_json:
        output = {
            "model": model,
            "api_url": api_url,
            "score": f"{passed}/{total}",
            "total_time_s": round(total_time, 2),
            "median_time_s": round(median_time, 2),
            "tests": [r.to_dict() for r in results],
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to {output_json}")

    return results


def print_matrix(all_results: dict[str, list[TestResult]]):
    """Print a comparison matrix across all models."""
    short = ["Contam", "Reuse", "Verb", "Who", "Implied", "Noise", "JSON"]
    test_names = [
        "contamination", "tag_reuse", "verb_accuracy",
        "pronoun_resolution", "implied_facts", "noise", "json_compliance",
    ]

    print(f"\n{'='*80}")
    print("RESULTS MATRIX")
    print(f"{'='*80}")

    header = f"{'Model':<40} " + " ".join(f"{s:>7}" for s in short) + f" {'Score':>6} {'Speed':>6}"
    print(header)
    print("-" * len(header))

    for model, results in all_results.items():
        model_short = model.split("/")[-1][:38]
        cells = []
        for tn in test_names:
            r = next((r for r in results if r.name == tn), None)
            if r is None:
                cells.append("   --  ")
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
    parser = argparse.ArgumentParser(
        description="Virtual Context Model Evaluation Battery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single model via OpenRouter
  OPENROUTER_API_KEY=sk-... python run_battery.py --model google/gemini-2.0-flash-001

  # Test a local model via Ollama
  python run_battery.py --model glm4:32b --api-url http://localhost:11434/v1/chat/completions --api-key unused

  # Test a local model via LM Studio
  python run_battery.py --model loaded-model --api-url http://localhost:1234/v1/chat/completions --api-key unused

  # Test all known OpenRouter models
  OPENROUTER_API_KEY=sk-... python run_battery.py --model all

  # Save results to JSON
  OPENROUTER_API_KEY=sk-... python run_battery.py --model thudm/glm-4-32b --output results.json
""",
    )
    parser.add_argument(
        "--model", default="google/gemini-2.0-flash-001",
        help="Model ID (OpenRouter format, or local model name). Use 'all' for all known models.",
    )
    parser.add_argument(
        "--api-url", default="https://openrouter.ai/api/v1/chat/completions",
        help="API endpoint URL. Override for local models (Ollama, LM Studio, vLLM, etc.)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (or set OPENROUTER_API_KEY env var). Use 'unused' for local models.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw output on failures")
    parser.add_argument("--output", "-o", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    models = KNOWN_MODELS if args.model == "all" else [args.model]

    all_results = {}
    for model in models:
        output_file = args.output
        if args.output and len(models) > 1:
            base, ext = os.path.splitext(args.output)
            output_file = f"{base}_{model.replace('/', '_')}{ext}"
        results = run_battery(model, api_key, args.api_url, verbose=args.verbose, output_json=output_file)
        all_results[model] = results

    if len(models) > 1:
        print_matrix(all_results)


if __name__ == "__main__":
    main()
