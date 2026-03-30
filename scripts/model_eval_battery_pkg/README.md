# Virtual Context Model Evaluation Battery

Tests whether an LLM is suitable as a **tagger/compactor** in the Virtual Context (VC) pipeline.

VC uses an LLM at ingestion time to:
1. **Tag** conversation turns with 2-5 topic labels (e.g. "gaming", "fitness")
2. **Summarize** segments into structured JSON with facts, entities, decisions

Different models fail in different ways. This battery tests 7 failure modes using real conversational data. A model that passes all 7 is viable for the VC pipeline.

## Quick Start

```bash
# One dependency
pip install requests

# Run against a model via OpenRouter (~30 seconds, ~$0.01)
OPENROUTER_API_KEY=sk-... python run_battery.py --model google/gemini-2.0-flash-001

# Run against a local model (Ollama, LM Studio, vLLM, etc.)
python run_battery.py \
  --model glm4:32b \
  --api-url http://localhost:11434/v1/chat/completions \
  --api-key unused

# Run all known models + comparison matrix
OPENROUTER_API_KEY=sk-... python run_battery.py --model all

# Save results to JSON
python run_battery.py --model thudm/glm-4-32b --output results.json
```

## The 7 Tests

### Test 1: Contamination

**What it tests:** Given 480 real topic tags from a session + a new scene about *finding pants under couch cushions*, does the model hallucinate `lost-keys` by cross-contaminating from the tag list (which contains keys-related tags from unrelated conversations)?

**Why it matters:** VC feeds ALL existing session tags to the tagger so it can reuse them. A weak model sees "keys" and "couch-cushions" in the list and invents `lost-keys` for an unrelated pants scene. This pollutes the tag index and causes wrong segments to match future queries.

**Pass:** Output tags do NOT include `lost-keys` or `keys`.

### Test 2: Tag Reuse

**What it tests:** 4 sequential turns -- 3 about gaming (Detroit, learning to play, Walking Dead) then 1 about running/yoga. Does the model reuse the gaming tags on turns 2-3, and create new tags on the yoga turn?

**Why it matters:** VC groups turns by tag. If the model creates synonyms (`gaming`, `video-games`, `console-gaming`) instead of reusing the original tag, related turns scatter across different segments. Compaction produces fragmented summaries and retrieval degrades.

**Pass:** 0-2 new tags on gaming turns 2+3, AND at least 1 new tag on yoga turn 4.

### Test 3: Verb Accuracy

**What it tests:** "We were given a new game for the console last week, it is Battlefield 1." The extracted verb should be `were given` / `received`, NOT `mentioned` / `discussed`.

**Why it matters:** VC stores structured facts (subject-verb-object). If every verb becomes `mentioned`, facts lose meaning. "Jolene received Battlefield 1" answers "What game did Jolene get?" while "Jolene mentioned Battlefield 1" does not.

**Pass:** Verb in {`were given`, `was given`, `received`, `got`, `were gifted`}. Fails on {`mentioned`, `discussed`, `talked about`, `said`}.

### Test 4: Pronoun Resolution

**What it tests:** Prior context says "Here's me and my partner gaming." Current segment says "We are planning to play Walking Dead." Does the model resolve "We" to include "partner" in the fact's `who` field?

**Why it matters:** Unresolved pronouns destroy fact utility. "Jolene plans to play Walking Dead" misses half the answer to "Who is Jolene playing Walking Dead with?"

**Pass:** `partner` appears in the Walking Dead fact (in who, subject, or what field).

### Test 5: Implied Facts

**What it tests:** "This is the last photo with Karlie... our last one... sadly the last time we were together" + reply "I'm sorry for your loss." Does the model extract that Karlie passed away?

**Why it matters:** Users don't always state things directly. VC must capture implied facts so that "What happened to Karlie?" gets an answer months later when the original conversation has been compacted away.

**Pass:** Any Karlie-related fact contains `passed away`, `died`, `death`, `passing`, or `lost`.

### Test 6: Noise

**What it tests:** A cooking conversation where Deborah talks about stir-fry and Jolene about exams. Does the model extract junk facts like "Deborah asked a question" or "Jolene greeted Deborah"?

**Why it matters:** Noise facts bloat the fact database and confuse retrieval. Good facts: "Deborah loves stir-fry", "Jolene has exams coming up." Bad facts: "Deborah asked Jolene a question" (that's a conversational act, not a substantive fact).

**Pass:** Zero noise facts. Noise verbs: `asked`, `greeted`, `responded`, `said`, `replied`, `mentioned`, `discussed`, `talked`, `inquired`, `questioned`, `noted`, `expressed`, `shared`, `commented`, `acknowledged`, `suggested`.

### Test 7: JSON Compliance

**What it tests:** Did all outputs from tests 1-6 parse as valid JSON without needing to strip markdown fences (` ```json ... ``` `)?

**Why it matters:** VC parses LLM output as raw JSON. Models that wrap in fences require fragile stripping. Clean JSON is more reliable in production.

**Pass:** All test outputs parse via `json.loads()` without preprocessing.

## Reference Results (2026-03-15)

### Models that pass ALL 7 tests:

| Model | Size | Speed (OpenRouter) | Local 5090 | Local Spark |
|-------|:----:|:------------------:|:----------:|:-----------:|
| GLM-4 32B | 32B | 0.2 turn/s | YES (19GB) | YES (19GB) |
| MiniMax M2.5 | 45.9B | 3-14s | NO (138GB) | NO |
| Qwen2.5 72B* | 72B | ~30 tok/s | YES (~36GB) | YES |

*Qwen2.5 72B leaks 2 tags on reuse test (borderline pass).

### Models that pass 4-5/7:

| Model | Fails | Notes |
|-------|-------|-------|
| Llama 4 Maverick | Implied | 17B active, 243GB total |
| Command-A | Implied | ~100B |
| Qwen3 235B-A22B | Verb | 22B active |
| Kimi K2.5 | Reuse | Very slow: 35-167s/call |

### Models that fail multiple tests:

| Model | Fails |
|-------|-------|
| Gemini 2.0 Flash | Who, Implied (but clean at scale, current VC default) |
| GLM-4.7 Flash | Who, Implied |
| Mistral Large | Implied, JSON |
| Hermes 4 70B | Who, JSON |

## What Each Prompt Does

The battery uses two prompts from the VC codebase:

### Tagger Prompt (tests 1-2)
Tells the model to assign 2-5 topic tags to a conversation turn. Key rules:
- Reuse existing tags over creating synonyms
- Generate `related_tags` for vocabulary bridging
- Extract structured facts alongside tags
- No tags about communication medium ("chat", "messaging")

### Compactor Prompt (tests 3-6)
Tells the model to summarize a conversation segment into structured JSON. Key rules:
- **VERB RULE**: Use the actual event verb, not "mentioned"/"discussed"
- **WHO RULE**: Resolve pronouns using preceding context
- **DATE RULES**: Resolve relative dates ("yesterday") to calendar dates
- **IMPLIED FACTS**: Extract death, events from indirect language
- Preserve ALL numbers, names, dates exactly as stated
- Skip conversational acts (greetings, filler)

## File Structure

```
model_eval_battery_pkg/
  run_battery.py           -- Self-contained test runner (no VC dependencies)
  README.md                -- This file
  data/
    mrcr_tags.json         -- 480 topic tags from a real benchmark session
    test_fixtures.json     -- Conversation text for the noise test
```

## Interpreting Results

- **7/7**: Model is production-ready for the VC pipeline
- **6/7**: Check which test failed. Implied facts and pronoun resolution are the hardest; failing one of these is common and may be acceptable depending on use case
- **5/7**: Borderline. Check whether the failures are in tagging (tests 1-2) or fact extraction (tests 3-6). Tagging failures are worse because they affect segment grouping
- **<5/7**: Not recommended for VC

The **speed** column matters too. A model that passes all tests but takes 60s per call is unusable as a real-time tagger. Target: <5s per call for production use.
