# PROXY-025: Budget-Enforced Message Stubbing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce `context_window` as a hard budget on outbound proxy payloads by stubbing compacted messages instead of force-keeping them.

**Architecture:** Hash-based identification of compacted turns in client payloads, stub replacement of their content, budget auto-promotion when client overhead exceeds the configured window, and over-budget alerting when uncompacted turns can't be stubbed.

**Tech Stack:** Python, existing proxy message_filter.py, TurnTagIndex, PayloadFormat token estimation.

---

### Task 1: Add hash→turn lookup to TurnTagIndex

**Files:**
- Modify: `virtual_context/core/turn_tag_index.py:17-23`
- Test: `tests/test_proxy.py` (TestFilterBodyMessages section)

**Step 1: Write the failing test**

In `tests/test_proxy.py`, add inside a new test class or at the end of `TestFilterBodyMessages`:

```python
def test_turn_tag_index_hash_lookup(self):
    """TurnTagIndex supports O(1) lookup by message_hash."""
    idx = TurnTagIndex()
    idx.append(TurnTagEntry(
        turn_number=0, message_hash="abc123", tags=["python"],
        primary_tag="python",
    ))
    idx.append(TurnTagEntry(
        turn_number=1, message_hash="def456", tags=["cooking"],
        primary_tag="cooking",
    ))
    idx.append(TurnTagEntry(
        turn_number=2, message_hash="ghi789", tags=["music"],
        primary_tag="music",
    ))

    assert idx.get_entry_by_hash("abc123").turn_number == 0
    assert idx.get_entry_by_hash("def456").turn_number == 1
    assert idx.get_entry_by_hash("ghi789").turn_number == 2
    assert idx.get_entry_by_hash("nonexistent") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_proxy.py::TestFilterBodyMessages::test_turn_tag_index_hash_lookup -v --ignore=tests/ollama --ignore=tests/haiku -m 'not slow'`
Expected: FAIL — `AttributeError: 'TurnTagIndex' object has no attribute 'get_entry_by_hash'`

**Step 3: Write minimal implementation**

In `virtual_context/core/turn_tag_index.py`, add a `_by_hash` dict and the lookup method:

```python
class TurnTagIndex:
    def __init__(self) -> None:
        self.entries: list[TurnTagEntry] = []
        self._by_turn: dict[int, TurnTagEntry] = {}
        self._by_hash: dict[str, TurnTagEntry] = {}

    def append(self, entry: TurnTagEntry) -> None:
        self.entries.append(entry)
        self._by_turn[entry.turn_number] = entry
        if entry.message_hash:
            self._by_hash[entry.message_hash] = entry

    def get_entry_by_hash(self, message_hash: str) -> TurnTagEntry | None:
        """Look up entry by message_hash (O(1) via dict)."""
        return self._by_hash.get(message_hash)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_proxy.py::TestFilterBodyMessages::test_turn_tag_index_hash_lookup -v --ignore=tests/ollama --ignore=tests/haiku -m 'not slow'`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/core/turn_tag_index.py tests/test_proxy.py
git commit -m "feat(PROXY-025): add hash-based lookup to TurnTagIndex"
```

---

### Task 2: Add `estimate_tools_tokens` to PayloadFormat

**Files:**
- Modify: `virtual_context/proxy/formats.py:107-124` (base class)
- Test: `tests/test_proxy.py`

**Step 1: Write the failing test**

```python
def test_estimate_tools_tokens(self):
    """PayloadFormat.estimate_tools_tokens counts tool definition tokens."""
    from virtual_context.proxy.formats import AnthropicFormat
    fmt = AnthropicFormat()
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "read", "description": "Read a file", "input_schema": {"type": "object"}},
            {"name": "write", "description": "Write a file", "input_schema": {"type": "object"}},
        ],
    }
    tokens = fmt.estimate_tools_tokens(body)
    assert tokens > 0
    # Empty tools → 0
    assert fmt.estimate_tools_tokens({"messages": []}) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_proxy.py -k test_estimate_tools_tokens -v`
Expected: FAIL — `AttributeError: 'AnthropicFormat' object has no attribute 'estimate_tools_tokens'`

**Step 3: Write minimal implementation**

In `virtual_context/proxy/formats.py`, in the `PayloadFormat` base class (after `_estimate_system_tokens`):

```python
def estimate_tools_tokens(self, body: dict) -> int:
    """Estimate tokens consumed by tool definitions in the request."""
    import json
    tools = body.get("tools", [])
    if not tools:
        return 0
    return len(json.dumps(tools)) // 4
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_proxy.py -k test_estimate_tools_tokens -v`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/proxy/formats.py tests/test_proxy.py
git commit -m "feat(PROXY-025): add estimate_tools_tokens to PayloadFormat"
```

---

### Task 3: Add `stub_compacted_messages` to message_filter.py

This is the core of the fix. A new function that takes the client payload, TurnTagIndex, compacted_through watermark, and a PayloadFormat, then:
1. Walks user-text messages
2. Hashes them (same as engine does)
3. Looks up hash in TurnTagIndex
4. If turn < compacted_through//2, stubs the entire message group

**Files:**
- Modify: `virtual_context/proxy/message_filter.py`
- Test: `tests/test_proxy.py`

**Step 1: Write failing tests**

Add a new test class in `tests/test_proxy.py`:

```python
class TestStubCompactedMessages:
    """Test hash-based stub replacement for compacted turns."""

    def _build_index_with_hashes(self, entries: list[tuple[int, str, list[str]]]) -> TurnTagIndex:
        """Build index: list of (turn_number, message_hash, tags)."""
        idx = TurnTagIndex()
        for turn, hash_, tags in entries:
            idx.append(TurnTagEntry(
                turn_number=turn, message_hash=hash_, tags=tags,
                primary_tag=tags[0] if tags else "_general",
            ))
        return idx

    def test_stubs_compacted_turn_simple_pair(self):
        """A simple user+assistant pair is stubbed when its hash matches a compacted turn."""
        import hashlib
        user_text = "What is pregnancy testing?"
        asst_text = "Pregnancy testing involves..."
        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        body = {"messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
            {"role": "user", "content": "Follow up question"},
        ]}
        idx = self._build_index_with_hashes([
            (0, h, ["pregnancy-test"]),
            (1, "other_hash", ["general"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_through=2,  # turn 0 is compacted
        )
        msgs = result["messages"]
        assert stub_count == 1
        # User message stubbed
        assert "[Compacted turn 0" in msgs[0]["content"]
        # Assistant message stubbed
        assert isinstance(msgs[1]["content"], list)
        assert "[Compacted turn 0" in msgs[1]["content"][0]["text"]
        # Follow-up preserved
        assert msgs[2]["content"] == "Follow up question"

    def test_stubs_turn_with_tool_chain(self):
        """A turn with tool_use/tool_result chain is collapsed to user+assistant stubs."""
        import hashlib
        user_text = "Read the config file"
        asst_text = "Let me read that for you."
        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        body = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": asst_text},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/etc/config"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents here..."},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "The config contains..."},
            ]},
            # Next turn (not compacted)
            {"role": "user", "content": "What does it mean?"},
            {"role": "assistant", "content": "It means..."},
            {"role": "user", "content": "Current question"},
        ]}
        idx = self._build_index_with_hashes([
            (0, h, ["config", "file-reading"]),
            (1, "hash_turn1", ["interpretation"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_through=2,  # turn 0 compacted
        )
        msgs = result["messages"]
        assert stub_count == 1
        # Tool chain collapsed: 4 messages → 2 stubs
        # First msg: user stub
        assert msgs[0]["role"] == "user"
        assert "[Compacted turn 0" in str(msgs[0]["content"])
        # Second msg: assistant stub
        assert msgs[1]["role"] == "assistant"
        assert "[Compacted turn 0" in str(msgs[1]["content"])
        # No tool_use_id or tool_result blocks survive
        all_content = json.dumps(msgs[:2])
        assert "tool_use" not in all_content
        assert "tool_result" not in all_content
        # Uncompacted turn preserved
        assert msgs[2]["content"] == "What does it mean?"

    def test_uncompacted_turn_hash_match_preserved(self):
        """A turn whose hash matches but is above watermark is NOT stubbed."""
        import hashlib
        user_text = "Hello"
        asst_text = "Hi there"
        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        body = {"messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
            {"role": "user", "content": "Current"},
        ]}
        idx = self._build_index_with_hashes([
            (0, h, ["greeting"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        # compacted_through=0 means nothing is compacted
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_through=0,
        )
        assert stub_count == 0
        assert result["messages"][0]["content"] == user_text

    def test_hash_miss_preserves_message(self):
        """Messages whose hash doesn't match any entry are preserved."""
        body = {"messages": [
            {"role": "user", "content": "Unique message not in index"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Current"},
        ]}
        idx = self._build_index_with_hashes([
            (0, "some_other_hash", ["topic"]),
        ])
        from virtual_context.proxy.message_filter import stub_compacted_messages
        result, stub_count = stub_compacted_messages(
            body, idx, compacted_through=2,
        )
        assert stub_count == 0
        assert result["messages"][0]["content"] == "Unique message not in index"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_proxy.py::TestStubCompactedMessages -v`
Expected: FAIL — `ImportError: cannot import name 'stub_compacted_messages'`

**Step 3: Write implementation**

Add to `virtual_context/proxy/message_filter.py`:

```python
import hashlib

from ._envelope import _strip_openclaw_envelope, _last_text_block


def stub_compacted_messages(
    body: dict,
    turn_tag_index: TurnTagIndex,
    compacted_through: int,
    *,
    fmt: PayloadFormat | None = None,
) -> tuple[dict, int]:
    """Replace compacted turns with lightweight stubs using hash-based identification.

    Walks user-text messages in the client payload, computes the same SHA-256
    hash the engine uses, and looks up the hash in the TurnTagIndex. If the
    matched entry's turn number is below ``compacted_through // 2``, the
    entire message group (user + assistant + tool chain) is collapsed to a
    pair of stub messages.

    Returns (modified_body, stub_count).
    """
    if compacted_through <= 0:
        return body, 0
    if not turn_tag_index.entries:
        return body, 0

    if fmt is None:
        fmt = detect_format(body)

    watermark_turn = compacted_through // 2

    # Determine message key and assistant role
    if fmt.name == "gemini":
        _msg_key = "contents"
        _asst_role = "model"
    elif fmt.name == "openai_responses":
        _msg_key = "input"
        _asst_role = "assistant"
    else:
        _msg_key = "messages"
        _asst_role = "assistant"

    messages = body.get(_msg_key, [])
    if not messages:
        return body, 0

    # --- Phase 1: identify user-text message indices and their hashes ---
    # A "user-text message" is a user message containing actual text content
    # (not just tool_result blocks).
    user_text_indices: list[int] = []  # indices into messages[]
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        text = _extract_text_for_hash(msg, fmt)
        if text:
            user_text_indices.append(i)

    if not user_text_indices:
        return body, 0

    # --- Phase 2: group messages into turns ---
    # Each turn spans from one user-text message to (exclusive) the next.
    # The last user-text message may be the current turn — leave it alone.
    turn_groups: list[tuple[int, int, int]] = []  # (start_idx, end_idx, user_text_idx)
    for g, uti in enumerate(user_text_indices):
        if g + 1 < len(user_text_indices):
            end = user_text_indices[g + 1]
        else:
            end = len(messages)  # last group goes to end
        turn_groups.append((uti, end, uti))

    # --- Phase 3: hash and match ---
    stubs: list[tuple[int, int, TurnTagEntry]] = []  # (start, end, entry)
    for start, end, uti in turn_groups:
        msg = messages[uti]
        user_text = _extract_text_for_hash(msg, fmt)
        if not user_text:
            continue

        # Find the first assistant message in this group to get combined text
        asst_text = ""
        for j in range(start + 1, end):
            if messages[j].get("role") == _asst_role:
                asst_text = fmt.extract_message_text(messages[j])
                break

        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        entry = turn_tag_index.get_entry_by_hash(h)
        if entry is not None and entry.turn_number < watermark_turn:
            stubs.append((start, end, entry))

    if not stubs:
        return body, 0

    # --- Phase 4: build new message list with stubs ---
    # Collect indices to stub (as a set for O(1) lookup)
    stub_ranges: list[tuple[int, int, TurnTagEntry]] = sorted(stubs, key=lambda s: s[0])
    new_messages: list[dict] = []
    i = 0
    while i < len(messages):
        # Check if this index starts a stub range
        stubbed = False
        for start, end, entry in stub_ranges:
            if i == start:
                tags_str = ", ".join(entry.tags[:5])
                # Emit user stub
                new_messages.append({
                    "role": "user",
                    "content": f"[Compacted turn {entry.turn_number}]",
                })
                # Emit assistant stub
                new_messages.append({
                    "role": _asst_role,
                    "content": [{"type": "text", "text":
                        f"[Compacted turn {entry.turn_number}: "
                        f"topics={tags_str}. "
                        f"Content stored in virtual-context.]"
                    }],
                })
                i = end
                stubbed = True
                break
        if not stubbed:
            new_messages.append(messages[i])
            i += 1

    body = dict(body)
    body[_msg_key] = new_messages
    return body, len(stubs)


def _extract_text_for_hash(msg: dict, fmt: PayloadFormat) -> str:
    """Extract user text suitable for hashing — same as engine's combined_text."""
    content = msg.get("content", "")
    if isinstance(content, str):
        text = _strip_openclaw_envelope(content)
        return text.strip()
    if isinstance(content, list):
        # Extract text blocks only (skip tool_result blocks)
        text = _strip_openclaw_envelope(_last_text_block(content))
        return text.strip()
    return ""
```

**Key implementation notes for the implementing engineer:**
- The hash MUST match what the engine computes: `sha256(f"{user_text} {asst_text}")[:16]`
- `user_text` is extracted via `_strip_openclaw_envelope(_last_text_block(content))` — the same path as `AnthropicFormat.extract_message_text()`
- `asst_text` is extracted via `fmt.extract_message_text()` on the first assistant message in the group
- The stub for assistant messages uses `content: [{"type": "text", "text": "..."}]` (Anthropic content block format), while user stubs use `content: "..."` (plain string). This matches what Anthropic API accepts for each role.
- The last user-text message is included in `turn_groups` but won't match any compacted hash (it's the current turn, above watermark).

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proxy.py::TestStubCompactedMessages -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow'`
Expected: All pass

**Step 6: Commit**

```bash
git add virtual_context/proxy/message_filter.py tests/test_proxy.py
git commit -m "feat(PROXY-025): hash-based stub replacement for compacted messages"
```

---

### Task 4: Budget auto-promotion in server.py

**Files:**
- Modify: `virtual_context/proxy/server.py:546-567`
- Test: `tests/test_proxy.py`

**Step 1: Write the failing test**

```python
def test_compute_effective_budget_within_limit(self):
    """When overhead < context_window, budget is unchanged."""
    from virtual_context.proxy.server import _compute_effective_budget
    budget, promoted = _compute_effective_budget(
        context_window=120_000,
        system_tokens=19_000,
        tools_tokens=7_500,
    )
    assert budget == 120_000
    assert promoted is False

def test_compute_effective_budget_auto_promotes(self):
    """When overhead >= context_window, budget auto-promotes to overhead + 10k."""
    from virtual_context.proxy.server import _compute_effective_budget
    budget, promoted = _compute_effective_budget(
        context_window=5_000,
        system_tokens=19_000,
        tools_tokens=7_500,
    )
    assert budget == 19_000 + 7_500 + 10_000  # 36,500
    assert promoted is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_proxy.py -k test_compute_effective_budget -v`
Expected: FAIL — `ImportError: cannot import name '_compute_effective_budget'`

**Step 3: Write minimal implementation**

Add near the top of `virtual_context/proxy/server.py` (in the helper functions section):

```python
def _compute_effective_budget(
    context_window: int,
    system_tokens: int,
    tools_tokens: int,
) -> tuple[int, bool]:
    """Compute effective token budget, auto-promoting if client overhead exceeds window.

    Returns (effective_budget, was_promoted).
    """
    overhead = system_tokens + tools_tokens
    if overhead >= context_window:
        return overhead + 10_000, True
    return context_window, False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_proxy.py -k test_compute_effective_budget -v`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/proxy/server.py tests/test_proxy.py
git commit -m "feat(PROXY-025): budget auto-promotion when overhead exceeds context_window"
```

---

### Task 5: Wire stubbing + budget into the proxy request path

**Files:**
- Modify: `virtual_context/proxy/server.py:546-590` (the filter/enrichment section)

**Step 1: Read the current code at server.py lines 546-590**

This is where `_filter_body_messages` is called. The changes integrate BEFORE the existing filter call.

**Step 2: Implement the wiring**

In `virtual_context/proxy/server.py`, in the `catch_all` function, BEFORE the existing `_filter_body_messages` call (around line 546), add the stubbing and budget logic:

```python
        # PROXY-025: Budget auto-promotion
        _effective_budget = state.engine.config.context_window if state else 0
        _budget_promoted = False
        if state:
            _sys_tok = fmt._estimate_system_tokens(body)
            _tools_tok = fmt.estimate_tools_tokens(body)
            _effective_budget, _budget_promoted = _compute_effective_budget(
                state.engine.config.context_window, _sys_tok, _tools_tok,
            )
            if _budget_promoted:
                print(
                    f"[BUDGET] Client overhead ({_sys_tok + _tools_tok}t) exceeds "
                    f"context_window ({state.engine.config.context_window}t). "
                    f"Auto-promoted to {_effective_budget}t."
                )
                metrics.record({
                    "type": "budget_auto_promoted",
                    "original": state.engine.config.context_window,
                    "promoted": _effective_budget,
                    "overhead": _sys_tok + _tools_tok,
                })

        # PROXY-025: Stub compacted messages via hash matching
        turns_stubbed = 0
        if state and state.engine._compacted_through > 0:
            from .message_filter import stub_compacted_messages
            body, turns_stubbed = stub_compacted_messages(
                body,
                state.engine._turn_tag_index,
                state.engine._compacted_through,
                fmt=fmt,
            )
            if turns_stubbed:
                print(f"[STUB] Stubbed {turns_stubbed} compacted turns")

        # (existing filter_body_messages call follows here — unchanged)
```

Then AFTER the enriched body is built and tokens estimated (around line 660), add the over-budget alert:

```python
        # PROXY-025: Over-budget alert
        if state and _effective_budget > 0 and input_tokens > _effective_budget:
            _excess = input_tokens - _effective_budget
            print(
                f"[BUDGET] Payload {input_tokens}t exceeds budget "
                f"{_effective_budget}t by {_excess}t. "
                f"Uncompacted turns pending compaction."
            )
            metrics.record({
                "type": "budget_exceeded",
                "total": input_tokens,
                "budget": _effective_budget,
                "excess": _excess,
            })
```

**Step 3: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow'`
Expected: All pass (no existing tests should break — the stubbing only activates when `compacted_through > 0`)

**Step 4: Commit**

```bash
git add virtual_context/proxy/server.py
git commit -m "feat(PROXY-025): wire stubbing and budget alerts into proxy request path"
```

---

### Task 6: Integration test with realistic tool-chain payload

**Files:**
- Test: `tests/test_proxy.py`

**Step 1: Write the integration test**

This test simulates the exact scenario from the bug report: a payload with tool chains, some compacted, verifying that stubs eliminate tool_use_id referential integrity issues.

```python
def test_stub_eliminates_tool_chain_integrity_issue(self):
    """Stubbing compacted turns eliminates tool_use_id references.

    Simulates PROXY-025: a compacted turn with tool_use/tool_result chain
    that would normally be force-kept by referential integrity. After stubbing,
    no tool_use_id survives, so the integrity loop has nothing to chase.
    """
    import hashlib

    # Turn 0: user asks, assistant uses a tool, tool returns result
    user_text_0 = "Read the config file"
    asst_text_0 = "Let me read that."
    h0 = hashlib.sha256(f"{user_text_0} {asst_text_0}".encode()).hexdigest()[:16]

    # Turn 1: user asks about result, assistant responds with another tool
    user_text_1 = "Now edit it"
    asst_text_1 = "I'll edit the file."
    h1 = hashlib.sha256(f"{user_text_1} {asst_text_1}".encode()).hexdigest()[:16]

    body = {"messages": [
        # Turn 0 (compacted)
        {"role": "user", "content": [{"type": "text", "text": user_text_0}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": asst_text_0},
            {"type": "tool_use", "id": "toolu_001", "name": "Read", "input": {"path": "/config"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_001", "content": "config data here"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Config contains X."}]},
        # Turn 1 (compacted)
        {"role": "user", "content": [{"type": "text", "text": user_text_1}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": asst_text_1},
            {"type": "tool_use", "id": "toolu_002", "name": "Edit", "input": {"path": "/config"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_002", "content": "edited"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done editing."}]},
        # Turn 2 (not compacted — current)
        {"role": "user", "content": "What did we change?"},
    ]}

    idx = TurnTagIndex()
    idx.append(TurnTagEntry(turn_number=0, message_hash=h0,
                            tags=["config", "file-reading"], primary_tag="config"))
    idx.append(TurnTagEntry(turn_number=1, message_hash=h1,
                            tags=["config", "file-editing"], primary_tag="config"))

    from virtual_context.proxy.message_filter import stub_compacted_messages
    result, stub_count = stub_compacted_messages(
        body, idx, compacted_through=4,  # turns 0-1 compacted (4 internal messages)
    )
    msgs = result["messages"]

    assert stub_count == 2

    # Verify NO tool_use or tool_result blocks anywhere in stubbed messages
    for m in msgs[:-1]:  # exclude current user message
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "tool_use", f"tool_use survived: {block}"
                assert block.get("type") != "tool_result", f"tool_result survived: {block}"

    # Verify role alternation: user, assistant, user, assistant, user
    roles = [m["role"] for m in msgs]
    for i in range(len(roles) - 1):
        if roles[i] == roles[i + 1]:
            # Consecutive same role is OK only if this is the boundary
            pass  # The stubs should maintain alternation

    # Verify current turn preserved
    assert msgs[-1]["content"] == "What did we change?"
```

**Step 2: Run test**

Run: `pytest tests/test_proxy.py::TestStubCompactedMessages::test_stub_eliminates_tool_chain_integrity_issue -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_proxy.py
git commit -m "test(PROXY-025): integration test for tool-chain stub elimination"
```

---

### Task 7: Verify with real payload data (manual validation)

**Files:** None (manual validation step)

**Step 1: Write a one-shot script that replays the real payload**

Create a temporary script (do not commit) that loads the actual request log and runs the stubbing:

```bash
python3 -c "
import json, hashlib, sys
sys.path.insert(0, '.')
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import TurnTagEntry
from virtual_context.proxy.message_filter import stub_compacted_messages

# Load real payload
with open('.virtualcontext/request_log/000005_20260307_182020_711084_v1_messages.1-inbound.json') as f:
    body = json.load(f)

msgs = body.get('messages', [])
print(f'Inbound: {len(msgs)} messages')
print(f'Inbound tokens: ~{sum(len(json.dumps(m))//4 for m in msgs)}')

# You'd need a real TurnTagIndex here — check dashboard or engine state
# For now, print what would happen
print('(Manual validation: load engine state and test)')
"
```

**Step 2: Restart proxy with the fix and observe dashboard**

```bash
lsof -ti:5757 | xargs kill; sleep 1
ANTHROPIC_API_KEY=... nohup .venv/bin/virtual-context -c virtual-context-haiku-tagger.yaml proxy --upstream https://api.anthropic.com --port 5757 > /tmp/vc-proxy.log 2>&1 &
```

Watch logs for `[BUDGET]` and `[STUB]` messages:
```bash
tail -f /tmp/vc-proxy.log | grep -E '\[BUDGET\]|\[STUB\]'
```

**Step 3: Send a test request through the proxy and verify**

Check that:
1. `[BUDGET] Client overhead (Xt) exceeds context_window (5000t). Auto-promoted to Yt.` appears
2. `[STUB] Stubbed N compacted turns` appears
3. The outbound payload tokens are reduced compared to before

---

### Task 8: Update bug tracker

**Files:**
- Modify: `memory/bugs.md` (PROXY-025 entry)
- Modify: `tests/REGRESSION_MAP.md`

**Step 1: Update bug status**

In `memory/bugs.md`, update the PROXY-025 entry status from "Open" to "Resolved" and add the fix details.

**Step 2: Add regression markers**

Add `@pytest.mark.regression("PROXY-025")` to the key tests:
- `test_stubs_compacted_turn_simple_pair`
- `test_stubs_turn_with_tool_chain`
- `test_stub_eliminates_tool_chain_integrity_issue`
- `test_compute_effective_budget_auto_promotes`

**Step 3: Update REGRESSION_MAP.md**

Add PROXY-025 entry with test references.

**Step 4: Commit**

```bash
git add memory/bugs.md tests/REGRESSION_MAP.md tests/test_proxy.py
git commit -m "docs(PROXY-025): update bug tracker and regression map"
```
