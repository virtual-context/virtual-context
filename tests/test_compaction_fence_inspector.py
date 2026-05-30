"""Structural inspector for the compaction-fence contract.

Per fencing plan §8 (Phase 6). The inspector is a static-analysis
pin -- it parses the engine source via ``ast`` + targeted string
search and asserts five contracts plus one negative-assertion group:

* PHASE_WRITER_ALLOWLIST: every function whose body contains a SQL
  string that writes ``conversations.phase`` must be in the allowlist.
* ACTIVE_OP_INSERT_ALLOWLIST: every function whose body contains an
  ``INSERT INTO compaction_operation`` SQL string must be in the
  allowlist. Inserts whose status string is ``'queued'`` or
  ``'running'`` (active rows protected by the partial unique index)
  also require the function to be in the active-insert subset.
* OP_FENCE_ALLOWLIST: every fenced storage method must call
  ``_validate_compaction_guard_kwargs`` near the top of its body so
  mixed-partial guard kwargs are rejected as programming errors.
* OP_FENCE_CALLSITE_METHODS: every call to a fenced storage method
  from a compaction-call path (compaction_pipeline.py,
  semantic_search.py, ingest/supersession.py) must pass guard kwargs
  (either explicit ``operation_id=`` etc. or a ``**guard``-style
  unpacking of the pipeline helper).
* EXCLUDED_COMPACTION_WRITES: methods listed here must NOT be called
  from ``_run_compaction`` -- a future patch that wants to write
  those tables from compaction must update schema + cleanup +
  callsite kwarg propagation + tests in the same change.

Tests also include synthetic negative scenarios to prove the inspector
actually rejects violations -- exercised via temporary source strings
parsed in-memory rather than mutating the real repo.

The inspector is intentionally pragmatic: it does not run sqlparse
or a full SQL grammar. The SQL strings it checks are simple enough
that targeted regex / substring matches suffice. Where an AST walk
cannot prove safety (e.g. C2R-gate verification on store_segment
merge vs new-ref paths), the V1 inspector defers to the existing
behavioral test suite rather than encoding the proof.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PG = _REPO_ROOT / "virtual_context" / "storage" / "postgres.py"
_SQ = _REPO_ROOT / "virtual_context" / "storage" / "sqlite.py"
_PIPELINE = _REPO_ROOT / "virtual_context" / "core" / "compaction_pipeline.py"
_SEMANTIC = _REPO_ROOT / "virtual_context" / "core" / "semantic_search.py"
_SUPERSESSION = _REPO_ROOT / "virtual_context" / "ingest" / "supersession.py"


# ---------------------------------------------------------------------------
# Allowlists per plan §8.1.
# ---------------------------------------------------------------------------


PHASE_WRITER_ALLOWLIST = frozenset({
    "begin_compaction_with_lock",
    "drain_compaction_exit",
    "set_phase",
    "set_phase_and_drain_pending_raw",
    # Admin / VCMERGE / lifecycle helpers operate outside the
    # compaction fence on phases other than 'compacting'. They each
    # carry their own conversation_lifecycle lock or are explicitly
    # admin-tier.
    "delete_conversation",
    "restore_conversation",
    "merge_conversation_data",
    "cleanup_abandoned_compaction",
    "mark_conversation_deleted",
    "increment_lifecycle_epoch_on_resurrect",
    # The schema-bootstrap path may issue defensive UPDATEs in
    # migration helpers.
    "_ensure_canonical_turn_schema",
    "_ensure_canonical_turn_views",
    "_ensure_compaction_scoping_columns",
    "_run_lifecycle_admin_actions",
})


# Functions that the plan explicitly carves out as legacy
# non-compaction callers of fenced storage methods. Per fencing plan
# §5.6 + §7.2 #4: the lazy semantic-search backfill at
# ``semantic_search.backfill_chunk_embeddings`` and the ingest-side
# utilities ``dedup_facts`` and ``_merge_facts`` reach fenced methods
# without guard kwargs because they are not part of the compaction
# pipeline. The inspector skips calls inside these enclosing functions
# so the contract holds for compaction-call paths without breaking
# legacy entry points.
_LEGACY_NON_COMPACTION_CALLERS = frozenset({
    # semantic_search.py
    "backfill_chunk_embeddings",
    "embed_and_store_turn",
    # ingest/supersession.py
    "dedup_facts",
    "_merge_facts",
})


ACTIVE_OP_INSERT_ALLOWLIST = frozenset({
    "begin_compaction_with_lock",
    "cleanup_abandoned_compaction",
    # Legacy/test helper. Inserts at status='queued' but is reached
    # only by tests per repo-wide grep at the time of this commit. The
    # inspector tolerates its presence here so the pin matches the
    # current codebase; if a future patch wires it back into a
    # production path, that patch must either convert it to the locked
    # primitive or remove it from this allowlist.
    "start_compaction_operation",
    # Schema migration that recreates the table on legacy SQLite
    # installs. Inserts go into ``compaction_operation_new`` (the
    # rename target); the inspector matches the literal substring
    # ``INSERT INTO compaction_operation`` which catches the legacy
    # rebuild path too.
    "_ensure_schema",
})


# Fenced storage methods that the per-write fence shipped in commit
# 14cf5e6 covers. Each must call ``_validate_compaction_guard_kwargs``
# at the top of its body so mixed-partial guard kwargs are rejected
# as programming errors per fencing plan §5.7 T3.19.
OP_FENCE_ALLOWLIST = frozenset({
    "store_facts",
    "replace_facts_for_segment",
    "set_fact_superseded",
    "update_fact_fields",
    "store_fact_links",
    "store_chunk_embeddings",
    "link_segment_tool_output",
})


# Compaction-call paths that must pass guard kwargs into these store
# methods. The inspector walks each module and confirms every call to
# the listed methods carries either an explicit ``operation_id=`` kwarg
# or a ``**self._compaction_guard_kwargs(...)`` unpack.
OP_FENCE_CALLSITE_METHODS = frozenset({
    "store_segment", "update_segment", "save_tag_summary",
    "store_tag_summary_embedding", "mark_canonical_turns_compacted",
    "store_facts", "replace_facts_for_segment",
    "set_fact_superseded", "update_fact_fields",
    "store_fact_links", "store_chunk_embeddings",
    "link_segment_tool_output",
})


# Methods that the compaction pipeline must NOT touch. If a future
# patch starts writing either table from ``_run_compaction``, that
# patch must update schema, cleanup predicates, callsite kwarg
# propagation, and tests in the same change.
EXCLUDED_COMPACTION_WRITES = frozenset({
    "link_turn_tool_output",
    "store_chain_snapshot",
})


# ---------------------------------------------------------------------------
# AST helpers.
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _all_string_constants(node: ast.AST) -> list[str]:
    """Return every ``ast.Constant`` ``str`` value reachable from
    ``node``. ``ast.walk`` traverses every child so multi-line SQL
    strings inside ``conn.execute(...)`` calls are visited.
    """
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def _walk_functions(tree: ast.Module):
    """Yield ``(qualname, FunctionDef)`` pairs for every function and
    method defined directly under the module or nested inside a
    class. We only descend ONE level into classes so test helpers can
    reason about ``ClassName.method`` -> ``method`` mapping cleanly.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield sub.name, sub


# Captures any SQL fragment that writes ``conversations.phase``. The
# pattern is anchored on the ``conversations`` table token so a
# ``SET phase = X`` write against an unrelated table does not trip
# the rule. Both UPDATE bulk-rewrite forms and ON CONFLICT DO UPDATE
# SET forms are covered; the latter relies on the surrounding INSERT
# referencing ``conversations`` as the target table.
_PHASE_WRITE_RE = re.compile(
    r"""
    (?:
        # Bulk-rewrite form: ``UPDATE conversations SET ... phase = X``
        UPDATE\s+conversations\b
        (?:(?!\bWHERE\b)[\s\S])*?
        \bSET\b
        (?:(?!\bWHERE\b)[\s\S])*?
        \bphase\s*=\s*
        (?:%s|\?|'\w+')
        |
        # UPSERT form: ``INSERT INTO conversations ... ON CONFLICT ...
        # DO UPDATE SET ... phase = X``. The ``conversations`` token
        # appears before the SET clause.
        INSERT\s+INTO\s+conversations\b
        [\s\S]*?
        \bON\s+CONFLICT\b
        [\s\S]*?
        \bDO\s+UPDATE\s+SET\b
        (?:(?!\bWHERE\b)[\s\S])*?
        \bphase\s*=\s*
        (?:%s|\?|'\w+')
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Matches the literal ``'queued'`` or ``'running'`` status values
# inside the VALUES tuple of a ``compaction_operation`` INSERT.
# Active-row inserts are the ones protected by the partial unique
# index ``idx_compaction_operation_active``; the inspector enforces a
# tighter allowlist on those than on completed/abandoned inserts.
_ACTIVE_STATUS_RE = re.compile(
    r"VALUES\s*\([^)]*'(?:queued|running)'", re.IGNORECASE,
)


def _function_writes_phase(fn: ast.FunctionDef) -> bool:
    for s in _all_string_constants(fn):
        if _PHASE_WRITE_RE.search(s):
            return True
    return False


def _function_inserts_compaction_op(fn: ast.FunctionDef) -> bool:
    for s in _all_string_constants(fn):
        if "INSERT INTO compaction_operation" in s:
            return True
    return False


def _function_inserts_active_compaction_op(fn: ast.FunctionDef) -> bool:
    """An ``INSERT INTO compaction_operation`` whose VALUES list
    contains a ``'queued'`` or ``'running'`` literal. Catches the
    protected-by-partial-unique-index active-row inserts.
    """
    for s in _all_string_constants(fn):
        if "INSERT INTO compaction_operation" in s and _ACTIVE_STATUS_RE.search(s):
            return True
    return False


# Names whose presence in the first prefix of a function body counts
# as "the validator was invoked before any DB-touching work."
# ``_validate_compaction_guard_kwargs`` is the only module-level
# guard validator; the position check rejects functions that hide
# the call after an INSERT or UPDATE.
_VALIDATOR_NAME = "_validate_compaction_guard_kwargs"


def _function_has_validator_call(fn: ast.FunctionDef) -> bool:
    """Return True iff the validator is called BEFORE any
    DB-touching statement in the function body.

    The first prefix of ``fn.body`` is scanned linearly: a top-level
    import statement, a docstring, and any assignments / Expr lines
    that do not include a ``conn.execute`` / ``cur.execute`` /
    ``conn.transaction`` call are skipped. The first statement that
    either (a) contains the validator call -> True, or (b) contains
    a DB-touching call -> False. This rejects a function that places
    the validator at the bottom of its body after the writes have
    already fired.
    """
    db_call_attrs = frozenset({"execute", "executemany", "transaction"})

    for stmt in fn.body:
        calls = [
            child for child in ast.walk(stmt)
            if isinstance(child, ast.Call)
        ]
        calls.sort(
            key=lambda call: (
                getattr(call, "lineno", 0),
                getattr(call, "col_offset", 0),
            )
        )
        for call in calls:
            func = call.func
            if isinstance(func, ast.Name) and func.id == _VALIDATOR_NAME:
                return True
            if isinstance(func, ast.Attribute) and func.attr in db_call_attrs:
                return False
    return False


def _call_target_name(call: ast.Call) -> str | None:
    """Return the simple method/function name being invoked, or
    ``None`` when the call shape doesn't surface one.

    Handles:
      * ``foo(...)`` -> ``"foo"``
      * ``self._store.foo(...)`` -> ``"foo"``
      * ``self._semantic.foo(...)`` -> ``"foo"``
      * ``self.store.foo(...)`` -> ``"foo"``
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _call_passes_guard_kwargs(call: ast.Call) -> bool:
    """True if the call carries either an explicit ``operation_id=``
    kwarg OR a ``**self._compaction_guard_kwargs(...)`` (or any
    ``**name``) unpack.
    """
    for kw in call.keywords:
        if kw.arg is None:
            # ``**something`` star-kwargs unpack -- treat as a guard
            # delegation. The pipeline helper is the canonical source;
            # we don't AST-prove it expands to operation_id, but the
            # tests in test_compaction_caller_wiring.py already pin
            # that contract.
            return True
        if kw.arg == "operation_id":
            return True
    return False


# ---------------------------------------------------------------------------
# T6.1: PHASE_WRITER_ALLOWLIST -- every phase-writing function in
# postgres.py / sqlite.py is in the allowlist.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_path", [_PG, _SQ], ids=["postgres", "sqlite"])
def test_phase_writer_allowlist(backend_path):
    tree = _parse(backend_path)
    offenders: list[str] = []
    for qualname, fn in _walk_functions(tree):
        if _function_writes_phase(fn) and qualname not in PHASE_WRITER_ALLOWLIST:
            offenders.append(qualname)
    assert not offenders, (
        f"{backend_path.name}: functions writing conversations.phase "
        f"outside the allowlist: {sorted(offenders)}. "
        f"Either add the function to PHASE_WRITER_ALLOWLIST with a "
        f"justification comment OR refactor the write through one of: "
        f"{sorted(PHASE_WRITER_ALLOWLIST)}."
    )


# ---------------------------------------------------------------------------
# T6.2: ACTIVE_OP_INSERT_ALLOWLIST -- every function that inserts a
# row into compaction_operation is in the allowlist, AND every
# function that inserts an ACTIVE-status row (status='queued' or
# 'running') is in the allowlist's active-insert subset.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_path", [_PG, _SQ], ids=["postgres", "sqlite"])
def test_compaction_operation_insert_allowlist(backend_path):
    tree = _parse(backend_path)
    offenders: list[str] = []
    for qualname, fn in _walk_functions(tree):
        if _function_inserts_compaction_op(fn) and qualname not in ACTIVE_OP_INSERT_ALLOWLIST:
            offenders.append(qualname)
    assert not offenders, (
        f"{backend_path.name}: functions inserting into "
        f"compaction_operation outside the allowlist: "
        f"{sorted(offenders)}. Active-row inserts must route through "
        f"begin_compaction_with_lock or cleanup_abandoned_compaction "
        f"(legacy start_compaction_operation is tolerated as long as "
        f"it stays out of production paths)."
    )


# ---------------------------------------------------------------------------
# T6.3: OP_FENCE_ALLOWLIST -- every fenced storage method calls the
# guard-kwargs validator near the top of its body.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_path", [_PG, _SQ], ids=["postgres", "sqlite"])
def test_op_fence_validator_present(backend_path):
    tree = _parse(backend_path)
    seen: dict[str, ast.FunctionDef] = {}
    for qualname, fn in _walk_functions(tree):
        if qualname in OP_FENCE_ALLOWLIST:
            seen[qualname] = fn
    missing_definition = OP_FENCE_ALLOWLIST - set(seen.keys())
    assert not missing_definition, (
        f"{backend_path.name}: OP_FENCE_ALLOWLIST methods not found "
        f"in this backend: {sorted(missing_definition)}. Both Postgres "
        f"and SQLite must define each fenced method so callers can "
        f"use them backend-agnostically."
    )
    missing_validator: list[str] = []
    for qualname, fn in seen.items():
        if not _function_has_validator_call(fn):
            missing_validator.append(qualname)
    assert not missing_validator, (
        f"{backend_path.name}: OP_FENCE_ALLOWLIST methods missing the "
        f"_validate_compaction_guard_kwargs(...) call: "
        f"{sorted(missing_validator)}. The validator rejects mixed "
        f"partial guard kwargs as programming errors per fencing "
        f"plan §5.7 T3.19."
    )


# ---------------------------------------------------------------------------
# T6.4: OP_FENCE_CALLSITE_METHODS -- every call to a fenced storage
# method from a compaction-call path carries guard kwargs.
# ---------------------------------------------------------------------------


_CALLSITE_PATHS = (_PIPELINE, _SEMANTIC, _SUPERSESSION)


def _iter_calls_to(tree: ast.Module, names: frozenset[str]):
    for child in ast.walk(tree):
        if isinstance(child, ast.Call):
            target = _call_target_name(child)
            if target in names:
                yield child, target


def _flat(tree: ast.Module, names: frozenset[str], skip_in: frozenset[str]):
    """Flat walker: visit every function in the tree, then every Call
    inside that function, yielding the (call, target, enclosing_fn)
    triple when target matches.
    """
    out: list[tuple[ast.Call, str, str | None]] = []

    def _visit(fn_name: str | None, body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _visit(node.name, node.body)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _visit(sub.name, sub.body)
            else:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        target = _call_target_name(child)
                        if target in names and (fn_name or "") not in skip_in:
                            out.append((child, target, fn_name))

    _visit(None, tree.body)
    return out


@pytest.mark.parametrize("path", _CALLSITE_PATHS,
                         ids=lambda p: p.relative_to(_REPO_ROOT).as_posix())
def test_compaction_callsite_guard_kwargs(path):
    tree = _parse(path)
    offenders: list[str] = []
    for call, target, enclosing in _flat(
        tree, OP_FENCE_CALLSITE_METHODS, _LEGACY_NON_COMPACTION_CALLERS,
    ):
        if not _call_passes_guard_kwargs(call):
            offenders.append(
                f"{target} in {enclosing!r} (line {call.lineno})"
            )
    assert not offenders, (
        f"{path.relative_to(_REPO_ROOT)}: calls to fenced storage "
        f"methods without operation_id / guard kwargs: "
        f"{offenders}. Use ``**self._compaction_guard_kwargs(...)`` "
        f"to forward the all-or-nothing tuple, or explicit "
        f"``operation_id=`` etc. kwargs."
    )


# ---------------------------------------------------------------------------
# T6.5: EXCLUDED_COMPACTION_WRITES -- _run_compaction AND every
# helper method it can reach do not call methods that the fence does
# not yet cover. The recursive closure catches a stealth write that
# hides inside ``_compact_and_store`` / ``_build_tag_summaries`` /
# ``_propagate_tool_output_links`` etc.
# ---------------------------------------------------------------------------


def _collect_self_method_names_called(fn: ast.FunctionDef) -> set[str]:
    """Return the set of ``self.<name>`` method names invoked anywhere
    inside ``fn``. Used to build the call closure rooted at
    ``_run_compaction``.
    """
    out: set[str] = set()
    for child in ast.walk(fn):
        if isinstance(child, ast.Call):
            func = child.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"):
                out.add(func.attr)
    return out


def _build_self_method_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Map every method defined in ``tree`` (at one class level deep)
    to its ``FunctionDef`` node, keyed by bare method name.
    """
    out: dict[str, ast.FunctionDef] = {}
    for qualname, fn in _walk_functions(tree):
        out[qualname] = fn
    return out


def _closure_from(root_name: str, method_map: dict[str, ast.FunctionDef]) -> set[str]:
    """Compute the transitive set of self-method names reachable from
    ``root_name`` inside ``method_map``. External calls (anything not
    a ``self.<name>`` invocation, or names not defined in ``tree``)
    are ignored.
    """
    if root_name not in method_map:
        return set()
    seen: set[str] = {root_name}
    frontier = [root_name]
    while frontier:
        current = frontier.pop()
        for callee in _collect_self_method_names_called(method_map[current]):
            if callee in method_map and callee not in seen:
                seen.add(callee)
                frontier.append(callee)
    return seen


def test_excluded_compaction_writes_not_in_run_compaction_closure():
    tree = _parse(_PIPELINE)
    method_map = _build_self_method_map(tree)
    assert "_run_compaction" in method_map, (
        "_run_compaction not found in compaction_pipeline.py -- the "
        "inspector cannot verify the excluded-writes contract."
    )
    closure = _closure_from("_run_compaction", method_map)
    assert closure, "closure walk produced an empty set"

    offenders: list[str] = []
    for name in sorted(closure):
        fn = method_map[name]
        for call, target in _iter_calls_to(
            ast.Module(body=[fn], type_ignores=[]),
            EXCLUDED_COMPACTION_WRITES,
        ):
            offenders.append(
                f"{target} in {name} (line {call.lineno})"
            )
    assert not offenders, (
        f"_run_compaction call closure reaches excluded methods: "
        f"{offenders}. If a new compaction path needs to write these "
        f"tables, update schema, cleanup predicates, callsite kwarg "
        f"propagation, and tests in the same change."
    )


# ---------------------------------------------------------------------------
# T6.5b: ACTIVE_OP_INSERT subset -- any function that inserts an
# active-status (queued / running) compaction_operation row must be
# in the strict allowlist below. This is tighter than the broader
# any-INSERT check because active inserts collide with the partial
# unique index ``idx_compaction_operation_active`` and so directly
# affect the at-most-one-active invariant.
# ---------------------------------------------------------------------------


ACTIVE_STATUS_INSERT_ALLOWLIST = frozenset({
    "begin_compaction_with_lock",
    "cleanup_abandoned_compaction",
    # Legacy/test helper -- still tolerated, see commentary on
    # ACTIVE_OP_INSERT_ALLOWLIST above.
    "start_compaction_operation",
})


@pytest.mark.parametrize("backend_path", [_PG, _SQ], ids=["postgres", "sqlite"])
def test_active_status_compaction_insert_allowlist(backend_path):
    tree = _parse(backend_path)
    offenders: list[str] = []
    for qualname, fn in _walk_functions(tree):
        if (_function_inserts_active_compaction_op(fn)
                and qualname not in ACTIVE_STATUS_INSERT_ALLOWLIST):
            offenders.append(qualname)
    assert not offenders, (
        f"{backend_path.name}: functions inserting active-status "
        f"compaction_operation rows (queued / running) outside the "
        f"strict allowlist: {sorted(offenders)}. Active rows collide "
        f"with idx_compaction_operation_active and so the inspector "
        f"requires them to route through one of: "
        f"{sorted(ACTIVE_STATUS_INSERT_ALLOWLIST)}."
    )


# ---------------------------------------------------------------------------
# T6.6: NEGATIVE assertion -- a synthetic phase write in a function
# outside the allowlist makes the inspector reject the tree. Confirms
# the inspector actually rejects violations rather than passing
# vacuously.
# ---------------------------------------------------------------------------


def test_phase_writer_inspector_rejects_synthetic_violation():
    """Parse an in-memory module containing a synthetic phase-writing
    function whose name is NOT in the allowlist. The same function
    body shape that legitimate writers use ought to trip the rule.
    """
    synthetic = """
class FakeStore:
    def malicious_phase_write(self, conn, conv_id):
        conn.execute(
            "UPDATE conversations SET phase = 'compacting' WHERE conversation_id = %s",
            (conv_id,),
        )
"""
    tree = ast.parse(synthetic)
    offenders: list[str] = []
    for qualname, fn in _walk_functions(tree):
        if _function_writes_phase(fn) and qualname not in PHASE_WRITER_ALLOWLIST:
            offenders.append(qualname)
    assert offenders == ["malicious_phase_write"], (
        "Negative test failed: the inspector should have flagged "
        f"`malicious_phase_write` but reported {offenders}. The rule "
        "is silently dormant if this fires."
    )


def test_phase_writer_inspector_ignores_where_only_phase_predicate():
    """A conversation UPDATE that only reads phase in the predicate
    must not count as a phase write.
    """
    synthetic = """
class FakeStore:
    def touch_conversation_if_compacting(self, conn, conv_id):
        conn.execute(
            "UPDATE conversations SET updated_at = %s "
            "WHERE conversation_id = %s AND phase = %s",
            ("now", conv_id, "compacting"),
        )
"""
    tree = ast.parse(synthetic)
    offenders: list[str] = []
    for qualname, fn in _walk_functions(tree):
        if _function_writes_phase(fn) and qualname not in PHASE_WRITER_ALLOWLIST:
            offenders.append(qualname)
    assert offenders == [], (
        "Negative test failed: the inspector treated a WHERE-only "
        f"phase predicate as a conversations.phase write: {offenders}."
    )


def test_compaction_operation_insert_inspector_rejects_synthetic_violation():
    """Symmetric negative for the compaction_operation insert rule."""
    synthetic = """
class FakeStore:
    def malicious_active_insert(self, conn, op_id, conv_id):
        conn.execute(
            \"\"\"INSERT INTO compaction_operation
               (operation_id, conversation_id, lifecycle_epoch,
                phase_index, phase_count, phase_name, status,
                started_at, heartbeat_ts, owner_worker_id, created_at)
               VALUES (%s, %s, 1, 0, 7, 'starting', 'running',
                       %s, %s, %s, %s)\"\"\",
            (op_id, conv_id, "now", "now", "w", "now"),
        )
"""
    tree = ast.parse(synthetic)
    offenders: list[str] = []
    for qualname, fn in _walk_functions(tree):
        if _function_inserts_compaction_op(fn) and qualname not in ACTIVE_OP_INSERT_ALLOWLIST:
            offenders.append(qualname)
    assert offenders == ["malicious_active_insert"], (
        "Negative test failed: the inspector should have flagged "
        f"`malicious_active_insert` but reported {offenders}."
    )


def test_op_fence_inspector_rejects_synthetic_violation():
    """Negative for the op-fence allowlist: a function named like a
    fenced method but lacking the validator call must trip the rule.
    """
    synthetic = """
class FakeStore:
    def store_facts(self, facts, *, operation_id=None,
                    owner_worker_id=None, lifecycle_epoch=None):
        # Missing the _validate_compaction_guard_kwargs call.
        for f in facts:
            self._insert(f)
"""
    tree = ast.parse(synthetic)
    found_store_facts: list[ast.FunctionDef] = []
    for qualname, fn in _walk_functions(tree):
        if qualname == "store_facts":
            found_store_facts.append(fn)
    assert len(found_store_facts) == 1
    assert not _function_has_validator_call(found_store_facts[0]), (
        "Negative test failed: the synthetic store_facts shouldn't "
        "appear to call the validator. The detector is too permissive."
    )


def test_op_fence_inspector_rejects_validator_after_write():
    """Negative for validator position: a fenced method with a
    DB-touching call before the validator must fail even when both
    calls live inside the same top-level block.
    """
    synthetic = """
class FakeStore:
    def store_facts(self, conn, facts, *, operation_id=None,
                    owner_worker_id=None, lifecycle_epoch=None):
        with conn.transaction():
            conn.execute("INSERT INTO facts (id) VALUES (%s)", ("f1",))
            _validate_compaction_guard_kwargs(
                operation_id, owner_worker_id, lifecycle_epoch,
            )
"""
    tree = ast.parse(synthetic)
    found_store_facts: list[ast.FunctionDef] = []
    for qualname, fn in _walk_functions(tree):
        if qualname == "store_facts":
            found_store_facts.append(fn)
    assert len(found_store_facts) == 1
    assert not _function_has_validator_call(found_store_facts[0]), (
        "Negative test failed: a validator hidden after a DB write "
        "must not satisfy the op-fence position rule."
    )


def test_callsite_inspector_rejects_synthetic_violation():
    """Negative for the callsite rule: a synthetic compaction-call
    path that invokes ``store_chunk_embeddings`` without
    ``operation_id`` or ``**guard`` must trip.
    """
    synthetic = """
class FakeStub:
    def _bad_caller(self, stored):
        self._semantic.embed_and_store_chunks(stored)
        self._store.store_chunk_embeddings(stored.ref, [])
"""
    tree = ast.parse(synthetic)
    offenders: list[str] = []
    for call, target in _iter_calls_to(tree, OP_FENCE_CALLSITE_METHODS):
        if not _call_passes_guard_kwargs(call):
            offenders.append(target)
    assert offenders == ["store_chunk_embeddings"], (
        "Negative test failed: the synthetic call site should be "
        f"flagged but the inspector reported {offenders}."
    )
