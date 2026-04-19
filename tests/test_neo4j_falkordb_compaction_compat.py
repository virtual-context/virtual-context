"""Regression: Neo4jFactStore and FalkorDBFactStore must accept
compaction-guard kwargs (operation_id, owner_worker_id, lifecycle_epoch)
on store_facts and replace_facts_for_segment.

CompositeStore forwards these kwargs unconditionally to the underlying
fact store. Neo4j and FalkorDB backends don't have compaction_operation
tables — guards are a no-op — but the signatures must accept the kwargs
or the pipeline TypeErrors on the call site.

We verify via AST parsing rather than inspect.signature so the test
runs without the optional neo4j/falkordb driver libraries installed.
The signature contract is the regression boundary: if the kwargs aren't
declared, the production pipeline fails before any network traffic.
"""
from __future__ import annotations

import ast
from pathlib import Path


_GUARD_KWARGS = ("operation_id", "owner_worker_id", "lifecycle_epoch")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NEO4J_PATH = _REPO_ROOT / "virtual_context" / "storage" / "neo4j.py"
_FALKORDB_PATH = _REPO_ROOT / "virtual_context" / "storage" / "falkordb.py"


def _find_method(module_path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    return sub
    raise AssertionError(f"{class_name}.{method_name} not found in {module_path}")


def _assert_accepts_guard_kwargs(method: ast.FunctionDef, label: str) -> None:
    kwonly_names = [a.arg for a in method.args.kwonlyargs]
    kwonly_defaults = method.args.kw_defaults

    for kwarg in _GUARD_KWARGS:
        assert kwarg in kwonly_names, (
            f"{label} missing keyword-only {kwarg!r} — CompositeStore forwards "
            f"this, would TypeError at the call site. Declared kwonlys: {kwonly_names}"
        )
        idx = kwonly_names.index(kwarg)
        default = kwonly_defaults[idx]
        # None default parses as ast.Constant(value=None)
        assert isinstance(default, ast.Constant) and default.value is None, (
            f"{label} {kwarg!r} must default to None for accept-and-ignore "
            f"(got {ast.unparse(default) if default else 'no default'})"
        )


# ---------------------------------------------------------------------------
# Neo4jFactStore
# ---------------------------------------------------------------------------

def test_neo4j_store_facts_accepts_guard_kwargs():
    method = _find_method(_NEO4J_PATH, "Neo4jFactStore", "store_facts")
    _assert_accepts_guard_kwargs(method, "Neo4jFactStore.store_facts")


def test_neo4j_replace_facts_for_segment_accepts_guard_kwargs():
    method = _find_method(_NEO4J_PATH, "Neo4jFactStore", "replace_facts_for_segment")
    _assert_accepts_guard_kwargs(method, "Neo4jFactStore.replace_facts_for_segment")


# ---------------------------------------------------------------------------
# FalkorDBFactStore
# ---------------------------------------------------------------------------

def test_falkordb_store_facts_accepts_guard_kwargs():
    method = _find_method(_FALKORDB_PATH, "FalkorDBFactStore", "store_facts")
    _assert_accepts_guard_kwargs(method, "FalkorDBFactStore.store_facts")


def test_falkordb_replace_facts_for_segment_accepts_guard_kwargs():
    method = _find_method(_FALKORDB_PATH, "FalkorDBFactStore", "replace_facts_for_segment")
    _assert_accepts_guard_kwargs(method, "FalkorDBFactStore.replace_facts_for_segment")
