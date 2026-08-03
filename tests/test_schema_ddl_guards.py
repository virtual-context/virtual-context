"""Every CREATE in a storage backend must be safe to run twice.

The failure this prevents: a bootstrap statement without ``IF NOT EXISTS``
and without a preceding ``DROP ... IF EXISTS`` aborts on the second boot
against an existing database, and takes the rest of its guarded block with
it. Ten such statements shipped at once — nine triggers, a view, and a
migration scratch table with no guard of any kind under a comment claiming
it was idempotent.

Finding those took a database, eight threads and forty trials. This finds
them by reading the source, in milliseconds, with no database at all. That
trade is the whole point: cheap-and-total beats thorough-and-slow for a
property every statement must have.

Strings are collected through the AST rather than by scanning lines, because
a line scan cannot tell SQL from prose. The first version of this sweep
reported ``# ... the CREATE TABLE above`` as a defect — 8 of its 23 hits were
comments.

The AST removes comments but NOT docstrings, which is where the second
version still went wrong: "this CREATE TABLE is a no-op" inside a docstring
produced ``TABLE is``. Docstrings are therefore excluded explicitly. Prose
that merely mentions DDL is the dominant false positive here, and a lint
whose output has to be triaged is a lint people stop reading.

Known strictness, stated rather than discovered later: a DROP whose object
name is interpolated — ``f"DROP TRIGGER IF EXISTS {name}"`` inside a loop —
cannot be credited to a specific object, so a CREATE relying on it is
flagged even though it is guarded at runtime. Both backends are clean today,
so this costs nothing now; if it ever fires that way, the fix is to add
IF NOT EXISTS rather than to loosen the rule, because a runtime-only guard
is exactly the thing that is hard to verify by reading.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

BACKENDS = ["virtual_context/storage/sqlite.py",
            "virtual_context/storage/postgres.py"]

# CREATE <kind> [IF NOT EXISTS] <name>
_CREATE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:UNIQUE\s+|TEMP\s+|TEMPORARY\s+|VIRTUAL\s+)?"
    r"(TABLE|INDEX|VIEW|TRIGGER)\s+(?:CONCURRENTLY\s+)?"
    r"(IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w]*)",
    re.I,
)
_DROP = re.compile(
    r"\bDROP\s+(TABLE|INDEX|VIEW|TRIGGER)\s+IF\s+EXISTS\s+([A-Za-z_][\w]*)",
    re.I,
)


def _sql_strings(path: str) -> list[str]:
    """Every non-docstring string constant in the module.

    Comments never reach the AST. Docstrings do, and they are the ones that
    talk *about* DDL rather than being DDL, so they are dropped by identity
    against the nodes ast.get_docstring would return.
    """
    tree = ast.parse(pathlib.Path(path).read_text())

    def _joined(node: ast.JoinedStr) -> str:
        """Reconstruct an f-string, standing a name in for each hole.

        The AST splits an f-string into its literal parts, so
        f"CREATE INDEX IF NOT EXISTS {name} ON t" yields a constant that
        stops at "EXISTS " — and a regex expecting a name then reads the
        guard itself as the object, reporting ``INDEX IF``. The placeholder
        keeps the statement parseable without inventing a real name.
        """
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("interpolated_name")
        return "".join(out)

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    out = []
    inside_fstring = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                inside_fstring.add(id(part))
            text = _joined(node)
            if "CREATE" in text.upper() or "DROP" in text.upper():
                out.append(text)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings
                and id(node) not in inside_fstring):
            if "CREATE" in node.value.upper() or "DROP" in node.value.upper():
                out.append(node.value)
    return out


_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments from inside a SQL string.

    Python comments never reach the AST and docstrings are excluded, but a
    ``--`` comment lives INSIDE the SQL string and is indistinguishable from
    DDL to a regex. The shipped schema carries several, including "on a
    pre-existing facts table this CREATE TABLE is a no-op", which the sweep
    reported as an unguarded ``TABLE is``.
    """
    return _SQL_BLOCK_COMMENT.sub(" ", _SQL_LINE_COMMENT.sub(" ", sql))


def _unguarded(path: str) -> list[str]:
    """Objects created without IF NOT EXISTS and without a DROP IF EXISTS.

    Each string is scanned on its own. An earlier version joined them into
    one blob, which invented statements across the seams: a string ending
    "CREATE UNIQUE INDEX" followed by one starting "BEGIN IMMEDIATE" was
    read as ``CREATE UNIQUE INDEX BEGIN``. The DROP set is still module-wide,
    because a drop and its create are legitimately separate statements.
    """
    strings = [_strip_sql_comments(s) for s in _sql_strings(path)]
    dropped = set()
    for sql in strings:
        dropped |= {name.lower() for _kind, name in _DROP.findall(sql)}
    bad = []
    for sql in strings:
        for kind, if_not_exists, name in _CREATE.findall(sql):
            if if_not_exists or name.lower() in dropped:
                continue
            if re.search(
                rf"CREATE\s+OR\s+REPLACE\s+(?:\w+\s+)?{re.escape(name)}\b",
                sql, re.I,
            ):
                continue
            bad.append(f"{kind.upper()} {name}")
    return sorted(set(bad))


class TestNoUnguardedCreate:
    @pytest.mark.parametrize("path", BACKENDS)
    def test_every_create_is_guarded(self, path):
        """Either IF NOT EXISTS, or a DROP IF EXISTS for the same object.

        A DROP counts because drop-then-create is a deliberate pattern here:
        it replaces a definition that changed. What it is NOT is protection
        against a concurrent bootstrapper, which is why both backends also
        serialize their bootstrap — a different defect with a different fix.
        """
        unguarded = _unguarded(path)
        assert unguarded == [], (
            f"{path} has CREATE statements that are neither IF NOT EXISTS "
            f"nor preceded by a DROP IF EXISTS: {unguarded}"
        )


class TestTheLintCanActuallyFail:
    """A lint that cannot fail is a lint that proves nothing."""

    def _scan(self, source: str, tmp_path) -> list[str]:
        path = tmp_path / "probe.py"
        path.write_text(source)
        return _unguarded(str(path))

    def test_it_flags_an_unguarded_create(self, tmp_path):
        found = self._scan('q = """CREATE TABLE widgets (id INT)"""', tmp_path)
        assert found == ["TABLE widgets"]

    def test_it_accepts_if_not_exists(self, tmp_path):
        assert self._scan(
            'q = """CREATE TABLE IF NOT EXISTS widgets (id INT)"""', tmp_path,
        ) == []

    def test_it_accepts_a_preceding_drop(self, tmp_path):
        assert self._scan(
            'a = "DROP TRIGGER IF EXISTS trg_x ON t"\n'
            'b = "CREATE TRIGGER trg_x BEFORE UPDATE ON t"',
            tmp_path,
        ) == []

    def test_it_accepts_create_or_replace(self, tmp_path):
        assert self._scan(
            'q = "CREATE OR REPLACE VIEW v AS SELECT 1"', tmp_path,
        ) == []

    def test_a_comment_is_not_a_hit(self, tmp_path):
        """The false positive that made the first sweep unusable."""
        assert self._scan(
            "# the CREATE TABLE above must agree with the migration\n"
            'q = "SELECT 1"',
            tmp_path,
        ) == []

    def test_it_flags_the_scratch_table_shape_that_shipped(self, tmp_path):
        """conversations_ed4_new: no guard, no DROP, comment claiming idempotent."""
        assert self._scan(
            '# Idempotent re-runnable (skips if already migrated).\n'
            'q = """CREATE TABLE conversations_ed4_new (id TEXT PRIMARY KEY)"""',
            tmp_path,
        ) == ["TABLE conversations_ed4_new"]

    def test_it_flags_an_fstring_create_without_a_guard(self, tmp_path):
        """f-strings must not slip through the reconstruction."""
        assert self._scan(
            'name = "x"\nq = f"CREATE INDEX {name} ON t(a)"', tmp_path,
        ) == ["INDEX interpolated_name"]

    def test_an_fstring_with_a_guard_is_accepted(self, tmp_path):
        assert self._scan(
            'name = "x"\nq = f"CREATE INDEX IF NOT EXISTS {name} ON t(a)"',
            tmp_path,
        ) == []

    def test_prose_inside_a_sql_comment_is_not_a_hit(self, tmp_path):
        """Shipped text: 'this CREATE TABLE is a no-op'."""
        assert self._scan(
            'q = """CREATE TABLE IF NOT EXISTS t (\n'
            '  -- on a pre-existing facts table this CREATE TABLE is a no-op\n'
            '  id INT)"""',
            tmp_path,
        ) == []

    def test_a_docstring_mentioning_ddl_is_not_a_hit(self, tmp_path):
        assert self._scan(
            'def f():\n    """Runs the CREATE TABLE widgets migration."""\n'
            '    return 1',
            tmp_path,
        ) == []

    def test_the_seam_does_not_produce_a_phantom_object(self, tmp_path):
        found = self._scan(
            'a = "CREATE UNIQUE INDEX IF NOT EXISTS ix ON t(a)"\n'
            'b = "BEGIN IMMEDIATE;"',
            tmp_path,
        )
        assert "INDEX BEGIN" not in found
