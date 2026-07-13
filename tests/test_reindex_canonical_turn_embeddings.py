"""The canonical-turn-embedding admin reindex.

An idempotent repair with a dry-run default. It backfills missing
``subject``-side chunks for physical rows whose ``reply_target_body`` lane
was never indexed, replaces stale subject chunks, and reports/deletes
chunks orphaned by a missing physical canonical row. Requester and
assistant chunks are never rewritten: their text feeds the live retrieval
branch, so touching them is not behavior-preserving.
"""

from __future__ import annotations

from types import SimpleNamespace

from virtual_context.config import VirtualContextConfig
from virtual_context.core.semantic_search import (
    SemanticSearchManager,
    chunk_turn_text,
)
from virtual_context.types import (
    CanonicalTurnChunkEmbedding,
    CanonicalTurnRow,
    StorageConfig,
    TagGeneratorConfig,
)


def _vec(text: str) -> list[float]:
    return [1.0, 0.0] if "MATCH" in text else [0.8, 0.6]


def _physical(ct_id, turn_number, *, conversation_id="c", **kw):
    base = dict(
        conversation_id=conversation_id,
        canonical_turn_id=ct_id,
        turn_number=turn_number,
        turn_group_number=turn_number // 2,
        primary_tag="chat",
        tags=["chat"],
        audience_conversation_id="c",
        audience_attribution_version=1,
    )
    base.update(kw)
    return CanonicalTurnRow(**base)


def _chunk(ct_id, turn_number, side, text, *, conversation_id="c"):
    return CanonicalTurnChunkEmbedding(
        conversation_id=conversation_id,
        side=side,
        chunk_index=0,
        text=text,
        embedding=_vec(text),
        canonical_turn_id=ct_id,
        turn_number=turn_number,
    )


def _semantic(store) -> SemanticSearchManager:
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    manager = SemanticSearchManager(store=store, config=config)
    manager._embed_fn = lambda texts: [_vec(t) for t in texts]
    return manager


class _ReindexStore:
    def __init__(self, rows, chunks, orphans):
        self._rows = list(rows)
        self.chunks = list(chunks)
        self.orphans = list(orphans)
        self.stored = []
        self.deleted = []

    def get_all_canonical_turns(self, conversation_id):
        return [r for r in self._rows if r.conversation_id == conversation_id]

    def get_all_canonical_turn_chunk_embeddings(self, conversation_id=None):
        return list(self.chunks)

    def get_orphan_canonical_turn_chunk_embeddings(self, conversation_id=None):
        return list(self.orphans)

    def store_canonical_turn_chunk_embeddings(
        self, conversation_id, turn_number, side, chunks, canonical_turn_id=None,
    ):
        self.stored.append((conversation_id, side, canonical_turn_id,
                            [c.text for c in chunks]))
        self.chunks = [
            c for c in self.chunks
            if not (
                c.conversation_id == conversation_id
                and c.canonical_turn_id == canonical_turn_id
                and c.side == side
            )
        ]
        self.chunks.extend(chunks)

    def delete_canonical_turn_chunk_embeddings(
        self, conversation_id, turn_number=None, canonical_turn_id=None,
    ):
        before = len(self.orphans)
        self.orphans = [
            c for c in self.orphans
            if not (
                c.conversation_id == conversation_id
                and c.canonical_turn_id == canonical_turn_id
            )
        ]
        self.deleted.append((conversation_id, canonical_turn_id))
        return before - len(self.orphans)


def _reindex_fixture():
    rows = [
        # Missing subject chunks.
        _physical("ct-1", 0, user_content="q1",
                  reply_target_body="alpha beta gamma"),
        # Stale subject chunks: stored text no longer matches the lane.
        _physical("ct-2", 2, user_content="q2",
                  reply_target_body="delta epsilon zeta"),
        # No reply body: not a subject candidate at all.
        _physical("ct-3", 4, user_content="q3"),
        # Up to date already.
        _physical("ct-4", 6, user_content="q4",
                  reply_target_body="eta theta iota"),
    ]
    chunks = [
        _chunk("ct-2", 2, "subject", "stale old text"),
        _chunk("ct-4", 6, "subject", "eta theta iota"),
        # Requester chunks must never be rewritten by this repair.
        _chunk("ct-1", 0, "user", "q1"),
    ]
    orphans = [_chunk("ct-gone", 9, "user", "orphaned text")]
    store = _ReindexStore(rows, chunks, orphans)
    engine_self = SimpleNamespace(_store=store, _semantic=_semantic(store))
    return store, engine_self


def _run_reindex(engine_self, conversation_id="c", **kwargs):
    from virtual_context.engine import VirtualContextEngine

    return VirtualContextEngine.reindex_canonical_turn_embeddings(
        engine_self, conversation_id, **kwargs,
    )


class TestReindexAdminOperation:
    def test_dry_run_is_the_default_and_writes_nothing(self):
        store, engine_self = _reindex_fixture()
        report = _run_reindex(engine_self)
        assert report["dry_run"] is True
        assert report["subject_missing"] == 1
        assert report["subject_stale"] == 1
        assert report["subject_ok"] == 1
        assert report["orphan_chunks"] == 1
        assert report["orphan_rows"] == 1
        assert report["subject_created"] == 0
        assert report["subject_replaced"] == 0
        assert report["orphan_deleted"] == 0
        assert store.stored == []
        assert store.deleted == []
        assert len(store.orphans) == 1

    def test_apply_repairs_missing_stale_and_orphan_chunks(self):
        store, engine_self = _reindex_fixture()
        report = _run_reindex(engine_self, dry_run=False)
        assert report["subject_created"] == 1
        assert report["subject_replaced"] == 1
        assert report["orphan_deleted"] == 1
        assert store.orphans == []
        written = {(side, ct_id): texts for _c, side, ct_id, texts in store.stored}
        assert written == {
            ("subject", "ct-1"): ["alpha beta gamma"],
            ("subject", "ct-2"): ["delta epsilon zeta"],
        }
        # The expected text is the same chunking the ingest path uses.
        assert written[("subject", "ct-1")] == chunk_turn_text("alpha beta gamma")

    def test_apply_is_idempotent(self):
        store, engine_self = _reindex_fixture()
        _run_reindex(engine_self, dry_run=False)
        writes_after_first = list(store.stored)
        report = _run_reindex(engine_self, dry_run=False)
        assert report["subject_ok"] == 3
        assert report["subject_missing"] == 0
        assert report["subject_stale"] == 0
        assert report["subject_created"] == 0
        assert report["subject_replaced"] == 0
        assert report["orphan_chunks"] == 0
        assert store.stored == writes_after_first

    def test_requester_chunks_are_never_rewritten(self):
        store, engine_self = _reindex_fixture()
        _run_reindex(engine_self, dry_run=False)
        assert all(side == "subject" for _c, side, _ct, _t in store.stored)

    def test_store_without_orphan_enumeration_fails_closed(self):
        store, engine_self = _reindex_fixture()
        # A store lacking the orphan seam cannot silently skip repair.
        bare = SimpleNamespace(
            get_all_canonical_turns=store.get_all_canonical_turns,
            get_all_canonical_turn_chunk_embeddings=(
                store.get_all_canonical_turn_chunk_embeddings
            ),
        )
        broken_self = SimpleNamespace(_store=bare, _semantic=engine_self._semantic)
        try:
            _run_reindex(broken_self)
        except RuntimeError as exc:
            assert "orphan" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for missing orphan seam")


class TestReindexCli:
    def test_cli_defaults_to_dry_run_and_apply_inverts_it(self, monkeypatch):
        from virtual_context.cli import main as cli_main

        captured = {}

        def fake_shell(args, method_name, total_keys):
            captured["dry_run"] = args.dry_run
            captured["method"] = method_name
            captured["totals"] = total_keys

        monkeypatch.setattr(cli_main, "_cmd_admin_actor_operation", fake_shell)
        cli_main.cmd_admin_reindex_canonical_turn_embeddings(
            SimpleNamespace(apply=False),
        )
        assert captured["dry_run"] is True
        assert captured["method"] == "reindex_canonical_turn_embeddings"
        assert "subject_created" in captured["totals"]

        cli_main.cmd_admin_reindex_canonical_turn_embeddings(
            SimpleNamespace(apply=True),
        )
        assert captured["dry_run"] is False

    def test_parser_registers_the_subcommand(self, monkeypatch):
        import sys

        from virtual_context.cli import main as cli_main

        seen = {}
        monkeypatch.setattr(
            cli_main,
            "cmd_admin_reindex_canonical_turn_embeddings",
            lambda args: seen.setdefault("args", args),
        )
        monkeypatch.setattr(sys, "argv", [
            "virtual-context", "admin", "reindex-canonical-turn-embeddings",
            "conv-1", "--apply", "--limit", "3",
        ])
        cli_main.main()
        args = seen["args"]
        assert args.conversation_id == "conv-1"
        assert args.apply is True
        assert args.limit == 3
