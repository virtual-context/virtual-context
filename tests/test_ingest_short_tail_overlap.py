"""A payload whose overlap with storage is shorter than any anchor window (BUG-136).

A reused host thread can carry only the last answered exchange plus the new
message as native items. Those three rows overlap storage by two, below the
smallest anchor window, and the short-overlap search only ran for payloads of
fewer than three rows, so the stored exchange was appended a second time.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore


def _reconciler(store):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _fmt():
    from virtual_context.proxy.formats import detect_format
    return detect_format({"messages": []})


def _user_texts(store):
    return [r[0] for r in store._get_conn().execute(
        "SELECT normalized_user_text FROM canonical_turns WHERE conversation_id='c' "
        "AND normalized_user_text != '' ORDER BY sort_key"
    ).fetchall()]


@pytest.mark.regression("BUG-136")
def test_last_exchange_plus_new_message_is_not_appended_twice(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    rec = _reconciler(store)
    seed = []
    for i in range(6):
        seed += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    rec.ingest_batch(conversation_id="c", body={"messages": seed}, fmt=_fmt(), expected_lifecycle_epoch=1)
    assert _user_texts(store).count("question 5") == 1

    rec.ingest_batch(
        conversation_id="c",
        body={"messages": [
            {"role": "user", "content": "question 5"},
            {"role": "assistant", "content": "answer 5"},
            {"role": "user", "content": "finish the check"},
            {"role": "assistant", "content": "done"},
        ]},
        fmt=_fmt(), expected_lifecycle_epoch=1,
    )
    texts = _user_texts(store)
    assert texts.count("question 5") == 1
    assert texts[-1] == "finish the check"
