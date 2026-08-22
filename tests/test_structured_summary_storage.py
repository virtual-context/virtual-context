from __future__ import annotations

import sqlite3

from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    STRUCTURED_SUMMARY_MAX_TEXT_CHARS,
    SegmentMetadata,
    StoredSegment,
    StructuredSummary,
    SummaryClaim,
    SummarySource,
    TagSummary,
    strict_structured_summary,
    structured_summary_to_dict,
)


def _bigtex_summary() -> StructuredSummary:
    return StructuredSummary(
        schema_version=1,
        claims=(
            SummaryClaim(
                text="I stopped tesamorelin after experiencing side effects.",
                claim_type="personal",
                temporal_status="ceased",
                modality="asserted",
                event_time="",
                sources=(
                    SummarySource(
                        canonical_turn_id="ct-bigtex-1",
                        source_role="requester",
                        speaker_label="BigTex",
                        evidence_excerpt=(
                            "I stopped tesamorelin after experiencing side effects."
                        ),
                        session_date="2026-08-18",
                        source_provenance_digest="b" * 64,
                    ),
                ),
            ),
        ),
        source_digest="a" * 64,
        generation_model="test-summary-model",
    )


def _segment(structured: StructuredSummary) -> StoredSegment:
    return StoredSegment(
        ref="seg-bigtex",
        conversation_id="conv-bigtex",
        primary_tag="medical",
        tags=["medical"],
        summary="BigTex discussed tesamorelin.",
        summary_tokens=6,
        full_text="BigTex: I stopped tesamorelin after experiencing side effects.",
        full_tokens=12,
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct-bigtex-1"],
            source_mapping_complete=True,
            structured_summary=structured,
        ),
    )


def _tag_summary(structured: StructuredSummary) -> TagSummary:
    return TagSummary(
        tag="medical",
        summary="Tesamorelin history.",
        source_segment_refs=["seg-bigtex"],
        source_turn_numbers=[7],
        source_canonical_turn_ids=["ct-bigtex-1"],
        covers_through_turn=7,
        covers_through_canonical_turn_id="ct-bigtex-1",
        structured_summary=structured,
    )


def test_structured_summary_codec_is_strict_and_fail_empty():
    structured = _bigtex_summary()
    encoded = structured_summary_to_dict(structured)

    assert strict_structured_summary(encoded) == structured
    assert (
        strict_structured_summary({"schema_version": True, "claims": []})
        == StructuredSummary()
    )
    assert strict_structured_summary(
        dict(encoded, source_digest="A" * 64)
    ) == StructuredSummary()
    assert strict_structured_summary(
        dict(encoded, source_digest="")
    ) == StructuredSummary()

    invalid_role = encoded.copy()
    invalid_role["claims"] = [dict(encoded["claims"][0])]
    invalid_role["claims"][0]["sources"] = [
        dict(encoded["claims"][0]["sources"][0], source_role="subject")
    ]
    assert strict_structured_summary(invalid_role) == StructuredSummary()

    oversized = encoded.copy()
    oversized["claims"] = [
        dict(encoded["claims"][0], text="x" * (STRUCTURED_SUMMARY_MAX_TEXT_CHARS + 1))
    ]
    assert strict_structured_summary(oversized) == StructuredSummary()

    too_many_claims = dict(encoded, claims=encoded["claims"] * 257)
    assert strict_structured_summary(too_many_claims) == StructuredSummary()

    too_many_sources = encoded.copy()
    too_many_sources["claims"] = [dict(encoded["claims"][0])]
    too_many_sources["claims"][0]["sources"] = (
        encoded["claims"][0]["sources"] * 2
    )
    assert strict_structured_summary(too_many_sources) == StructuredSummary()

    assistant_personal = StructuredSummary(
        schema_version=1,
        claims=(
            SummaryClaim(
                text="BigTex currently uses tesamorelin.",
                claim_type="personal",
                temporal_status="active",
                modality="asserted",
                sources=(
                    SummarySource(
                        canonical_turn_id="ct-assistant-1",
                        source_role="assistant",
                        speaker_label="Assistant",
                        evidence_excerpt="BigTex currently uses tesamorelin.",
                    ),
                ),
            ),
        ),
        source_digest="b" * 64,
        generation_model="test-summary-model",
    )
    assert structured_summary_to_dict(assistant_personal) == {
        "schema_version": 0,
        "claims": [],
        "source_digest": "",
        "generation_model": "",
    }

    assistant_timed_world = structured_summary_to_dict(StructuredSummary(
        schema_version=1,
        claims=(SummaryClaim(
            text="The deployment is currently healthy.",
            claim_type="world",
            temporal_status="active",
            modality="asserted",
            sources=(SummarySource(
                canonical_turn_id="ct-assistant-2",
                source_role="assistant",
                speaker_label="Assistant",
                evidence_excerpt="The deployment is currently healthy.",
            ),),
        ),),
        source_digest="c" * 64,
    ))
    assert assistant_timed_world["schema_version"] == 0

    mixed_roles = encoded.copy()
    mixed_roles["claims"] = [dict(encoded["claims"][0])]
    mixed_roles["claims"][0]["sources"] = [
        encoded["claims"][0]["sources"][0],
        dict(
            encoded["claims"][0]["sources"][0],
            canonical_turn_id="ct-assistant-2",
            source_role="assistant",
            speaker_label="Assistant",
        ),
    ]
    assert strict_structured_summary(mixed_roles) == StructuredSummary()

    generated_paraphrase = encoded.copy()
    generated_paraphrase["claims"] = [
        dict(
            encoded["claims"][0],
            text="BigTex still uses tesamorelin.",
        )
    ]
    assert strict_structured_summary(generated_paraphrase) == StructuredSummary()


def test_sqlite_round_trips_segment_and_tag_structured_summaries(tmp_path):
    structured = _bigtex_summary()
    store = SQLiteStore(db_path=tmp_path / "structured.db")
    try:
        store.store_segment(_segment(structured))
        loaded_segment = store.get_segment("seg-bigtex", "conv-bigtex")
        loaded_summary = store.get_summary("seg-bigtex", "conv-bigtex")

        assert loaded_segment is not None
        assert loaded_summary is not None
        assert loaded_segment.metadata.structured_summary == structured
        assert loaded_summary.metadata.structured_summary == structured

        store.save_tag_summary(_tag_summary(structured), "conv-bigtex")
        loaded_tag = store.get_tag_summary("medical", "conv-bigtex")
        assert loaded_tag is not None
        assert loaded_tag.structured_summary == structured
        assert loaded_tag.source_canonical_turn_ids == ["ct-bigtex-1"]
        assert loaded_tag.covers_through_canonical_turn_id == "ct-bigtex-1"
    finally:
        store.close()


def test_sqlite_migrates_legacy_tag_summary_to_v0_envelope(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE tag_summaries (
            tag TEXT NOT NULL,
            conversation_id TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            code_refs TEXT NOT NULL DEFAULT '[]',
            summary_tokens INTEGER NOT NULL DEFAULT 0,
            source_segment_refs TEXT NOT NULL DEFAULT '[]',
            source_turn_numbers TEXT NOT NULL DEFAULT '[]',
            source_canonical_turn_ids TEXT NOT NULL DEFAULT '[]',
            covers_through_turn INTEGER NOT NULL DEFAULT -1,
            covers_through_canonical_turn_id TEXT NOT NULL DEFAULT '',
            generated_by_turn_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (tag, conversation_id)
        );
        INSERT INTO tag_summaries
            (tag, conversation_id, summary, created_at, updated_at)
        VALUES
            ('medical', 'conv-bigtex', 'Legacy prose',
             '2026-08-18T00:00:00+00:00', '2026-08-18T00:00:00+00:00');
    """)
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=path)
    try:
        columns = {
            row[1] for row in store._get_conn().execute(
                "PRAGMA table_info(tag_summaries)"
            ).fetchall()
        }
        loaded = store.get_tag_summary("medical", "conv-bigtex")
        assert "structured_summary_json" in columns
        assert loaded is not None
        assert loaded.structured_summary == StructuredSummary()
    finally:
        store.close()


def test_filesystem_round_trips_structured_and_tag_canonical_provenance(tmp_path):
    structured = _bigtex_summary()
    store = FilesystemStore(tmp_path / "store")

    store.store_segment(_segment(structured))
    loaded_segment = store.get_segment(
        "seg-bigtex", conversation_id="conv-bigtex",
    )
    loaded_summary = store.get_summary(
        "seg-bigtex", conversation_id="conv-bigtex",
    )
    assert loaded_segment is not None
    assert loaded_summary is not None
    assert loaded_segment.metadata.structured_summary == structured
    assert loaded_summary.metadata.structured_summary == structured

    store.save_tag_summary(_tag_summary(structured), "conv-bigtex")
    loaded_tag = store.get_tag_summary("medical", "conv-bigtex")
    assert loaded_tag is not None
    assert loaded_tag.structured_summary == structured
    assert loaded_tag.source_canonical_turn_ids == ["ct-bigtex-1"]
    assert loaded_tag.covers_through_canonical_turn_id == "ct-bigtex-1"
