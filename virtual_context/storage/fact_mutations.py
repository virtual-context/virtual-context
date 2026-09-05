"""Transactional fact decisions and revision history shared by SQL backends."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import uuid


# Only content and provenance can invalidate source evidence. Re-ingestion and
# background maintenance update timestamps without changing what was said.
_SOURCE_VERSION_FIELDS = (
    "canonical_turn_id",
    "conversation_id",
    "origin_conversation_id",
    "turn_hash",
    "hash_version",
    "normalized_user_text",
    "normalized_assistant_text",
    "user_content",
    "assistant_content",
    "user_raw_content",
    "assistant_raw_content",
    "session_date",
    "sender",
    "sender_actor_id",
    "source_message_id",
    "origin_channel_id",
    "origin_channel_label",
    "audience_conversation_id",
    "audience_attribution_version",
    "reply_target_message_id",
    "reply_subject_actor_id",
    "reply_subject_label",
    "reply_target_body",
    "reply_attribution_version",
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _source_version(row):
    source = dict(row)
    return hashlib.sha256(
        _json({key: source.get(key) for key in _SOURCE_VERSION_FIELDS}).encode()
    ).hexdigest()


class FactMutationMixin:
    def _lock_fact_owners(
        self, conn, fact_ids, operation_id, owner_worker_id, lifecycle_epoch, site, tenant_id=None
    ):
        """Acquire lifecycle, operation, then data locks like compaction cleanup."""
        p = self._placeholder
        supplied = sum(
            value is not None for value in (operation_id, owner_worker_id, lifecycle_epoch)
        )
        if supplied not in (0, 3):
            raise ValueError("compaction guard kwargs must be all-None or all-non-None")
        identities = conn.execute(
            f"SELECT id,conversation_id FROM facts WHERE id IN ({','.join([p] * len(fact_ids))})",
            fact_ids,
        ).fetchall()
        owners = {row["conversation_id"] for row in identities}
        op = None
        if supplied and not self._compaction_fence_mode.is_off:
            op = conn.execute(
                f"SELECT conversation_id FROM compaction_operation WHERE operation_id={p}",
                (operation_id,),
            ).fetchone()
            if op:
                owners.add(op["conversation_id"])
        lock = " FOR SHARE" if self._relational_dialect == "postgres" else ""
        for owner in sorted(owners):
            conn.execute(
                f"""INSERT INTO conversation_lifecycle
                (conversation_id,generation,deleted,updated_at) VALUES ({p},0,{p},{p})
                ON CONFLICT (conversation_id) DO NOTHING""",
                (owner, False, datetime.now(timezone.utc).isoformat()),
            )
            state = conn.execute(
                f"SELECT deleted FROM conversation_lifecycle WHERE conversation_id={p}{lock}",
                (owner,),
            ).fetchone()
            if state["deleted"]:
                raise ValueError("Fact mutation refused for a deleted conversation")
        if tenant_id is not None:
            for owner in sorted(owners):
                current = conn.execute(
                    f"""SELECT 1 FROM conversations
                    WHERE conversation_id={p} AND tenant_id={p}
                    AND phase NOT IN ('deleted','merged') AND deleted_at IS NULL{lock}""",
                    (owner, tenant_id),
                ).fetchone()
                if current is None:
                    raise PermissionError("TENANT_SCOPE: fact owner changed or is not authorized")
        if supplied and not self._compaction_fence_mode.is_off:
            valid = conn.execute(
                f"""SELECT 1 FROM compaction_operation WHERE operation_id={p}
                AND owner_worker_id={p} AND lifecycle_epoch={p} AND status='running'{lock}""",
                (operation_id, owner_worker_id, lifecycle_epoch),
            ).fetchone()
            if not valid and self._compaction_fence_mode.enforces:
                self._enforce_or_observe_mismatch(operation_id=operation_id, write_site=site)
        return {row["id"]: row["conversation_id"] for row in identities}

    def _ensure_fact_decision_schema(self, conn):
        conn.execute("""CREATE TABLE IF NOT EXISTS fact_decisions (
            decision_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
            fact_id TEXT NOT NULL, replacement_fact_id TEXT NOT NULL,
            action TEXT NOT NULL, accepted INTEGER NOT NULL, reason TEXT NOT NULL,
            observed_at TEXT NOT NULL, event_date TEXT NOT NULL,
            policy_version TEXT NOT NULL, proposal_json TEXT NOT NULL,
            before_json TEXT NOT NULL, after_json TEXT NOT NULL,
            source_versions_json TEXT NOT NULL, operation_id TEXT,
            observed_fact_versions_json TEXT NOT NULL DEFAULT '{}',
            origin_conversation_id TEXT NOT NULL DEFAULT '')""")
        if self._relational_dialect == "postgres":
            conn.execute("""ALTER TABLE fact_decisions ADD COLUMN IF NOT EXISTS
                observed_fact_versions_json TEXT NOT NULL DEFAULT '{}'""")
            conn.execute("""CREATE OR REPLACE FUNCTION guard_fact_decision_content()
                RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
                IF (to_jsonb(NEW) - 'conversation_id') IS DISTINCT FROM
                   (to_jsonb(OLD) - 'conversation_id') THEN
                    RAISE EXCEPTION 'fact decision content is immutable';
                END IF;
                RETURN NEW;
                END $$""")
            if not conn.execute("""SELECT 1 FROM pg_trigger WHERE
                    tgrelid='fact_decisions'::regclass AND tgname='guard_fact_decision_content'""").fetchone():
                conn.execute("""CREATE TRIGGER guard_fact_decision_content
                    BEFORE UPDATE ON fact_decisions FOR EACH ROW
                    EXECUTE FUNCTION guard_fact_decision_content()""")
        else:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(fact_decisions)")}
            if "observed_fact_versions_json" not in columns:
                conn.execute("""ALTER TABLE fact_decisions ADD COLUMN
                    observed_fact_versions_json TEXT NOT NULL DEFAULT '{}'""")
                columns.add("observed_fact_versions_json")
            immutable = " OR ".join(
                f"NEW.{key} IS NOT OLD.{key}" for key in sorted(columns - {"conversation_id"})
            )
            trigger_sql = f"""CREATE TRIGGER guard_fact_decision_content
                BEFORE UPDATE ON fact_decisions WHEN {immutable}
                BEGIN SELECT RAISE(ABORT, 'fact decision content is immutable'); END"""
            existing = conn.execute(
                "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name='guard_fact_decision_content'"
            ).fetchone()
            # A prior guard enumerates only the columns that existed when it
            # was created. Refresh it after additive schema changes under the
            # same bootstrap transaction; unchanged startups need no trigger DDL.
            if existing is None or " ".join(existing["sql"].split()) != " ".join(trigger_sql.split()):
                if existing is not None:
                    conn.execute("DROP TRIGGER guard_fact_decision_content")
                conn.execute(trigger_sql)
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_fact_decisions_owner
            ON fact_decisions (conversation_id,observed_at,decision_id)""")

    def _fact_guard(self, conn, facts, operation_id, owner_worker_id, lifecycle_epoch, site):
        supplied = sum(
            value is not None for value in (operation_id, owner_worker_id, lifecycle_epoch)
        )
        if supplied not in (0, 3):
            raise ValueError("compaction guard kwargs must be all-None or all-non-None")
        if not supplied or self._compaction_fence_mode.is_off:
            return
        owners = {fact.conversation_id for fact in facts if fact is not None}
        valid = all(facts) and len(owners) == 1
        p = self._placeholder
        if valid:
            lock = " FOR SHARE" if self._relational_dialect == "postgres" else ""
            valid = (
                conn.execute(
                    f"""SELECT 1 FROM compaction_operation
                WHERE conversation_id={p} AND operation_id={p} AND owner_worker_id={p}
                AND lifecycle_epoch={p} AND status='running'{lock}""",
                    (next(iter(owners)), operation_id, owner_worker_id, lifecycle_epoch),
                ).fetchone()
                is not None
            )
        if not valid:
            self._enforce_or_observe_mismatch(operation_id=operation_id, write_site=site)

    def _fact_sources(self, conn, fact, *, lock=False):
        """Capture exact source versions under the mutation transaction's locks."""
        if fact is None or not fact.segment_ref:
            return ()
        p = self._placeholder
        locking = " FOR SHARE" if lock and self._relational_dialect == "postgres" else ""
        segment = conn.execute(
            f"SELECT metadata_json FROM segments WHERE ref={p} AND conversation_id={p}{locking}",
            (fact.segment_ref, fact.conversation_id),
        ).fetchone()
        if segment is None:
            return ()
        try:
            meta = json.loads(segment["metadata_json"] or "{}")
            ids = meta.get("canonical_turn_ids")
            if (
                meta.get("source_mapping_complete") is not True
                or not isinstance(ids, list)
                or not ids
                or len(ids) > 500
                or any(not isinstance(key, str) or not key for key in ids)
            ):
                return ()
        except (TypeError, ValueError, AttributeError):
            return ()
        ids = sorted(set(ids))
        rows = conn.execute(
            f"SELECT * FROM canonical_turns WHERE conversation_id={p} AND canonical_turn_id IN ({','.join([p] * len(ids))}) ORDER BY canonical_turn_id{locking}",
            [fact.conversation_id, *ids],
        ).fetchall()
        if len(rows) != len(ids):
            return ()
        from ..core.fact_lifecycle import source_author_matches

        if not source_author_matches(fact, [dict(row) for row in rows]):
            return ()
        return (
            ("segment:" + fact.segment_ref, hashlib.sha256(_json(meta).encode()).hexdigest()),
        ) + tuple((str(row["canonical_turn_id"]), _source_version(row)) for row in rows)

    def get_fact_admission_snapshot(self, fact_id, *, tenant_id=None):
        from ..core.fact_lifecycle import fact_version

        p = self._placeholder
        with self._relational_connection() as conn:
            if tenant_id is not None:
                row = conn.execute(
                    f"""SELECT f.* FROM facts f JOIN conversations c
                    ON c.conversation_id=f.conversation_id WHERE f.id={p} AND c.tenant_id={p}
                    AND c.phase NOT IN ('deleted','merged') AND c.deleted_at IS NULL""",
                    (fact_id, tenant_id),
                ).fetchone()
            else:
                row = conn.execute(f"SELECT * FROM facts WHERE id={p}", (fact_id,)).fetchone()
            if row is None:
                return None
            fact = self._row_to_fact(row)
            versions = self._fact_sources(conn, fact)
            return {
                "fact_version": fact_version(fact),
                "source_versions": versions,
                "audience": self._fact_audience(conn, fact) if versions else None,
            }

    def get_fact_admission_scope(self, fact_id, *, tenant_id=None):
        snapshot = self.get_fact_admission_snapshot(fact_id, tenant_id=tenant_id)
        return snapshot["audience"] if snapshot else None

    def _record_fact_decision(
        self,
        conn,
        *,
        proposal,
        decision,
        before,
        after,
        conversation_id,
        operation_id,
        observed_source_versions,
        observed_fact_versions,
    ):
        values = (
            uuid.uuid4().hex,
            conversation_id,
            proposal.old_fact_id,
            proposal.new_fact_id,
            proposal.action,
            int(decision.accepted),
            decision.reason,
            proposal.observed_at,
            proposal.event_date,
            decision.policy_version,
            _json(asdict(proposal)),
            _json(before),
            _json(after),
            _json(observed_source_versions),
            operation_id,
            _json(observed_fact_versions),
            conversation_id,
        )
        conn.execute(
            f"INSERT INTO fact_decisions (decision_id,conversation_id,fact_id,replacement_fact_id,action,accepted,reason,observed_at,event_date,policy_version,proposal_json,before_json,after_json,source_versions_json,operation_id,observed_fact_versions_json,origin_conversation_id) VALUES ({','.join([self._placeholder] * len(values))})",
            values,
        )

    def _set_fact_superseded(
        self,
        old_fact_id,
        new_fact_id,
        *,
        operation_id=None,
        owner_worker_id=None,
        lifecycle_epoch=None,
        tenant_id=None,
        expected_old_version=None,
        expected_new_version=None,
        expected_source_versions=None,
    ):
        from ..core.fact_lifecycle import (
            FactProposal,
            AdmissionDecision,
            decide_supersession,
            fact_version,
        )

        p = self._placeholder
        with self._relational_connection(
            write=True, scope="fact-decision:" + min(old_fact_id, new_fact_id)
        ) as conn:
            owners = self._lock_fact_owners(
                conn,
                [old_fact_id, new_fact_id],
                operation_id,
                owner_worker_id,
                lifecycle_epoch,
                "set_fact_superseded",
                tenant_id=tenant_id,
            )
            locking = " FOR UPDATE" if self._relational_dialect == "postgres" else ""
            rows = conn.execute(
                f"SELECT * FROM facts WHERE id IN ({p},{p}) ORDER BY id{locking}",
                (old_fact_id, new_fact_id),
            ).fetchall()
            facts = {row["id"]: self._row_to_fact(row) for row in rows}
            if any(owners.get(key) != fact.conversation_id for key, fact in facts.items()):
                raise ValueError("Fact ownership changed; retry the proposal with current sources")
            old, new = facts.get(old_fact_id), facts.get(new_fact_id)
            self._fact_guard(
                conn,
                (old, new),
                operation_id,
                owner_worker_id,
                lifecycle_epoch,
                "set_fact_superseded",
            )
            if old is None or new is None:
                return False
            old_versions = self._fact_sources(conn, old, lock=True)
            new_versions = self._fact_sources(conn, new, lock=True)
            versions = tuple(sorted(set(old_versions + new_versions)))
            old_audience = self._fact_audience(conn, old) if old_versions else None
            new_audience = self._fact_audience(conn, new) if new_versions else None
            decision = decide_supersession(
                new, old, new_audience=new_audience, old_audience=old_audience
            )
            if (
                (expected_old_version is not None and expected_old_version != fact_version(old))
                or (expected_new_version is not None and expected_new_version != fact_version(new))
                or (
                    expected_source_versions is not None
                    and tuple(sorted(expected_source_versions)) != versions
                )
            ):
                decision = AdmissionDecision(False, "stale_proposal")
            proposal = FactProposal(
                action="supersede",
                old_fact_id=old_fact_id,
                new_fact_id=new_fact_id,
                proposed_fields=(("superseded_by", new_fact_id),),
                observed_at=datetime.now(timezone.utc).isoformat(),
                event_date=new.when_date,
                source_versions=expected_source_versions
                if expected_source_versions is not None
                else versions,
                expected_old_version=expected_old_version or "",
                expected_new_version=expected_new_version or "",
            )
            before = asdict(old)
            after = dict(before)
            if decision.accepted:
                conn.execute(
                    f"UPDATE facts SET superseded_by={p} WHERE id={p}", (new_fact_id, old_fact_id)
                )
                after["superseded_by"] = new_fact_id
            self._record_fact_decision(
                conn,
                proposal=proposal,
                decision=decision,
                before=before,
                after=after,
                conversation_id=old.conversation_id,
                operation_id=operation_id,
                observed_source_versions=versions,
                observed_fact_versions={old.id: fact_version(old), new.id: fact_version(new)},
            )
            return decision.accepted

    def _update_fact_fields(
        self,
        fact_id,
        verb,
        object,
        status,
        what,
        *,
        operation_id=None,
        owner_worker_id=None,
        lifecycle_epoch=None,
        tenant_id=None,
    ):
        from ..core.fact_lifecycle import FactProposal, AdmissionDecision, fact_version

        p = self._placeholder
        with self._relational_connection(write=True, scope="fact-decision:" + fact_id) as conn:
            owners = self._lock_fact_owners(
                conn,
                [fact_id],
                operation_id,
                owner_worker_id,
                lifecycle_epoch,
                "update_fact_fields",
                tenant_id=tenant_id,
            )
            locking = " FOR UPDATE" if self._relational_dialect == "postgres" else ""
            row = conn.execute(f"SELECT * FROM facts WHERE id={p}{locking}", (fact_id,)).fetchone()
            fact = self._row_to_fact(row) if row is not None else None
            if fact and owners.get(fact.id) != fact.conversation_id:
                raise ValueError("Fact ownership changed; retry the proposal with current sources")
            self._fact_guard(
                conn, (fact,), operation_id, owner_worker_id, lifecycle_epoch, "update_fact_fields"
            )
            if fact is None:
                return False
            versions = self._fact_sources(conn, fact, lock=True)
            # This low-level revision API is for admitted callers. Capture its
            # provenance and preserve the old projection for audit; it does not
            # turn elapsed dates or a model proposal into completion evidence.
            proposal = FactProposal(
                action="revise",
                old_fact_id=fact_id,
                new_fact_id="",
                proposed_fields=(
                    ("verb", verb),
                    ("object", object),
                    ("status", status),
                    ("what", what),
                ),
                observed_at=datetime.now(timezone.utc).isoformat(),
                event_date=fact.when_date,
                source_versions=versions,
            )
            before = asdict(fact)
            after = {**before, **dict(proposal.proposed_fields)}
            conn.execute(
                f"UPDATE facts SET verb={p},object={p},status={p},what={p} WHERE id={p}",
                (verb, object, status, what, fact_id),
            )
            if (fact.verb, fact.object, fact.what) != (verb, object, what):
                conn.execute(f"DELETE FROM fact_embeddings WHERE fact_id={p}", (fact_id,))
            self._record_fact_decision(
                conn,
                proposal=proposal,
                decision=AdmissionDecision(
                    True, "admitted_revision" if versions else "legacy_unattributed_revision"
                ),
                before=before,
                after=after,
                conversation_id=fact.conversation_id,
                operation_id=operation_id,
                observed_source_versions=versions,
                observed_fact_versions={fact.id: fact_version(fact)},
            )
            return True

    def get_fact_decisions(self, conversation_id, *, limit=100, before=None):
        self._page_limit(limit)
        p = self._placeholder
        params = [conversation_id]
        cursor = ""
        if before is not None:
            if len(before) != 2:
                raise ValueError("Invalid fact decision cursor")
            cursor = f" AND (observed_at,decision_id)<({p},{p})"
            params.extend(before)
        with self._relational_connection() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM fact_decisions WHERE conversation_id={p}{cursor} ORDER BY observed_at DESC,decision_id DESC LIMIT {p}",
                    [*params, limit],
                ).fetchall()
            ]
