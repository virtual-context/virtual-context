"""Load the attestation records the verifier compares against.

``verify_candidates`` needs a second, independently written record for every
candidate. That record lives in the attestation ledger the trusted adapter
writes at ingest, and until now nothing outside a test could produce one — so
the verifier would have received an empty mapping in production and rejected
every candidate as ``no_attested_source``. The pipeline would have been
correct and silent.

This module is deliberately the only place that turns ledger rows into
``MessageSourceRecord``. Reassembling the mapping in a caller would put the
two-source invariant outside the package that owns it, and let it drift from
the storage guards the first time either side changes.

Nothing here repairs a disagreement. A row missing from the ledger simply
does not appear in the mapping, and the verifier rejects the candidate that
needed it — which is the intended outcome, because an unattested row is
exactly what must never be quoted.
"""

from __future__ import annotations

from typing import Mapping

from .verify import MessageSourceRecord


def load_message_sources(
    store,
    *,
    tenant_id: str,
    canonical_turn_ids,
) -> Mapping[str, MessageSourceRecord]:
    """Attestation records for these turns, keyed by canonical turn id.

    Bounded to the ids asked for and scoped to one tenant, so this cannot
    become a table scan as the ledger grows and cannot reach another
    tenant's rows. Turn ids with no ledger row are absent from the result
    rather than represented by a blank record: a missing attestation and an
    empty one mean different things, and only the first is honest.
    """
    ids = [str(i) for i in (canonical_turn_ids or []) if str(i)]
    if not ids:
        return {}

    reader = getattr(store, "list_attested_message_sources", None)
    if not callable(reader):
        raise RuntimeError(
            "this store cannot list attested message sources; the engagement "
            "pipeline requires the attestation ledger, and proceeding without "
            "it would mean quoting rows whose authorship was never confirmed "
            "by a second record"
        )

    records: dict[str, MessageSourceRecord] = {}
    for row in reader(tenant_id=tenant_id, canonical_turn_ids=ids) or []:
        turn_id = str(row.get("canonical_turn_id") or "")
        if not turn_id:
            continue
        records[turn_id] = MessageSourceRecord(
            canonical_turn_id=turn_id,
            message_id=str(row.get("message_id") or ""),
            channel_id=str(row.get("channel_id") or ""),
            guild_id=str(row.get("guild_id") or ""),
            author_id=str(row.get("author_id") or ""),
            source_actor_id=str(row.get("source_actor_id") or ""),
        )
    return records
