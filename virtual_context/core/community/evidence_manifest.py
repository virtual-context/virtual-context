"""Deterministic evidence fingerprints without an aggregate JSON allocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


def evidence_digest(
    metadata: Mapping[str, Any],
    *,
    records: Mapping[str, Iterable[dict]],
) -> str:
    """Hash the same canonical JSON as materialized lists, one record at a time.

    This is an input manifest, never a last-seen timestamp shortcut. Callers
    enumerate current exact sources (including older carryover citations), so a
    correction to an old row changes the digest even when its id is unchanged.
    """
    if metadata.keys() & records.keys():
        raise ValueError("manifest metadata and record sections must be distinct")
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))

    def write(value) -> None:
        for part in encoder.iterencode(value):
            digest.update(part.encode("utf-8"))

    digest.update(b"{")
    for index, key in enumerate(sorted(set(metadata) | set(records))):
        if index:
            digest.update(b",")
        write(key)
        digest.update(b":")
        if key in records:
            digest.update(b"[")
            for position, record in enumerate(records[key]):
                if position:
                    digest.update(b",")
                write(record)
            digest.update(b"]")
        else:
            write(metadata[key])
    digest.update(b"}")
    return digest.hexdigest()
