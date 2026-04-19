"""FalkorDBFactStore: graph-native backend for facts and fact links via FalkorDB."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import falkordb as fdb

from ..types import Fact, FactLink, LinkedFact
from .helpers import dt_to_str as _dt_to_str, str_to_dt as _str_to_dt

logger = logging.getLogger(__name__)


class FalkorDBFactStore:
    """FalkorDB backend implementing FactStore + FactLinkStore protocols.

    FalkorDB is a Redis-based graph database that speaks Cypher.
    Facts are ``(:Fact)`` nodes, links are ``[:FACT_LINK]`` edges,
    tags are ``(:Tag)`` nodes connected via ``[:HAS_TAG]``.

    This backend does NOT implement SegmentStore, StateStore, or SearchStore —
    those fall back to SQLite or Postgres via CompositeStore.
    """

    def __init__(
        self, host: str = "localhost", port: int = 6379,
        password: str = "", graph_name: str = "vc_facts",
    ) -> None:
        self._client = fdb.FalkorDB(host=host, port=port, password=password or None)
        self._graph = self._client.select_graph(graph_name)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._graph.create_node_unique_constraint("Fact", "id")
        except Exception:
            pass  # Already exists
        try:
            self._graph.create_node_unique_constraint("Tag", "name")
        except Exception:
            pass
        for prop in ("subject", "verb", "status", "fact_type", "segment_ref"):
            try:
                self._graph.create_node_range_index("Fact", prop)
            except Exception:
                pass

    def _query(self, cypher: str, params: dict | None = None) -> list[list]:
        result = self._graph.query(cypher, params=params or {})
        return result.result_set

    def _fact_to_props(self, fact: Fact) -> dict:
        return {
            "id": fact.id,
            "subject": fact.subject,
            "verb": fact.verb,
            "object": fact.object,
            "status": fact.status,
            "what": fact.what,
            "who": fact.who,
            "when_date": fact.when_date,
            "where_val": fact.where,
            "why": fact.why,
            "fact_type": fact.fact_type,
            "tags_json": json.dumps(fact.tags),
            "segment_ref": fact.segment_ref,
            "conversation_id": fact.conversation_id,
            "turn_numbers_json": json.dumps(fact.turn_numbers),
            "mentioned_at": _dt_to_str(fact.mentioned_at),
            "session_date": fact.session_date,
            "superseded_by": fact.superseded_by or "",
        }

    def _node_to_fact(self, node) -> Fact:
        return Fact.from_dict(node.properties, dt_parser=_str_to_dt)

    # ------------------------------------------------------------------
    # FactStore
    # ------------------------------------------------------------------

    def store_facts(
        self,
        facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        if not facts:
            return 0
        count = 0
        for fact in facts:
            props = self._fact_to_props(fact)
            self._query(
                "MERGE (f:Fact {id: $id}) SET f += $props",
                {"id": fact.id, "props": props},
            )
            # Manage tags
            self._query("MATCH (f:Fact {id: $id})-[r:HAS_TAG]->() DELETE r", {"id": fact.id})
            for tag in fact.tags:
                self._query(
                    "MERGE (t:Tag {name: $tag}) WITH t "
                    "MATCH (f:Fact {id: $id}) MERGE (f)-[:HAS_TAG]->(t)",
                    {"id": fact.id, "tag": tag},
                )
            count += 1
        return count

    def query_facts(
        self, *, subject: str | None = None, verb: str | None = None,
        verbs: list[str] | None = None, object_contains: str | None = None,
        status: str | None = None, fact_type: str | None = None,
        tags: list[str] | None = None, limit: int = 50,
    ) -> list[Fact]:
        conditions: list[str] = ["(f.superseded_by IS NULL OR f.superseded_by = '')"]
        params: dict = {"lim": limit}

        if subject:
            conditions.append("f.subject = $subject")
            params["subject"] = subject
        if verbs is not None:
            verb_conds = []
            for i, v in enumerate(verbs):
                key = f"v{i}"
                verb_conds.append(f"toLower(f.verb) CONTAINS toLower(${key})")
                params[key] = v
            conditions.append("(" + " OR ".join(verb_conds) + ")")
        elif verb is not None:
            conditions.append("toLower(f.verb) CONTAINS toLower($verb)")
            params["verb"] = verb
        if object_contains:
            conditions.append("toLower(f.object) CONTAINS toLower($obj)")
            params["obj"] = object_contains
        if status:
            conditions.append("f.status = $status")
            params["status"] = status
        if fact_type:
            conditions.append("f.fact_type = $fact_type")
            params["fact_type"] = fact_type

        where = " AND ".join(conditions)

        if tags:
            params["tags"] = tags
            cypher = (
                f"MATCH (f:Fact)-[:HAS_TAG]->(t:Tag) "
                f"WHERE t.name IN $tags AND {where} "
                f"WITH DISTINCT f "
                f"ORDER BY f.mentioned_at DESC LIMIT $lim "
                f"RETURN f"
            )
        else:
            cypher = (
                f"MATCH (f:Fact) WHERE {where} "
                f"RETURN f ORDER BY f.mentioned_at DESC LIMIT $lim"
            )

        rows = self._query(cypher, params)
        return [self._node_to_fact(row[0]) for row in rows]

    def get_unique_fact_verbs(self) -> list[str]:
        rows = self._query(
            "MATCH (f:Fact) WHERE f.verb <> '' AND (f.superseded_by IS NULL OR f.superseded_by = '') "
            "RETURN DISTINCT f.verb AS verb"
        )
        return [row[0] for row in rows]

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        rows = self._query(
            "MATCH (f:Fact {segment_ref: $ref}) RETURN f ORDER BY f.mentioned_at",
            {"ref": segment_ref},
        )
        return [self._node_to_fact(row[0]) for row in rows]

    def replace_facts_for_segment(
        self,
        conversation_id: str,
        segment_ref: str,
        facts: list,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> tuple[int, int]:
        rows = self._query(
            "MATCH (f:Fact {conversation_id: $conv_id, segment_ref: $seg_ref}) DETACH DELETE f RETURN count(f) as deleted",
            {"conv_id": conversation_id, "seg_ref": segment_ref},
        )
        deleted = rows[0][0] if rows and rows[0] else 0
        inserted = self.store_facts(facts) if facts else 0
        return deleted, inserted

    def search_facts(self, query: str, limit: int = 10) -> list[Fact]:
        terms = query.lower().split()
        if not terms:
            return []
        conds = []
        params: dict = {"lim": limit}
        for i, term in enumerate(terms):
            key = f"t{i}"
            params[key] = term
            conds.append(
                f"(toLower(f.subject) CONTAINS ${key} OR toLower(f.verb) CONTAINS ${key} "
                f"OR toLower(f.object) CONTAINS ${key} OR toLower(f.what) CONTAINS ${key})"
            )
        where = " OR ".join(conds)
        rows = self._query(
            f"MATCH (f:Fact) WHERE (f.superseded_by IS NULL OR f.superseded_by = '') "
            f"AND ({where}) RETURN f ORDER BY f.mentioned_at DESC LIMIT $lim",
            params,
        )
        return [self._node_to_fact(row[0]) for row in rows]

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        self._query(
            "MATCH (f:Fact {id: $id}) SET f.superseded_by = $new_id",
            {"id": old_fact_id, "new_id": new_fact_id},
        )

    def update_fact_fields(self, fact_id: str, verb: str, object: str, status: str, what: str) -> None:
        self._query(
            "MATCH (f:Fact {id: $id}) SET f.verb = $verb, f.object = $object, "
            "f.status = $status, f.what = $what",
            {"id": fact_id, "verb": verb, "object": object, "status": status, "what": what},
        )

    def get_fact_count_by_tags(self) -> dict[str, int]:
        rows = self._query(
            "MATCH (f:Fact)-[:HAS_TAG]->(t:Tag) RETURN t.name AS tag, count(f) AS cnt"
        )
        return {row[0]: row[1] for row in rows}

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        if not fact_ids:
            return []
        rows = self._query(
            "MATCH (f:Fact) WHERE f.superseded_by IN $ids "
            "RETURN f.superseded_by AS superseded_by, f.subject AS subject, "
            "f.verb AS verb, f.object AS object",
            {"ids": fact_ids},
        )
        return [{"superseded_by": r[0], "subject": r[1], "verb": r[2], "object": r[3]} for r in rows]

    # ------------------------------------------------------------------
    # FactLinkStore — native graph edges
    # ------------------------------------------------------------------

    def store_fact_links(self, links: list[FactLink]) -> int:
        if not links:
            return 0
        count = 0
        for link in links:
            self._query(
                "MATCH (src:Fact {id: $src_id}), (tgt:Fact {id: $tgt_id}) "
                "MERGE (src)-[r:FACT_LINK {id: $link_id}]->(tgt) "
                "SET r.relation_type = $rel_type, r.confidence = $confidence, "
                "r.context = $context, r.created_at = $created_at, r.created_by = $created_by",
                {
                    "src_id": link.source_fact_id, "tgt_id": link.target_fact_id,
                    "link_id": link.id, "rel_type": link.relation_type,
                    "confidence": link.confidence, "context": link.context,
                    "created_at": _dt_to_str(link.created_at), "created_by": link.created_by,
                },
            )
            count += 1
        return count

    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]:
        if direction == "outgoing":
            cypher = (
                "MATCH (src:Fact {id: $id})-[r:FACT_LINK]->(tgt:Fact) "
                "RETURN r.id AS id, src.id AS src, tgt.id AS tgt, "
                "r.relation_type AS rt, r.confidence AS conf, "
                "r.context AS ctx, r.created_at AS ca, r.created_by AS cb"
            )
        elif direction == "incoming":
            cypher = (
                "MATCH (src:Fact)-[r:FACT_LINK]->(tgt:Fact {id: $id}) "
                "RETURN r.id AS id, src.id AS src, tgt.id AS tgt, "
                "r.relation_type AS rt, r.confidence AS conf, "
                "r.context AS ctx, r.created_at AS ca, r.created_by AS cb"
            )
        else:
            cypher = (
                "MATCH (a:Fact {id: $id})-[r:FACT_LINK]-(b:Fact) "
                "RETURN r.id AS id, startNode(r).id AS src, endNode(r).id AS tgt, "
                "r.relation_type AS rt, r.confidence AS conf, "
                "r.context AS ctx, r.created_at AS ca, r.created_by AS cb"
            )

        rows = self._query(cypher, {"id": fact_id})
        return [
            FactLink(
                id=r[0], source_fact_id=r[1], target_fact_id=r[2],
                relation_type=r[3], confidence=r[4], context=r[5] or "",
                created_at=_str_to_dt(r[6]) if r[6] else datetime.now(timezone.utc),
                created_by=r[7] or "compaction",
            )
            for r in rows
        ]

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]:
        if not fact_ids:
            return []
        # FalkorDB's variable-length path returns Path objects that don't
        # support property access like r[0].relation_type. Use manual BFS
        # with single-hop queries instead (same approach as SQLiteStore).
        visited = set(fact_ids)
        result: list[LinkedFact] = []
        current_layer = set(fact_ids)

        for _hop in range(max(1, min(depth, 5))):
            if not current_layer:
                break
            layer_list = list(current_layer)
            rows = self._query(
                "MATCH (seed:Fact)-[r:FACT_LINK]-(linked:Fact) "
                "WHERE seed.id IN $layer "
                "AND (linked.superseded_by IS NULL OR linked.superseded_by = '') "
                "AND NOT linked.id IN $visited "
                "RETURN linked, seed.id AS from_id, "
                "r.relation_type AS rt, r.confidence AS conf, r.context AS ctx",
                {"layer": layer_list, "visited": list(visited)},
            )
            next_layer: set[str] = set()
            for r in rows:
                fact = self._node_to_fact(r[0])
                if fact.id in visited:
                    continue
                result.append(LinkedFact(
                    fact=fact, linked_from_fact_id=r[1],
                    relation_type=r[2], confidence=r[3],
                    link_context=r[4] or "",
                ))
                visited.add(fact.id)
                next_layer.add(fact.id)
            current_layer = next_layer

        return result

    def delete_fact_links(self, fact_id: str) -> int:
        rows = self._query(
            "MATCH (a:Fact {id: $id})-[r:FACT_LINK]-() DELETE r RETURN count(r) AS cnt",
            {"id": fact_id},
        )
        return rows[0][0] if rows else 0

    def delete_conversation(self, conversation_id: str) -> int:
        rows = self._query(
            "MATCH (f:Fact {conversation_id: $conversation_id}) RETURN count(f) AS cnt",
            {"conversation_id": conversation_id},
        )
        deleted = int(rows[0][0]) if rows else 0
        if deleted:
            self._query(
                "MATCH (f:Fact {conversation_id: $conversation_id}) DETACH DELETE f",
                {"conversation_id": conversation_id},
            )
            self._query(
                "MATCH (t:Tag) WHERE NOT (t)<-[:HAS_TAG]-(:Fact) DELETE t",
            )
        return deleted

    def migrate_supersession_to_links(self) -> int:
        rows = self._query(
            "MATCH (old:Fact) WHERE old.superseded_by IS NOT NULL AND old.superseded_by <> '' "
            "MATCH (new:Fact {id: old.superseded_by}) "
            "WHERE NOT (new)-[:FACT_LINK {relation_type: 'supersedes'}]->(old) "
            "MERGE (new)-[r:FACT_LINK]->(old) "
            "SET r.relation_type = 'supersedes', r.confidence = 1.0, "
            "r.context = 'Migrated from superseded_by property', "
            "r.created_by = 'migration', r.id = old.id + '_migration' "
            "RETURN count(r) AS cnt"
        )
        return rows[0][0] if rows else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._client:
            self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
