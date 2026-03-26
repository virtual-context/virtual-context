"""Neo4jFactStore: graph-native backend for facts and fact links via Neo4j."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from neo4j import GraphDatabase

from ..types import Fact, FactLink, LinkedFact
from .helpers import dt_to_str as _dt_to_str, str_to_dt as _str_to_dt

logger = logging.getLogger(__name__)


class Neo4jFactStore:
    """Neo4j backend implementing FactStore + FactLinkStore protocols.

    Facts are stored as ``(:Fact)`` nodes with all fields as properties.
    Fact links are stored as typed relationships between Fact nodes.
    Tags are stored as ``(:Tag)`` nodes connected via ``[:HAS_TAG]`` edges.

    This backend does NOT implement SegmentStore, StateStore, or SearchStore —
    those fall back to SQLite via CompositeStore.
    """

    def __init__(self, uri: str, auth: tuple[str, str] = ("neo4j", "")) -> None:
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._driver.session() as session:
            # Unique constraint on Fact.id
            session.run(
                "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.id IS UNIQUE"
            )
            # Indexes for common query patterns
            session.run("CREATE INDEX fact_subject IF NOT EXISTS FOR (f:Fact) ON (f.subject)")
            session.run("CREATE INDEX fact_verb IF NOT EXISTS FOR (f:Fact) ON (f.verb)")
            session.run("CREATE INDEX fact_status IF NOT EXISTS FOR (f:Fact) ON (f.status)")
            session.run("CREATE INDEX fact_type IF NOT EXISTS FOR (f:Fact) ON (f.fact_type)")
            session.run("CREATE INDEX fact_segment IF NOT EXISTS FOR (f:Fact) ON (f.segment_ref)")
            # Tag node constraint
            session.run(
                "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS "
                "FOR (t:Tag) REQUIRE t.name IS UNIQUE"
            )

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
            "where_val": fact.where,  # "where" is reserved in some contexts
            "why": fact.why,
            "fact_type": fact.fact_type,
            "tags_json": json.dumps(fact.tags),
            "segment_ref": fact.segment_ref,
            "conversation_id": fact.conversation_id,
            "turn_numbers_json": json.dumps(fact.turn_numbers),
            "mentioned_at": _dt_to_str(fact.mentioned_at),
            "session_date": fact.session_date,
            "superseded_by": fact.superseded_by,
        }

    def _record_to_fact(self, record) -> Fact:
        props = dict(record) if not hasattr(record, "data") else record.data()
        return Fact.from_dict(props, dt_parser=_str_to_dt)

    # ------------------------------------------------------------------
    # FactStore
    # ------------------------------------------------------------------

    def store_facts(self, facts: list[Fact]) -> int:
        if not facts:
            return 0
        with self._driver.session() as session:
            count = 0
            for fact in facts:
                props = self._fact_to_props(fact)
                session.run(
                    """MERGE (f:Fact {id: $id})
                    SET f += $props""",
                    id=fact.id, props=props,
                )
                # Manage tag relationships
                session.run(
                    "MATCH (f:Fact {id: $id})-[r:HAS_TAG]->() DELETE r",
                    id=fact.id,
                )
                for tag in fact.tags:
                    session.run(
                        """MERGE (t:Tag {name: $tag})
                        WITH t
                        MATCH (f:Fact {id: $id})
                        MERGE (f)-[:HAS_TAG]->(t)""",
                        id=fact.id, tag=tag,
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
        params: dict = {"limit": limit}

        if subject:
            conditions.append("f.subject = $subject")
            params["subject"] = subject
        if verbs is not None:
            # Match any of the expanded verbs (case-insensitive contains)
            verb_conds = [f"toLower(f.verb) CONTAINS toLower($verb_{i})" for i in range(len(verbs))]
            conditions.append("(" + " OR ".join(verb_conds) + ")")
            for i, v in enumerate(verbs):
                params[f"verb_{i}"] = v
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
            query = f"""
                MATCH (f:Fact)-[:HAS_TAG]->(t:Tag)
                WHERE t.name IN $tags AND {where}
                WITH DISTINCT f
                ORDER BY f.mentioned_at DESC
                LIMIT $limit
                RETURN f
            """
        else:
            query = f"""
                MATCH (f:Fact)
                WHERE {where}
                RETURN f
                ORDER BY f.mentioned_at DESC
                LIMIT $limit
            """

        with self._driver.session() as session:
            result = session.run(query, **params)
            return [self._record_to_fact(record["f"]) for record in result]

    def get_unique_fact_verbs(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Fact) WHERE f.verb <> '' AND (f.superseded_by IS NULL OR f.superseded_by = '') "
                "RETURN DISTINCT f.verb AS verb"
            )
            return [record["verb"] for record in result]

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Fact {segment_ref: $ref}) RETURN f ORDER BY f.mentioned_at",
                ref=segment_ref,
            )
            return [self._record_to_fact(record["f"]) for record in result]

    def search_facts(self, query: str, limit: int = 10) -> list[Fact]:
        # Neo4j full-text search requires an explicit index; fall back to CONTAINS
        terms = query.lower().split()
        if not terms:
            return []
        # Match facts where any field contains any search term
        conds = []
        params: dict = {"limit": limit}
        for i, term in enumerate(terms):
            key = f"term_{i}"
            params[key] = term
            conds.append(
                f"(toLower(f.subject) CONTAINS ${key} OR toLower(f.verb) CONTAINS ${key} "
                f"OR toLower(f.object) CONTAINS ${key} OR toLower(f.what) CONTAINS ${key})"
            )
        where = " OR ".join(conds)
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (f:Fact) WHERE (f.superseded_by IS NULL OR f.superseded_by = '') AND ({where}) "
                f"RETURN f ORDER BY f.mentioned_at DESC LIMIT $limit",
                **params,
            )
            return [self._record_to_fact(record["f"]) for record in result]

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (f:Fact {id: $id}) SET f.superseded_by = $new_id",
                id=old_fact_id, new_id=new_fact_id,
            )

    def update_fact_fields(self, fact_id: str, verb: str, object: str, status: str, what: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (f:Fact {id: $id}) SET f.verb = $verb, f.object = $object, "
                "f.status = $status, f.what = $what",
                id=fact_id, verb=verb, object=object, status=status, what=what,
            )

    def get_fact_count_by_tags(self) -> dict[str, int]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Fact)-[:HAS_TAG]->(t:Tag) RETURN t.name AS tag, count(f) AS cnt"
            )
            return {record["tag"]: record["cnt"] for record in result}

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        if not fact_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Fact) WHERE f.superseded_by IN $ids "
                "RETURN f.superseded_by AS superseded_by, f.subject AS subject, "
                "f.verb AS verb, f.object AS object",
                ids=fact_ids,
            )
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # FactLinkStore — native graph edges
    # ------------------------------------------------------------------

    def store_fact_links(self, links: list[FactLink]) -> int:
        if not links:
            return 0
        with self._driver.session() as session:
            count = 0
            for link in links:
                # Use APOC-free approach: MERGE on a generic FACT_LINK rel with properties
                session.run(
                    """MATCH (src:Fact {id: $src_id}), (tgt:Fact {id: $tgt_id})
                    MERGE (src)-[r:FACT_LINK {id: $link_id}]->(tgt)
                    SET r.relation_type = $rel_type, r.confidence = $confidence,
                        r.context = $context, r.created_at = $created_at,
                        r.created_by = $created_by""",
                    src_id=link.source_fact_id, tgt_id=link.target_fact_id,
                    link_id=link.id, rel_type=link.relation_type,
                    confidence=link.confidence, context=link.context,
                    created_at=_dt_to_str(link.created_at), created_by=link.created_by,
                )
                count += 1
        return count

    def get_fact_links(self, fact_id: str, direction: str = "both") -> list[FactLink]:
        with self._driver.session() as session:
            if direction == "outgoing":
                result = session.run(
                    "MATCH (src:Fact {id: $id})-[r:FACT_LINK]->(tgt:Fact) "
                    "RETURN r.id AS id, src.id AS source_fact_id, tgt.id AS target_fact_id, "
                    "r.relation_type AS relation_type, r.confidence AS confidence, "
                    "r.context AS context, r.created_at AS created_at, r.created_by AS created_by",
                    id=fact_id,
                )
            elif direction == "incoming":
                result = session.run(
                    "MATCH (src:Fact)-[r:FACT_LINK]->(tgt:Fact {id: $id}) "
                    "RETURN r.id AS id, src.id AS source_fact_id, tgt.id AS target_fact_id, "
                    "r.relation_type AS relation_type, r.confidence AS confidence, "
                    "r.context AS context, r.created_at AS created_at, r.created_by AS created_by",
                    id=fact_id,
                )
            else:
                result = session.run(
                    "MATCH (a:Fact)-[r:FACT_LINK]-(b:Fact) "
                    "WHERE a.id = $id "
                    "RETURN r.id AS id, startNode(r).id AS source_fact_id, "
                    "endNode(r).id AS target_fact_id, "
                    "r.relation_type AS relation_type, r.confidence AS confidence, "
                    "r.context AS context, r.created_at AS created_at, r.created_by AS created_by",
                    id=fact_id,
                )
            links = []
            for record in result:
                links.append(FactLink(
                    id=record["id"],
                    source_fact_id=record["source_fact_id"],
                    target_fact_id=record["target_fact_id"],
                    relation_type=record["relation_type"],
                    confidence=record["confidence"],
                    context=record["context"] or "",
                    created_at=_str_to_dt(record["created_at"]) if record.get("created_at") else datetime.now(timezone.utc),
                    created_by=record["created_by"] or "compaction",
                ))
            return links

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1) -> list[LinkedFact]:
        if not fact_ids:
            return []
        # Cypher does not allow parameterized variable-length path bounds,
        # so we interpolate depth directly (always a small int, no injection risk).
        depth = max(1, min(depth, 5))
        with self._driver.session() as session:
            result = session.run(
                f"""MATCH (seed:Fact)-[r:FACT_LINK*1..{depth}]-(linked:Fact)
                WHERE seed.id IN $ids
                  AND (linked.superseded_by IS NULL OR linked.superseded_by = '')
                  AND NOT linked.id IN $ids
                WITH seed, linked, r[0] AS first_rel
                RETURN DISTINCT linked AS fact,
                    seed.id AS linked_from,
                    first_rel.relation_type AS relation_type,
                    first_rel.confidence AS confidence,
                    first_rel.context AS context""",
                ids=fact_ids,
            )
            linked_facts = []
            seen: set[str] = set()
            for record in result:
                fact = self._record_to_fact(record["fact"])
                if fact.id in seen:
                    continue
                seen.add(fact.id)
                linked_facts.append(LinkedFact(
                    fact=fact,
                    linked_from_fact_id=record["linked_from"],
                    relation_type=record["relation_type"],
                    confidence=record["confidence"],
                    link_context=record["context"] or "",
                ))
            return linked_facts

    def delete_fact_links(self, fact_id: str) -> int:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Fact {id: $id})-[r:FACT_LINK]-() DELETE r RETURN count(r) AS cnt",
                id=fact_id,
            )
            record = result.single()
            return record["cnt"] if record else 0

    def delete_conversation(self, conversation_id: str) -> int:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Fact {conversation_id: $conversation_id}) RETURN count(f) AS cnt",
                conversation_id=conversation_id,
            )
            record = result.single()
            deleted = int(record["cnt"]) if record else 0
            if deleted:
                session.run(
                    "MATCH (f:Fact {conversation_id: $conversation_id}) DETACH DELETE f",
                    conversation_id=conversation_id,
                )
                session.run(
                    "MATCH (t:Tag) WHERE NOT (t)<-[:HAS_TAG]-(:Fact) DELETE t",
                )
            return deleted

    def migrate_supersession_to_links(self) -> int:
        """Migrate superseded_by properties to SUPERSEDES FACT_LINK edges."""
        with self._driver.session() as session:
            result = session.run(
                """MATCH (old:Fact) WHERE old.superseded_by IS NOT NULL
                MATCH (new:Fact {id: old.superseded_by})
                WHERE NOT (new)-[:FACT_LINK {relation_type: 'supersedes'}]->(old)
                MERGE (new)-[r:FACT_LINK]->(old)
                SET r.relation_type = 'supersedes', r.confidence = 1.0,
                    r.context = 'Migrated from superseded_by property',
                    r.created_by = 'migration',
                    r.id = old.id + '_migration'
                RETURN count(r) AS cnt"""
            )
            record = result.single()
            return record["cnt"] if record else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
