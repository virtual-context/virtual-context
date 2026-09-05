"""ActorCardAdmissionService: explicit dependencies for community memory work."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

from .actor_card_policy import (
    _ACTOR_CARD_SEMANTIC_CONTRACT,
    _ACTOR_CARD_JUDGMENT_RULES,
    _ActorCardAdmissionError,
)

if TYPE_CHECKING:
    from ...types import ActorCardEntry, ActorCardEntrySource

# Keep the existing operator log channel stable across the extraction.
logger = logging.getLogger("virtual_context.core.compaction_pipeline")


class ActorCardAdmissionService:
    def __init__(
        self,
        *,
        config,
        compactor,
        admission_provider: Callable,
        evidence_segments: Callable,
        prompt_turns: Callable,
    ) -> None:
        self._config = config
        self._compactor = compactor
        self._actor_card_admission_provider = admission_provider
        self._actor_card_evidence_segments = evidence_segments
        self._actor_card_prompt_turns = prompt_turns

    def admit_entries(
        self,
        actor_id: str,
        audience_conversation_id: str,
        fact_sources: list,
        turn_sources: list,
        normalized: list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]],
        *,
        curator_substantive: bool,
        existing_entry_ids: set[str] | None = None,
    ) -> tuple[
        list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]],
        str,
        Counter[str],
        bool,
    ]:
        """Independently check coverage and admit immutable candidates."""

        provider = self._actor_card_admission_provider()
        if provider is None:
            raise RuntimeError("actor-card semantic admission model is not configured")
        existing_entry_ids = set(existing_entry_ids or ())

        candidate_fact_ids = {
            source.fact_id
            for _entry, entry_sources in normalized
            for source in entry_sources
            if source.fact_id
        }
        required_fact_ids = {
            source.fact_id
            for entry, entry_sources in normalized
            if entry.id in existing_entry_ids
            for source in entry_sources
            if source.fact_id
        }
        evidence_segments, evidence_refs = self._actor_card_evidence_segments(
            actor_id,
            audience_conversation_id,
            fact_sources,
            candidate_fact_ids,
            required_fact_ids=required_fact_ids,
        )
        fact_source_by_id = {source.fact.id: source for source in fact_sources}
        turn_source_by_id = {source.turn.canonical_turn_id: source for source in turn_sources}
        actor_turns = self._actor_card_prompt_turns(
            turn_sources,
            max_chars=int(
                getattr(
                    self._config.assembler,
                    "actor_card_prompt_max_chars",
                    192_000,
                )
            ),
        )
        visible_turn_ids = {turn["id"] for turn in actor_turns if not turn.get("truncated")}
        candidates: list[dict] = []
        eligible: dict[str, tuple["ActorCardEntry", list["ActorCardEntrySource"]]] = {}
        rejection_counts: Counter[str] = Counter()
        for entry, entry_sources in normalized:
            fact_ids = [source.fact_id for source in entry_sources if source.fact_id]
            turn_ids = [
                source.canonical_turn_id for source in entry_sources if source.canonical_turn_id
            ]
            refs = {
                (
                    fact_source_by_id[fact_id].owner_conversation_id,
                    fact_source_by_id[fact_id].fact.segment_ref,
                )
                for fact_id in fact_ids
                if fact_id in fact_source_by_id
            }
            is_existing = entry.id in existing_entry_ids
            if any(fact_id not in fact_source_by_id for fact_id in fact_ids):
                rejection_counts["evidence_unavailable"] += 1
                continue
            if refs and not refs.issubset(evidence_refs):
                rejection_counts["evidence_unavailable"] += 1
                continue
            if any(
                turn_id not in turn_source_by_id or turn_id not in visible_turn_ids
                for turn_id in turn_ids
            ):
                rejection_counts["evidence_unavailable"] += 1
                continue
            if not fact_ids and not turn_ids:
                rejection_counts["evidence_unavailable"] += 1
                continue
            eligible[entry.id] = (entry, entry_sources)
            candidates.append(
                {
                    "candidate_id": entry.id,
                    "origin": "existing" if is_existing else "fresh",
                    "kind": entry.kind,
                    "body": entry.body,
                    "proposed_confidence": entry.confidence,
                    "fact_ids": fact_ids,
                    "turn_ids": turn_ids,
                    "source_segments": [
                        {
                            "owner_conversation_id": owner,
                            "segment_ref": ref,
                        }
                        for owner, ref in sorted(refs)
                    ],
                }
            )

        compact_facts = [
            {
                "id": source.fact.id,
                "owner_conversation_id": source.owner_conversation_id,
                "segment_ref": source.fact.segment_ref,
                "fact": source.fact.format_for_prompt(),
                "status": source.fact.status,
                "mentioned_at": source.fact.mentioned_at.isoformat(),
            }
            for source in fact_sources
        ]
        system = (
            (
                "You are the conservative semantic admission gate for a person "
                "card. Candidate bodies are immutable: you may admit or reject "
                "them, but you may not invent, rewrite, or merge candidates. Use "
                "only actor-authored facts and source "
                "messages. All bounded actor turns and compact facts are supplied "
                "so later evidence can revoke or replace a candidate. "
                "Independently decide whether this actor contributed substantive "
                "interaction; do not defer to the curator's claim. Substantive "
                "means at least one informative message that reveals a useful "
                "ongoing goal, durable preference/style, or a meaningful topic "
                "the actor discussed with the agent. Greetings, bot invocation "
                "checks, memory/preference probes, and isolated trivia questions "
                "are not substantive. A substantive actor must finish with at "
                "least one admitted card entry; relevant_history is appropriate "
                "for useful topic continuity when no narrower entry is justified. "
                "Return JSON only with exactly substantive, coverage_reason, and "
                "decisions. substantive must be boolean. coverage_reason must be "
                'exactly one of "substantive", "greeting_only", '
                '"one_off_trivia", "bot_meta_or_test", '
                '"no_durable_context", or "insufficient_evidence", and must '
                'be "substantive" exactly when substantive is true. '
                "Return exactly one decision for every candidate, with "
                "exactly candidate_id, admit, and reason. admit must be a boolean. "
                "reason must be exactly one of "
                '"durable", "temporary", "test_probe", '
                '"stopped_or_replaced", "completed", "contradicted", '
                '"insufficient_evidence", "not_durable", '
                '"not_person_card", "wrong_subject", "wrong_kind", '
                '"irrelevant_citation", "redundant", '
                '"explicit_privacy_request", "agent_refused", or '
                '"safety_posture_request". Use reason '
                '"durable" if and only if admit is true. '
                "Candidate origin is either fresh or existing. An existing "
                "candidate is an immutable entry that a prior independent "
                "admission accepted under an earlier policy. Curator omission is "
                "not evidence against it, but prior acceptance is not proof under "
                "this policy. Apply every subject, kind, citation, durability, "
                "privacy, and revocation check equally to fresh and existing "
                "candidates. Re-admit an existing candidate only if it still "
                "passes all current checks. Apply the shared semantic contract "
                "before judging durability. Reject with wrong_subject when the "
                "candidate violates its role-preservation clauses. Reject with "
                "wrong_kind when kind does not fit the immutable body and exact "
                "source. Reject with irrelevant_citation when any citation violates "
                "the contract's material-support rule. Apply those reasons in that "
                "order: a role error is wrong_subject even when the kind is also "
                "wrong; use wrong_kind only after subject and roles are correct. "
                "When fresh and existing "
                "candidates "
                "substantially overlap, prefer the existing candidate for "
                "continuity unless the fresh one materially corrects, updates, or "
                "better preserves the evidence; reject the other as redundant or "
                "stopped_or_replaced. Do not admit redundant copies. "
                "Reject temporary, test/probe, one-turn, session-only, "
                "channel-only, stopped, replaced, completed, or contradicted "
                "material. Later source messages revoke or replace earlier "
                "material. Requested answer prefixes and memory-system tests are "
                "not durable identity preferences. A communication preference or "
                "interaction style is admissible only when a source message "
                "explicitly establishes durability beyond the current test, "
                "session, and channel, or when consistent natural evidence appears "
                "across distinct actor-authored messages or interactions, ideally "
                "spread over time. Repeated test instructions do not establish a "
                "pattern. The immutable candidate body itself "
                "must be self-contained and unambiguous without relying on the "
                "surrounding segment; reject with insufficient_evidence when an "
                "essential referent (such as which medication, goal, or "
                "preference) is omitted. The body must be fully entailed by the "
                "cited actor-authored messages. Compact fact fields and tags help "
                "locate evidence but cannot independently justify body text. "
                "Full entailment requires preserving every material qualifier in "
                "the source, including exceptions, exclusions, uncertainty, "
                "frequency, timing, and scope. Reject with insufficient_evidence "
                "if the immutable body drops a qualifier, broadens the statement, "
                "or increases its certainty. For example, source text like "
                '"good blood pressure, excluding stressful events" does not '
                'entail the unqualified body "has good blood pressure." '
                "Reject with not_person_card when a body exposes internal "
                "ontology/tag language or serializes a machine fact triple rather "
                "than stating a natural person fact. Subject matter must never "
                "determine admission: medical, sexual, financial, location, "
                "credential, and other topics are evaluated by the same durability "
                "and evidence rules as every other topic. Do not omit, soften, or "
                "reject a candidate because of its subject. Use "
                "explicit_privacy_request only when actor-authored evidence "
                "explicitly and unambiguously asks that the cited information not "
                "be retained or reused; never infer privacy from the topic, from a "
                "DM, or from context. A visibly truncated turn cannot prove a "
                "claim whose qualifier may be in omitted text. When uncertain, "
                "reject. "
            )
            + _ACTOR_CARD_SEMANTIC_CONTRACT
            + _ACTOR_CARD_JUDGMENT_RULES
        )
        user = json.dumps(
            {
                "curator_substantive_claim": curator_substantive,
                "candidates": candidates,
                "facts": compact_facts,
                "actor_turns": actor_turns,
                "evidence_segments": evidence_segments,
            },
            separators=(",", ":"),
        )
        request_kwargs = {
            "system": system,
            "user": user,
            "max_tokens": max(
                800,
                min(
                    4000,
                    300 + 250 * len(candidates),
                ),
            ),
        }
        complete_with_source = getattr(provider, "complete_with_source", None)
        if callable(complete_with_source):
            response_text, _usage, admission_source = complete_with_source(
                **request_kwargs,
            )
        else:
            response_text, _usage = provider.complete(**request_kwargs)
            admission_source = "provider"

        def _parse_admission(
            text: str,
        ) -> tuple[bool, dict[str, dict]]:
            try:
                parsed = self._compactor._parse_response(text)
            except Exception as exc:
                raise _ActorCardAdmissionError(
                    "actor-card admission response is not valid JSON",
                    text,
                ) from exc
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"substantive", "coverage_reason", "decisions"}
                or not isinstance(parsed["substantive"], bool)
                or not isinstance(parsed["coverage_reason"], str)
                or not isinstance(parsed["decisions"], list)
            ):
                raise _ActorCardAdmissionError(
                    "actor-card admission response has invalid coverage shape",
                    text,
                )
            independently_substantive = parsed["substantive"]
            valid_coverage_reasons = {
                "substantive",
                "greeting_only",
                "one_off_trivia",
                "bot_meta_or_test",
                "no_durable_context",
                "insufficient_evidence",
            }
            if parsed[
                "coverage_reason"
            ] not in valid_coverage_reasons or independently_substantive != (
                parsed["coverage_reason"] == "substantive"
            ):
                raise _ActorCardAdmissionError(
                    "actor-card admission response has invalid coverage decision",
                    text,
                )
            decisions: dict[str, dict] = {}
            valid_reasons = {
                "durable",
                "temporary",
                "test_probe",
                "stopped_or_replaced",
                "completed",
                "contradicted",
                "insufficient_evidence",
                "not_durable",
                "not_person_card",
                "wrong_subject",
                "wrong_kind",
                "irrelevant_citation",
                "redundant",
                "explicit_privacy_request",
                "agent_refused",
                "safety_posture_request",
            }
            for decision in parsed["decisions"]:
                if (
                    not isinstance(decision, dict)
                    or set(decision) != {"candidate_id", "admit", "reason"}
                    or not isinstance(decision.get("candidate_id"), str)
                    or not isinstance(decision.get("admit"), bool)
                    or decision.get("reason") not in valid_reasons
                    or (bool(decision.get("admit")) != (decision.get("reason") == "durable"))
                    or decision["candidate_id"] in decisions
                ):
                    raise _ActorCardAdmissionError(
                        "actor-card admission response contains an invalid decision",
                        text,
                    )
                decisions[decision["candidate_id"]] = decision
            if set(decisions) != set(eligible):
                raise _ActorCardAdmissionError(
                    "actor-card admission response does not cover every candidate",
                    text,
                )
            return independently_substantive, decisions

        try:
            independently_substantive, decisions = _parse_admission(
                response_text,
            )
        except _ActorCardAdmissionError as primary_exc:
            complete_fallback = getattr(provider, "complete_fallback", None)
            if admission_source == "fallback" or not callable(complete_fallback):
                raise
            logger.warning(
                "ACTOR_CARD_ADMISSION_FALLBACK reason=invalid_response response_hash=%s",
                hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:16],
            )
            fallback_text = ""
            try:
                fallback_text, _usage = complete_fallback(**request_kwargs)
                independently_substantive, decisions = _parse_admission(
                    fallback_text,
                )
            except Exception as fallback_exc:
                combined = json.dumps(
                    {
                        "primary": primary_exc.response_text,
                        "fallback": (getattr(fallback_exc, "response_text", "") or fallback_text),
                    },
                    separators=(",", ":"),
                )
                raise _ActorCardAdmissionError(
                    "actor-card admission primary and fallback responses were invalid",
                    combined,
                ) from fallback_exc
            response_text = json.dumps(
                {
                    "primary": response_text,
                    "fallback": fallback_text,
                    "selected": "fallback",
                },
                separators=(",", ":"),
            )
            admission_source = "fallback"
        if independently_substantive != curator_substantive:
            primary_substantive = independently_substantive
            complete_fallback = getattr(provider, "complete_fallback", None)
            if admission_source == "fallback" or not callable(complete_fallback):
                # No third adjudicator is available: the fallback already
                # served the admission call, or the provider has none.
                # Resolve to the admission judgment instead of failing the
                # rebuild. The admission decisions are the stricter,
                # internally consistent gate; a boolean coverage flag must
                # never wedge an active actor into a retry loop.
                logger.warning(
                    "ACTOR_CARD_COVERAGE_RESOLVED_CONSERVATIVE curator=%s "
                    "admission=%s admission_source=%s",
                    curator_substantive,
                    independently_substantive,
                    admission_source,
                )
                response_text = json.dumps(
                    {
                        "primary": response_text,
                        "selected": "admission_conservative",
                    },
                    separators=(",", ":"),
                )
            else:
                adjudication_text = ""
                try:
                    adjudication_text, _usage = complete_fallback(**request_kwargs)
                    adjudicated_substantive, adjudicated_decisions = _parse_admission(
                        adjudication_text
                    )
                except Exception as exc:
                    combined = json.dumps(
                        {
                            "primary": response_text,
                            "adjudicator": (getattr(exc, "response_text", "") or adjudication_text),
                            "selected": "error",
                            "error_type": type(exc).__name__,
                        },
                        separators=(",", ":"),
                    )
                    raise _ActorCardAdmissionError(
                        "actor-card coverage adjudicator failed",
                        combined,
                    ) from exc
                # With boolean coverage and an initial disagreement, the
                # third judgment necessarily agrees with either the curator
                # or the primary admission model. Select that two-of-three
                # result and its internally consistent candidate decisions.
                selected = "primary"
                if adjudicated_substantive == curator_substantive:
                    independently_substantive = adjudicated_substantive
                    decisions = adjudicated_decisions
                    selected = "curator_fallback"
                logger.warning(
                    "ACTOR_CARD_COVERAGE_ADJUDICATED curator=%s primary=%s fallback=%s selected=%s",
                    curator_substantive,
                    primary_substantive,
                    adjudicated_substantive,
                    selected,
                )
                response_text = json.dumps(
                    {
                        "primary": response_text,
                        "adjudicator": adjudication_text,
                        "selected": selected,
                    },
                    separators=(",", ":"),
                )

        admitted: list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]] = []
        for candidate_id, (entry, entry_sources) in eligible.items():
            decision = decisions[candidate_id]
            if candidate_id in existing_entry_ids:
                logger.info(
                    "ACTOR_CARD_CARRYOVER_DECISION actor=%s audience=%s "
                    "entry=%s admit=%s reason=%s",
                    actor_id[:24],
                    audience_conversation_id[:48],
                    candidate_id,
                    bool(decision["admit"]),
                    decision["reason"],
                )
            if not decision["admit"]:
                rejection_counts[f"semantic_{decision['reason']}"] += 1
                continue
            admitted.append((entry, entry_sources))
        if not independently_substantive and admitted:
            raise _ActorCardAdmissionError(
                "non-substantive actor cannot have an admitted card entry",
                response_text,
            )
        return (
            admitted,
            response_text,
            rejection_counts,
            independently_substantive,
        )
