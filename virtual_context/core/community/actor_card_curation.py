"""ActorCardCurationService: explicit dependencies for community memory work."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from .actor_card_policy import (
    _ACTOR_CARD_CITATION_LIMIT,
    _ACTOR_CARD_SEMANTIC_CONTRACT,
    _ACTOR_CARD_JUDGMENT_RULES,
    _ACTOR_CARD_CONFIDENCE_SCALE,
    _ActorCardAdmissionError,
    _EmptyResponseFallbackProvider,
)

if TYPE_CHECKING:
    pass

# Keep the existing operator log channel stable across the extraction.
logger = logging.getLogger("virtual_context.core.compaction_pipeline")


class ActorCardCurationService:
    def __init__(
        self,
        *,
        config,
        compactor,
        prompt_turns: Callable,
        curation_provider: Callable,
        provider_for_model: Callable,
        curation_override=None,
        admission_override=None,
    ) -> None:
        self._config = config
        self._compactor = compactor
        self._actor_card_prompt_turns = prompt_turns
        self._actor_card_curation_provider = curation_provider
        self._actor_card_provider_for_model = provider_for_model
        self._actor_card_curation_provider_override = curation_override
        self._actor_card_admission_provider_override = admission_override

    def curate_partition(
        self,
        fact_sources: list,
        turn_sources: list,
    ) -> tuple[str, bool, str, list, set[str]]:
        """Curate one audience partition without exposing another audience."""
        from ...types import CARD_ENTRY_BODY_MAX_CHARS

        prompt_facts = [
            {
                "id": source.fact.id,
                "fact": source.fact.format_for_prompt(),
                "author_role": source.fact.author_source_role,
                "status": source.fact.status,
                "fact_type": source.fact.fact_type,
                "mentioned_at": source.fact.mentioned_at.isoformat(),
                "session_date": source.fact.session_date,
            }
            for source in fact_sources
        ]
        prompt_turns = self._actor_card_prompt_turns(
            turn_sources,
            max_chars=int(
                getattr(
                    self._config.assembler,
                    "actor_card_prompt_max_chars",
                    192_000,
                )
            ),
        )
        prompt_turn_ids = {item["id"] for item in prompt_turns}
        system = (
            (
                "Curate a compact person card from exact messages and facts "
                "authored by one actor in one policy audience. The card is for "
                "durable interaction continuity, not a fact scrapbook or a "
                "transcript. Independently decide whether the actor contributed "
                "substantive interaction. Substantive means at least one "
                "informative message that reveals a useful ongoing goal, durable "
                "preference/style, or a meaningful topic the actor has discussed "
                "with the agent. Greetings, bot invocation checks, memory/"
                "preference probes, and isolated trivia questions are not "
                "substantive. Return JSON only with exactly: substantive, "
                "coverage_reason, and entries. substantive must be boolean. "
                'coverage_reason must be exactly one of "substantive", '
                '"greeting_only", "one_off_trivia", "bot_meta_or_test", '
                '"no_durable_context", or "insufficient_evidence". It must be '
                '"substantive" exactly when substantive is true. entries must '
                "be an array. A substantive actor must receive at least one entry; "
                "a non-substantive actor must receive none. Each entry must contain "
                "exactly kind, body, confidence, fact_ids, and turn_ids. kind must "
                'be exactly one of "communication_pref", "active_goal", '
                '"relevant_history", or "interaction_style". confidence must '
                "be a number from 0 through 1. fact_ids "
                "and turn_ids must be arrays, may individually be empty, and "
                "together must cite at least one provided id that fully supports "
                f"the body. Use at most {_ACTOR_CARD_CITATION_LIMIT} citation ids "
                "total per entry. Put "
                "fact ids only in fact_ids and turn ids only in turn_ids; never "
                "copy an id into both arrays. Obey entries_per_kind as a hard "
                "maximum. Use a neutral concise body and do not invent identity "
                "or intent. Every body must be self-contained and unambiguous when "
                "read without the surrounding transcript; include essential "
                "referents such as the specific medication, goal, preference, or "
                "discussion topic. Write natural person-facing language, not a "
                "serialization of subject/verb/object fields, ontology names, or "
                "tag labels. Preserve every material qualifier from the source, "
                "including exceptions, exclusions, uncertainty, frequency, "
                "timing, and scope. Represent one quoted utterance as an example; "
                "do not rewrite it as 'frequently', 'often', or 'always', or "
                "normalize its meaningful terms (for example, 'pal-o' to 'pal'). "
                "Do not turn a qualified statement into a "
                "broader or more certain claim. Do not promote temporary, "
                "test-only, one-turn, session-only, or channel-only instructions "
                "into communication_pref or interaction_style. Do not retain a "
                "preference or goal that later evidence stopped, replaced, "
                "completed, or contradicted. Use message timestamps, mentioned_at, "
                "and status to resolve conflicts, with the newest applicable "
                "evidence winning. A communication preference or interaction "
                "style is durable only when explicitly stated as lasting or "
                "consistently supported by repeated natural interactions. Use "
                "relevant_history for concise, useful continuity about a "
                "meaningful topic this actor actually discussed with the agent "
                "when no narrower durable preference or goal is justified. "
                "An isolated, underspecified follow-up whose missing referent "
                "cannot be recovered from the supplied evidence is insufficient "
                "for relevant_history, even if it sounds important. Subject matter "
                "must never determine admission: medical, sexual, financial, "
                "location, credential, and other topics are evaluated by the same "
                "durability and evidence rules as every other topic. Do not omit "
                "or soften a candidate because of its subject. If the actor "
                "explicitly and unambiguously asks that particular information "
                "not be retained or reused, do not propose it for the card. Do not "
                "infer such a request from the topic, from a DM, or from context. "
            )
            + _ACTOR_CARD_SEMANTIC_CONTRACT
            + _ACTOR_CARD_JUDGMENT_RULES
            + _ACTOR_CARD_CONFIDENCE_SCALE
        )
        user = json.dumps(
            {
                "facts": prompt_facts,
                "turns": prompt_turns,
                "limits": {
                    "entries_per_kind": int(self._config.assembler.actor_card_entries_per_kind),
                    "body_chars": CARD_ENTRY_BODY_MAX_CHARS,
                },
            },
            separators=(",", ":"),
        )
        valid_coverage_reasons = {
            "substantive",
            "greeting_only",
            "one_off_trivia",
            "bot_meta_or_test",
            "no_durable_context",
            "insufficient_evidence",
        }

        def _parse_curation(text: str) -> dict:
            try:
                parsed = self._compactor._parse_response(text)
            except Exception as exc:
                raise _ActorCardAdmissionError(
                    "actor card curation response is not valid JSON",
                    text,
                ) from exc
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"substantive", "coverage_reason", "entries"}
                or not isinstance(parsed.get("substantive"), bool)
                or not isinstance(parsed.get("coverage_reason"), str)
                or not isinstance(parsed.get("entries"), list)
                or parsed["coverage_reason"] not in valid_coverage_reasons
                or parsed["substantive"] != (parsed["coverage_reason"] == "substantive")
                or parsed["substantive"] != bool(parsed["entries"])
            ):
                raise _ActorCardAdmissionError(
                    "actor card curation response has invalid coverage shape",
                    text,
                )
            return parsed

        request_kwargs = {
            "system": system,
            "user": user,
            "max_tokens": max(
                2000,
                min(
                    4000,
                    2 * int(self._config.compactor.max_summary_tokens),
                ),
            ),
        }
        provider = self._actor_card_curation_provider()
        complete_with_source = getattr(provider, "complete_with_source", None)
        if callable(complete_with_source):
            response_text, _usage, curation_source = complete_with_source(
                **request_kwargs,
            )
        else:
            response_text, _usage = provider.complete(**request_kwargs)
            curation_source = "provider"

        try:
            parsed = _parse_curation(response_text)
        except _ActorCardAdmissionError as primary_exc:
            complete_fallback = getattr(provider, "complete_fallback", None)
            if curation_source == "fallback" or not callable(complete_fallback):
                raise
            logger.warning(
                "ACTOR_CARD_CURATION_FALLBACK reason=invalid_response response_hash=%s",
                hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:16],
            )
            fallback_text = ""
            try:
                fallback_text, _usage = complete_fallback(**request_kwargs)
                parsed = _parse_curation(fallback_text)
            except Exception as fallback_exc:
                combined = json.dumps(
                    {
                        "primary": primary_exc.response_text,
                        "fallback": (getattr(fallback_exc, "response_text", "") or fallback_text),
                    },
                    separators=(",", ":"),
                )
                raise _ActorCardAdmissionError(
                    "actor card curation primary and fallback responses were invalid",
                    combined,
                ) from fallback_exc
            response_text = fallback_text
        return (
            response_text,
            parsed["substantive"],
            parsed["coverage_reason"],
            parsed["entries"],
            prompt_turn_ids,
        )

    def provider_for_model(self, selected_model: str):
        """Create a zero-temperature provider through the configured gateway."""
        base = self._compactor.llm
        from ...providers.anthropic import AnthropicProvider
        from ...providers.generic_openai import GenericOpenAIProvider

        if isinstance(base, GenericOpenAIProvider):
            return GenericOpenAIProvider(
                base_url=base.base_url,
                model=selected_model,
                temperature=0.0,
                api_key=base.api_key,
                reasoning_effort="low",
            )
        if isinstance(base, AnthropicProvider):
            return AnthropicProvider(
                api_key=base.api_key,
                model=selected_model,
                temperature=0.0,
            )
        raise RuntimeError(f"actor-card model override is unsupported by {type(base).__name__}")

    def curation_provider(self):
        """Build the optional dedicated curator and malformed-response fallback."""
        override = getattr(
            self,
            "_actor_card_curation_provider_override",
            None,
        )
        if override is not None:
            return override
        model = (
            getattr(
                self._config.assembler,
                "actor_card_curation_model",
                "",
            )
            or ""
        ).strip()
        if not model:
            return self._compactor.llm
        fallback_model = (
            getattr(
                self._config.assembler,
                "actor_card_curation_fallback_model",
                "",
            )
            or ""
        ).strip()
        primary = self._actor_card_provider_for_model(model)
        if fallback_model and fallback_model != model:
            return _EmptyResponseFallbackProvider(
                primary,
                self._actor_card_provider_for_model(fallback_model),
                primary_model=model,
                fallback_model=fallback_model,
                stage="curation",
            )
        return primary

    def admission_provider(self):
        """Build the dedicated semantic admission provider.

        The curation model may be deliberately cheap. Admission is a separate
        safety boundary over immutable candidates and actor-authored evidence;
        it cannot invent or rewrite card bodies.
        """
        override = getattr(
            self,
            "_actor_card_admission_provider_override",
            None,
        )
        if override is not None:
            return override
        model = (
            getattr(
                self._config.assembler,
                "actor_card_admission_model",
                "",
            )
            or ""
        ).strip()
        if not model:
            return None
        fallback_model = (
            getattr(
                self._config.assembler,
                "actor_card_admission_fallback_model",
                "",
            )
            or ""
        ).strip()
        primary = self._actor_card_provider_for_model(model)
        if fallback_model and fallback_model != model:
            return _EmptyResponseFallbackProvider(
                primary,
                self._actor_card_provider_for_model(fallback_model),
                primary_model=model,
                fallback_model=fallback_model,
            )
        return primary
