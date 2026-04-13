from __future__ import annotations

from datetime import datetime, timedelta, timezone

from virtual_context.core.temporal_resolver import TemporalResolver
from virtual_context.types import (
    Fact,
    LinkedFact,
    QuoteResult,
    SegmentMetadata,
    StoredSegment,
    VirtualContextConfig,
)


def _make_quote(ref: str, tag: str, session_date: str, text: str) -> QuoteResult:
    return QuoteResult(
        text=text,
        tag=tag,
        segment_ref=ref,
        tags=[tag],
        match_type="fts",
        session_date=session_date,
        created_at="2024-08-01T00:00:00Z",
    )


def _make_segment(ref: str, tag: str, session_date: str, summary: str) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        conversation_id="conv-1",
        primary_tag=tag,
        tags=[tag],
        summary=summary,
        metadata=SegmentMetadata(session_date=session_date),
        created_at=datetime.now(timezone.utc),
    )


def _make_fact(
    fact_id: str,
    when_date: str,
    what: str,
    *,
    tags: list[str] | None = None,
    segment_ref: str = "",
) -> Fact:
    return Fact(
        id=fact_id,
        subject="system architecture",
        verb="changed",
        object="performance plan",
        what=what,
        when_date=when_date,
        conversation_id="conv-1",
        mentioned_at=datetime.now(timezone.utc),
        tags=tags or [],
        segment_ref=segment_ref,
    )


class FakeStore:
    def __init__(self) -> None:
        self.segment_hits: dict[str, list[QuoteResult]] = {}
        self.summary_hits: dict[str, list[StoredSegment]] = {}
        self.fact_hits: dict[str, list[Fact]] = {}
        self.segments: dict[str, StoredSegment] = {}
        self.fallback_facts: list[Fact] = []
        self.segment_facts: dict[str, list[Fact]] = {}
        self.linked_facts: list[LinkedFact] = []
        self.fallback_called = False

    def search(self, query: str, limit: int = 5, conversation_id: str | None = None):
        return self.summary_hits.get(query, [])[:limit]

    def search_full_text(self, query: str, limit: int = 5, conversation_id: str | None = None):
        return self.segment_hits.get(query, [])[:limit]

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None):
        return self.fact_hits.get(query, [])[:limit]

    def get_segment(self, ref: str, conversation_id: str | None = None):
        return self.segments.get(ref)

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ):
        items = list(self.segments.values())
        if conversation_id is not None:
            items = [seg for seg in items if seg.conversation_id == conversation_id]
        if limit is not None and limit > 0:
            items = items[:limit]
        return items

    def get_facts_by_segment(self, segment_ref: str):
        return self.segment_facts.get(segment_ref, [])

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ):
        self.fallback_called = True
        return self.fallback_facts[:limit]

    def get_linked_facts(self, fact_ids: list[str], depth: int = 1):
        return self.linked_facts[:]


class FakeSearch:
    def __init__(self) -> None:
        self.calls = 0

    def find_quote(self, query: str, max_results: int = 5):
        self.calls += 1
        return {
            "found": False,
            "results": [],
            "message": "No matches found.",
        }


class FakeSemantic:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def get_embed_fn(self):
        def embed(texts: list[str]) -> list[list[float]]:
            return [list(self._vectors.get(text, [0.0, 0.0])) for text in texts]

        return embed


def _make_config() -> VirtualContextConfig:
    cfg = VirtualContextConfig(conversation_id="conv-1")
    cfg.search.remember_when_max_results = 4
    return cfg


def test_remember_when_prefers_time_diverse_results_for_timeline_queries():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    quotes = {
        "architecture": [
            _make_quote("seg-1", "architecture", "July-01-2024", "architecture baseline"),
            _make_quote("seg-3", "architecture", "July-09-2024", "architecture revision"),
            _make_quote("seg-5", "architecture", "July-18-2024", "architecture split"),
            _make_quote("seg-6", "architecture", "July-29-2024", "architecture final"),
        ],
        "performance": [
            _make_quote("seg-2", "performance", "July-05-2024", "performance target"),
            _make_quote("seg-4", "performance", "July-12-2024", "performance tuning"),
            _make_quote("seg-5", "architecture", "July-18-2024", "throughput and performance"),
            _make_quote("seg-6", "architecture", "July-29-2024", "performance finalized"),
        ],
        "optimization": [
            _make_quote("seg-3", "architecture", "July-09-2024", "optimization applied"),
            _make_quote("seg-4", "performance", "July-12-2024", "optimization plan"),
            _make_quote("seg-6", "architecture", "July-29-2024", "optimization complete"),
        ],
    }
    store.segment_hits.update(quotes)
    for ref, tag, session, summary in [
        ("seg-1", "architecture", "July-01-2024", "July 1 summary"),
        ("seg-2", "performance", "July-05-2024", "July 5 summary"),
        ("seg-3", "architecture", "July-09-2024", "July 9 summary"),
        ("seg-4", "performance", "July-12-2024", "July 12 summary"),
        ("seg-5", "architecture", "July-18-2024", "July 18 summary"),
        ("seg-6", "architecture", "July-29-2024", "July 29 summary"),
    ]:
        store.segments[ref] = _make_segment(ref, tag, session, summary)

    result = resolver.remember_when(
        query="Can you summarize how my architecture performance optimization evolved?",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=4,
    )

    dates = [item["session_date_normalized"] for item in result["results"]]
    assert dates == sorted(dates)
    assert len(dates) == 4
    assert dates[0] == "2024-07-01"
    assert dates[-1] == "2024-07-29"
    assert len(set(dates)) == 4
    assert [item["excerpt"] for item in result["results"]] == [
        "July 1 summary",
        "July 9 summary",
        "July 12 summary",
        "July 29 summary",
    ]


def test_remember_when_default_max_results_increases_for_change_over_time():
    resolver = TemporalResolver(store=FakeStore(), search_engine=FakeSearch(), config=_make_config())

    assert resolver._default_remember_when_max_results("summarize_over_time") == 4
    assert resolver._default_remember_when_max_results("change_over_time") == 22


def test_change_over_time_date_buckets_keep_more_results_per_day():
    resolver = TemporalResolver(store=FakeStore(), search_engine=FakeSearch(), config=_make_config())

    results = [
        {
            "session_date_normalized": "2024-11-21",
            "topic": f"topic-{idx}",
            "excerpt": f"excerpt-{idx}",
            "matched_terms": ["error"],
            "segment_ref": f"seg-{idx}",
            "match_type": "summary",
        }
        for idx in range(5)
    ]
    facts = [
        {
            "when": "2024-11-21",
            "what": f"fact-{idx}",
            "tags": ["error-handling"],
            "matched_terms": ["handling"],
            "segment_ref": f"seg-f-{idx}",
        }
        for idx in range(4)
    ]

    buckets = resolver._build_change_date_buckets(results=results, facts=facts, max_results=18)

    assert len(buckets) == 1
    assert [item["topic"] for item in buckets[0]["results"]] == [
        "topic-0",
        "topic-1",
        "topic-2",
        "topic-3",
    ]
    assert [item["what"] for item in buckets[0]["facts"]] == [
        "fact-0",
        "fact-1",
        "fact-2",
    ]


def test_change_over_time_adds_semantic_summary_candidates_from_intent_context():
    store = FakeStore()
    search = FakeSearch()
    config = _make_config()
    semantic = FakeSemantic(
        {
            "Can you reconstruct the sequence in which I brought up the various error types and their handling challenges from 2024-11-01 to 2025-01-21 in order?": [1.0, 0.0],
            "error types handling challenges": [0.0, 1.0],
            "model-versioning-rollback model-versioning-rollback VersionConflictError during checkpoint saving": [0.95, 0.05],
            "feedback-parsing-error feedback-parsing-error feedback parsing problems in retry loop": [0.10, 0.90],
        }
    )
    resolver = TemporalResolver(store=store, search_engine=search, config=config, semantic=semantic)

    store.segments["seg-version"] = _make_segment(
        "seg-version",
        "model-versioning-rollback",
        "November-25-2024",
        "VersionConflictError during checkpoint saving",
    )
    store.segments["seg-feedback"] = _make_segment(
        "seg-feedback",
        "feedback-parsing-error",
        "November-21-2024",
        "feedback parsing problems in retry loop",
    )

    result = resolver.remember_when(
        query="error types handling challenges",
        intent_context=(
            "Can you reconstruct the sequence in which I brought up the various "
            "error types and their handling challenges from 2024-11-01 to "
            "2025-01-21 in order?"
        ),
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-30"},
        max_results=4,
        mode="change_over_time",
    )

    surfaced_dates = [item["session_date_normalized"] for item in result["results"]]
    surfaced_topics = [item["topic"] for item in result["results"]]
    assert "2024-11-25" in surfaced_dates
    assert "model-versioning-rollback" in surfaced_topics


def test_change_over_time_prunes_generic_unigrams() -> None:
    resolver = TemporalResolver(store=FakeStore(), search_engine=FakeSearch(), config=_make_config())
    variants = [
        ("error types", 3.0, {"error", "types"}),
        ("types handling", 3.0, {"types", "handling"}),
        ("handling challenges", 3.0, {"handling", "challenges"}),
        ("error", 1.75, {"error"}),
        ("types", 1.75, {"types"}),
        ("handling", 1.75, {"handling"}),
        ("challenges", 1.75, {"challenges"}),
    ]

    assert [variant for variant, _, _ in resolver._prune_change_search_variants(variants)] == [
        "error types",
        "types handling",
        "handling challenges",
    ]


def test_change_over_time_excerpt_uses_first_sentence() -> None:
    resolver = TemporalResolver(store=FakeStore(), search_engine=FakeSearch(), config=_make_config())

    assert resolver._change_result_excerpt(
        "First sentence about indexing progress. Second sentence about unrelated implementation details."
    ) == "First sentence about indexing progress."


def test_change_over_time_supplements_fact_backed_empty_dates() -> None:
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segments["seg-terraform"] = _make_segment(
        "seg-terraform",
        "terraform-configuration",
        "September-06-2024",
        "Terraform configuration for reproducible RAG environments and deployment workflows.",
    )
    store.segments["seg-other"] = _make_segment(
        "seg-other",
        "jira-integration",
        "September-06-2024",
        "Jira workflow automation for team coordination.",
    )

    supplemented = resolver._supplement_change_results_for_fact_backed_dates(
        query="RAG system development phases discussions",
        intent_context=(
            "How did my discussions about the development phases of our RAG system "
            "progress from 2024-08-01 to 2024-10-22 in order?"
        ),
        results=[],
        facts=[
            {
                "what": "Terraform modules were used for environment reproducibility.",
                "when": "2024-09-06",
                "tags": ["terraform-configuration", "iac-optimization"],
                "matched_terms": ["rag"],
                "segment_ref": "seg-terraform",
            },
            {
                "what": "Reusable IaC deployment setup was planned for the RAG system.",
                "when": "2024-09-06",
                "tags": ["terraform-configuration", "state-management"],
                "matched_terms": ["rag"],
                "segment_ref": "seg-terraform",
            },
        ],
        start=datetime(2024, 8, 1, tzinfo=timezone.utc).date(),
        end=datetime(2024, 10, 22, tzinfo=timezone.utc).date(),
    )

    assert len(supplemented) == 1
    assert supplemented[0]["session_date_normalized"] == "2024-09-06"
    assert supplemented[0]["topic"] == "terraform-configuration"
    assert supplemented[0]["segment_ref"] == "seg-terraform"
    assert supplemented[0]["match_type"] == "summary_date_expand"


def test_remember_when_uses_query_matched_facts_before_date_fallback():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["architecture"] = [
        _make_quote("seg-1", "architecture", "July-01-2024", "architecture baseline"),
    ]
    store.segments["seg-1"] = _make_segment("seg-1", "architecture", "July-01-2024", "July 1 summary")

    store.fact_hits["architecture"] = [
        _make_fact("fact-1", "2024-07-01", "Initial architecture defined"),
        _make_fact("fact-2", "2024-07-29", "Architecture changed to microservices"),
    ]
    store.fallback_facts = [
        _make_fact("fact-old", "2024-07-01", "Irrelevant early fallback fact"),
    ]

    result = resolver.remember_when(
        query="Summarize architecture changes",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
    )

    fact_dates = [item["when"] for item in result["facts_in_window"]]
    assert store.fallback_called is False
    assert "2024-07-29" in fact_dates
    assert any("microservices" in item["what"].lower() for item in result["facts_in_window"])


def test_remember_when_change_over_time_mode_merges_broad_window_facts():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["architecture"] = [
        _make_quote("seg-1", "architecture", "July-01-2024", "architecture baseline"),
    ]
    store.segments["seg-1"] = _make_segment("seg-1", "architecture", "July-01-2024", "July 1 summary")

    # Plain fact search returns only a generic architecture hit, which would
    # previously suppress the date-window scan entirely.
    store.fact_hits["architecture"] = [
        _make_fact("fact-generic", "2024-07-01", "Initial architecture defined"),
    ]
    store.fallback_facts = [
        _make_fact(
            "fact-health",
            "2024-07-02",
            "Health checks configured to detect unhealthy instances",
            tags=["load-balancing", "performance-optimization"],
        ),
        _make_fact(
            "fact-deploy",
            "2024-07-02",
            "Deployment spec with replicas and LoadBalancer service",
            tags=["kubernetes-deployment", "performance-optimization"],
        ),
        _make_fact(
            "fact-redis",
            "2024-07-03",
            "Redis caching service to store frequently accessed results",
            tags=["distributed-caching", "performance-optimization"],
        ),
    ]

    lookup_result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
        mode="lookup",
    )
    change_result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
        mode="change_over_time",
    )

    lookup_facts = [item["what"].lower() for item in lookup_result["facts_in_window"]]
    change_facts = [item["what"].lower() for item in change_result["facts_in_window"]]

    assert lookup_result["mode"] == "lookup"
    assert change_result["mode"] == "change_over_time"
    assert store.fallback_called is True
    assert not any("health checks" in fact for fact in lookup_facts)
    assert any("health checks" in fact for fact in change_facts)
    assert any("redis caching" in fact for fact in change_facts)


def test_remember_when_change_over_time_expands_sibling_facts_from_anchor_segments():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["architecture"] = [
        _make_quote("seg-1", "architecture", "July-03-2024", "architecture anchor"),
    ]
    store.segments["seg-1"] = _make_segment("seg-1", "architecture", "July-03-2024", "July 3 summary")

    anchor_fact = _make_fact(
        "fact-anchor",
        "2024-07-03",
        "Architecture updated for higher throughput",
        tags=["architecture", "performance-optimization"],
        segment_ref="seg-1",
    )
    store.fact_hits["architecture"] = [anchor_fact]
    store.segment_facts["seg-1"] = [
        anchor_fact,
        _make_fact(
            "fact-health",
            "2024-07-03",
            "Weighted round robin load balancing with health checks",
            tags=["load-balancing", "high-availability"],
            segment_ref="seg-1",
        ),
        _make_fact(
            "fact-redis",
            "2024-07-03",
            "Redis caching tier added for repeated queries",
            tags=["distributed-caching", "performance-optimization"],
            segment_ref="seg-1",
        ),
    ]

    lookup_result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
        mode="lookup",
    )
    change_result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
        mode="change_over_time",
    )

    lookup_facts = [item["what"].lower() for item in lookup_result["facts_in_window"]]
    change_facts = [item["what"].lower() for item in change_result["facts_in_window"]]

    assert not any("health checks" in fact for fact in lookup_facts)
    assert any("health checks" in fact for fact in change_facts)
    assert any("redis caching" in fact for fact in change_facts)


def test_remember_when_change_over_time_expands_linked_facts_from_anchor_facts():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    anchor_fact = _make_fact(
        "fact-anchor",
        "2024-07-11",
        "Architecture plan updated for higher throughput",
        tags=["architecture", "performance-optimization"],
        segment_ref="seg-11",
    )
    store.fact_hits["architecture"] = [anchor_fact]
    store.linked_facts = [
        LinkedFact(
            fact=_make_fact(
                "fact-monitoring",
                "2024-07-11",
                "Monitoring and alerting added to support the rollout",
                tags=["observability", "deployment"],
                segment_ref="seg-12",
            ),
            linked_from_fact_id="fact-anchor",
            relation_type="related_to",
            confidence=0.9,
        )
    ]

    lookup_result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
        mode="lookup",
    )
    change_result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=3,
        mode="change_over_time",
    )

    lookup_facts = [item["what"].lower() for item in lookup_result["facts_in_window"]]
    change_facts = [item["what"].lower() for item in change_result["facts_in_window"]]

    assert not any("monitoring and alerting" in fact for fact in lookup_facts)
    assert any("monitoring and alerting" in fact for fact in change_facts)


def test_remember_when_change_over_time_returns_raw_results_without_highlight_fields():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["architecture"] = [
        _make_quote("seg-1", "architecture", "July-03-2024", "architecture anchor"),
    ]
    store.segments["seg-1"] = _make_segment("seg-1", "architecture", "July-03-2024", "July 3 summary")

    store.fact_hits["architecture"] = [
        _make_fact(
            "fact-anchor",
            "2024-07-03",
            "Architecture updated to support higher throughput and availability",
            tags=["architecture", "performance-optimization"],
            segment_ref="seg-1",
        ),
    ]
    store.fallback_facts = [
        _make_fact(
            "fact-health",
            "2024-07-09",
            "Weighted round robin load balancing with health checks for unhealthy servers",
            tags=["load-balancing", "high-availability"],
        ),
        _make_fact(
            "fact-redis",
            "2024-07-11",
            "Redis caching tier added for repeated query traffic",
            tags=["distributed-caching", "performance-optimization"],
        ),
        _make_fact(
            "fact-cicd",
            "2024-07-24",
            "GitLab CI/CD pipeline added build, test, and deploy stages for the rollout",
            tags=["deployment", "ci-cd"],
        ),
    ]

    result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=4,
        mode="change_over_time",
    )

    assert "summary_highlights" not in result
    assert "fact_highlights" not in result
    assert "theme_overview" not in result
    fact_text = " ".join(item["what"].lower() for item in result["facts_in_window"])
    assert "health checks" in fact_text
    assert "redis caching" in fact_text
    assert "gitlab ci/cd" in fact_text


def test_remember_when_summarize_over_time_emits_ordered_milestones():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["architecture"] = [
        _make_quote("seg-1", "architecture", "July-03-2024", "architecture anchor"),
    ]
    store.segments["seg-1"] = _make_segment("seg-1", "architecture", "July-03-2024", "July 3 summary")

    store.fact_hits["architecture"] = [
        _make_fact(
            "fact-anchor",
            "2024-07-03",
            "Architecture updated to support higher throughput and availability",
            tags=["architecture", "performance-optimization"],
            segment_ref="seg-1",
        ),
    ]
    store.fallback_facts = [
        _make_fact(
            "fact-health",
            "2024-07-09",
            "Weighted round robin load balancing with health checks for unhealthy servers",
            tags=["load-balancing", "high-availability"],
        ),
        _make_fact(
            "fact-redis",
            "2024-07-11",
            "Redis caching tier added for repeated query traffic",
            tags=["distributed-caching", "performance-optimization"],
        ),
        _make_fact(
            "fact-cicd",
            "2024-07-24",
            "GitLab CI/CD pipeline added build, test, and deploy stages for the rollout",
            tags=["deployment", "ci-cd"],
        ),
    ]

    result = resolver.remember_when(
        query="system architecture performance optimization plans",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=4,
        mode="summarize_over_time",
    )

    milestones = result["ordered_milestones"]

    assert [item["date"] for item in milestones] == [
        "2024-07-03",
        "2024-07-09",
        "2024-07-11",
        "2024-07-24",
    ]
    assert [item["source"] for item in milestones] == ["segment", "fact", "fact", "fact"]
    assert "july 3 summary" in milestones[0]["point"].lower()
    assert "throughput and availability" in milestones[0]["supporting_point"].lower()
    assert "health checks" in milestones[1]["point"].lower()
    assert "redis caching" in milestones[2]["point"].lower()
    assert "deploy stages" in milestones[3]["point"].lower()


def test_ordered_milestones_prefer_segment_summary_for_shared_date():
    resolver = TemporalResolver(store=FakeStore(), search_engine=FakeSearch(), config=_make_config())

    milestones = resolver._build_ordered_milestones(
        query="error handling challenges",
        results=[
            {
                "excerpt": "Context window resizing and mismatch issues were the main challenge for this session.",
                "topic": "context-window-management",
                "session_date_normalized": "2024-11-05",
                "matched_terms": ["handling", "challenges"],
            }
        ],
        facts=[
            {
                "type": "fact",
                "what": "optimize_resizing includes try-except block with logging.error for error tracking",
                "when": "2024-11-05",
                "tags": ["error-handling-patterns"],
                "matched_terms": ["error", "handling"],
            }
        ],
        max_results=4,
    )

    assert milestones[0]["source"] == "segment"
    assert milestones[0]["theme"] == "context window management"
    assert "mismatch issues" in milestones[0]["point"].lower()
    assert "logging.error" in milestones[0]["supporting_point"].lower()


def test_remember_when_change_over_time_ordered_milestones_use_unique_dates_across_span():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    facts = [
        {
            "type": "fact",
            "what": "Initial error type handling baseline",
            "when": "2024-07-01",
            "tags": ["error-tracking"],
            "matched_terms": ["errors", "handling"],
        },
        {
            "type": "fact",
            "what": "Second milestone for error handling rollout",
            "when": "2024-07-03",
            "tags": ["error-tracking"],
            "matched_terms": ["errors", "handling", "rollout"],
        },
        {
            "type": "fact",
            "what": "Third milestone for validation failures",
            "when": "2024-07-05",
            "tags": ["validation-errors"],
            "matched_terms": ["errors"],
        },
        {
            "type": "fact",
            "what": "Fourth milestone for retry handling",
            "when": "2024-07-08",
            "tags": ["retry-mechanism"],
            "matched_terms": ["handling"],
        },
        {
            "type": "fact",
            "what": "Fifth milestone for logging and monitoring",
            "when": "2024-07-12",
            "tags": ["logging-monitoring"],
            "matched_terms": ["errors"],
        },
        {
            "type": "fact",
            "what": "Sixth milestone for recovery strategy",
            "when": "2024-07-15",
            "tags": ["recovery-strategy"],
            "matched_terms": ["rollout"],
        },
    ]

    milestones = resolver._build_ordered_milestones(
        query="errors handling rollout",
        results=[],
        facts=facts,
        max_results=4,
    )

    milestone_dates = [item["date"] for item in milestones]

    assert milestone_dates == sorted(milestone_dates)
    assert len(milestone_dates) == 4
    assert len(set(milestone_dates)) == 4
    assert milestone_dates[0] == "2024-07-01"
    assert milestone_dates[-1] == "2024-07-15"


def test_build_search_variants_preserves_short_acronyms():
    resolver = TemporalResolver(store=FakeStore(), search_engine=FakeSearch(), config=_make_config())

    variants = resolver._build_search_variants("RAG system development phases API discussions")

    texts = [text for text, _weight, _matched_terms in variants]

    assert "rag" in texts
    assert "api" in texts
    assert any("rag" in text for text in texts)
    assert any("api" in text for text in texts)
    assert not any(text == "system" for text in texts)
    assert not any(text == "development" for text in texts)


def test_remember_when_change_over_time_prefers_summary_hits_over_full_text_noise():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits.update({
        "vector search": [
            _make_quote(
                "seg-noise",
                "document-ingestion",
                "August-01-2024",
                "noisy full text mentions vector search and logging but focuses on ingestion details",
            ),
        ],
        "vector": [
            _make_quote(
                "seg-noise",
                "document-ingestion",
                "August-01-2024",
                "vector throughput notes",
            ),
        ],
        "search": [
            _make_quote(
                "seg-noise",
                "document-ingestion",
                "August-01-2024",
                "search endpoint notes",
            ),
        ],
        "logging": [
            _make_quote(
                "seg-noise",
                "document-ingestion",
                "August-01-2024",
                "logging details",
            ),
        ],
    })
    store.summary_hits.update({
        "vector search": [
            _make_segment(
                "seg-arch",
                "system-architecture",
                "August-01-2024",
                "Microservices architecture combines Elasticsearch and FAISS for high-availability search with monitoring and alerting.",
            ),
        ],
        "vector": [
            _make_segment(
                "seg-arch",
                "system-architecture",
                "August-01-2024",
                "Microservices architecture combines Elasticsearch and FAISS for high-availability search with monitoring and alerting.",
            ),
        ],
        "search": [
            _make_segment(
                "seg-arch",
                "system-architecture",
                "August-01-2024",
                "Microservices architecture combines Elasticsearch and FAISS for high-availability search with monitoring and alerting.",
            ),
        ],
        "logging": [
            _make_segment(
                "seg-arch",
                "system-architecture",
                "August-01-2024",
                "Microservices architecture combines Elasticsearch and FAISS for high-availability search with monitoring and alerting.",
            ),
        ],
    })
    store.segments["seg-noise"] = _make_segment(
        "seg-noise",
        "document-ingestion",
        "August-01-2024",
        "Noisy ingestion summary",
    )
    store.segments["seg-arch"] = _make_segment(
        "seg-arch",
        "system-architecture",
        "August-01-2024",
        "Microservices architecture combines Elasticsearch and FAISS for high-availability search with monitoring and alerting.",
    )

    result = resolver.remember_when(
        query="vector search logging capabilities improvements",
        time_range={"kind": "between_dates", "start": "2024-08-01", "end": "2024-08-31"},
        max_results=4,
        mode="change_over_time",
    )

    assert result["results"][0]["topic"] == "system-architecture"
    assert result["results"][0]["match_type"] == "summary"
    assert "Elasticsearch and FAISS" in result["results"][0]["excerpt"]


def test_remember_when_change_over_time_does_not_use_created_at_to_break_timeline_ties():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    first = _make_segment(
        "seg-first",
        "authentication-logging",
        "August-29-2024",
        "Authentication logging integrates Elasticsearch and FAISS for vector search error tracking.",
    )
    second = _make_segment(
        "seg-second",
        "jwt-authentication",
        "August-29-2024",
        "Authentication logging integrates Elasticsearch and FAISS for vector search error tracking.",
    )
    first.created_at = datetime(2024, 8, 29, tzinfo=timezone.utc)
    second.created_at = first.created_at + timedelta(minutes=5)

    store.summary_hits.update({
        "vector search": [first, second],
        "vector": [first, second],
        "search": [first, second],
        "logging": [first, second],
    })
    store.segments["seg-first"] = first
    store.segments["seg-second"] = second

    result = resolver.remember_when(
        query="vector search logging capabilities improvements",
        time_range={"kind": "between_dates", "start": "2024-08-01", "end": "2024-08-31"},
        max_results=4,
        mode="change_over_time",
    )

    assert result["results"][0]["topic"] == "authentication-logging"
    assert result["results"][0]["segment_ref"] == "seg-first"


def test_remember_when_timeline_summary_alias_maps_to_summarize_over_time():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["vector search"] = [
        _make_quote("seg-vec", "faiss-indexing", "September-20-2024", "FAISS indexing overview"),
    ]
    store.segment_hits["logging"] = [
        _make_quote("seg-log", "logging-optimization", "October-02-2024", "Logging monitoring overview"),
    ]
    store.segments["seg-vec"] = _make_segment("seg-vec", "faiss-indexing", "September-20-2024", "Vector summary")
    store.segments["seg-log"] = _make_segment("seg-log", "logging-optimization", "October-02-2024", "Logging summary")
    store.fact_hits["vector search"] = [
        _make_fact(
            "fact-faiss",
            "2024-09-20",
            "FAISS integrated with Elasticsearch for hybrid retrieval",
            tags=["faiss-indexing", "hybrid-retrieval"],
            segment_ref="seg-vec",
        ),
    ]
    store.fact_hits["logging"] = [
        _make_fact(
            "fact-alerts",
            "2024-10-02",
            "Monitoring and alerting added for the logging rollout",
            tags=["logging-optimization", "monitoring-and-alerting"],
            segment_ref="seg-log",
        ),
    ]

    result = resolver.remember_when(
        query="Can you summarize my vector search logging capabilities improvements?",
        time_range={"kind": "between_dates", "start": "2024-09-01", "end": "2024-10-22"},
        max_results=4,
        mode="timeline_summary",
    )

    fact_tags = {tag for item in result["facts_in_window"] for tag in item["tags"]}

    assert result["mode"] == "summarize_over_time"
    assert "hybrid-retrieval" in fact_tags
    assert "monitoring-and-alerting" in fact_tags
    assert "summary_highlights" not in result
    assert "fact_highlights" not in result
    assert "theme_overview" not in result

def test_remember_when_summarize_over_time_can_force_timeline_behavior():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits.update({
        "architecture": [
            _make_quote("seg-1", "architecture", "July-01-2024", "architecture baseline"),
            _make_quote("seg-3", "architecture", "July-15-2024", "architecture midpoint"),
            _make_quote("seg-5", "architecture", "July-29-2024", "architecture final"),
        ],
        "performance": [
            _make_quote("seg-2", "performance", "July-03-2024", "performance early"),
            _make_quote("seg-4", "performance", "July-20-2024", "performance late"),
        ],
    })
    for ref, tag, session, summary in [
        ("seg-1", "architecture", "July-01-2024", "July 1 summary"),
        ("seg-2", "performance", "July-03-2024", "July 3 summary"),
        ("seg-3", "architecture", "July-15-2024", "July 15 summary"),
        ("seg-4", "performance", "July-20-2024", "July 20 summary"),
        ("seg-5", "architecture", "July-29-2024", "July 29 summary"),
    ]:
        store.segments[ref] = _make_segment(ref, tag, session, summary)

    result = resolver.remember_when(
        query="architecture performance",
        time_range={"kind": "between_dates", "start": "2024-07-01", "end": "2024-07-29"},
        max_results=4,
        mode="summarize_over_time",
    )

    dates = [item["session_date_normalized"] for item in result["results"]]
    assert result["mode"] == "summarize_over_time"
    assert dates == sorted(dates)
    assert len(set(dates)) == 4


def test_remember_when_state_at_time_prioritizes_coherent_state_anchor():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    anchor_summary = (
        "User is managing a Jira sprint with 17 tasks targeting 88% completion "
        "while prioritizing caching, testing, deployment, and monitoring work."
    )
    noise_summary = (
        "User logged performance data for a resizing project and planned more "
        "logging improvements with 20% completion."
    )

    store.segment_hits["jira"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["tasks"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["sprint"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["logged"] = [
        _make_quote("seg-noise", "dynamic-context-resizing", "November-05-2024", noise_summary),
    ]
    store.segments["seg-anchor"] = _make_segment(
        "seg-anchor",
        "segmentation-optimization",
        "November-01-2024",
        anchor_summary,
    )
    store.segments["seg-noise"] = _make_segment(
        "seg-noise",
        "dynamic-context-resizing",
        "November-05-2024",
        noise_summary,
    )

    result = resolver.remember_when(
        query="Jira tasks logged sprint completion target percentage",
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-10"},
        max_results=4,
        mode="state_at_time",
    )

    assert result["mode"] == "state_at_time"
    assert result["target_date"] == "2024-11-05"
    assert result["state_anchor"]["source"] == "segment"
    assert result["state_anchor"]["date"] == "2024-11-01"
    assert result["results"][0]["state_anchor"] is True
    assert result["results"][0]["session_date_normalized"] == "2024-11-01"
    assert result["results"][0]["date_distance_days"] == -4
    assert "17 tasks" in result["results"][0]["excerpt"]
    assert "88%" in result["results"][0]["excerpt"]


def test_remember_when_state_at_time_scores_segment_summaries_not_numeric_quote_noise():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["jira"] = [
        _make_quote(
            "seg-noise",
            "jira-automation",
            "November-01-2024",
            "Jira automation errors included HTTP 401, 404, 400, and 500 status codes.",
        ),
        _make_quote(
            "seg-anchor",
            "segmentation-optimization",
            "November-01-2024",
            "Jira sprint planning discussion.",
        ),
    ]
    store.segment_hits["tasks"] = [
        _make_quote(
            "seg-noise",
            "jira-automation",
            "November-01-2024",
            "Task updates and 4 automation rules with 401 and 500 errors.",
        ),
        _make_quote(
            "seg-anchor",
            "segmentation-optimization",
            "November-01-2024",
            "Task planning for the sprint.",
        ),
    ]
    store.segment_hits["sprint"] = [
        _make_quote(
            "seg-noise",
            "jira-automation",
            "November-01-2024",
            "Sprint automation logging with 404 responses.",
        ),
        _make_quote(
            "seg-anchor",
            "segmentation-optimization",
            "November-01-2024",
            "Sprint planning for the architecture work.",
        ),
    ]
    store.segments["seg-noise"] = _make_segment(
        "seg-noise",
        "jira-automation",
        "November-01-2024",
        "User configured Jira automation rules and latency tracking uploads.",
    )
    store.segments["seg-anchor"] = _make_segment(
        "seg-anchor",
        "segmentation-optimization",
        "November-01-2024",
        "User is managing a Jira sprint with 17 tasks targeting 88% completion.",
    )

    result = resolver.remember_when(
        query="Jira tasks logged sprint completion target percentage",
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-10"},
        max_results=3,
        mode="state_at_time",
    )

    assert result["results"][0]["topic"] == "segmentation-optimization"
    assert "17 tasks" in result["results"][0]["excerpt"]
    assert "88%" in result["results"][0]["excerpt"]


def test_remember_when_state_at_time_keeps_late_query_terms_for_disambiguation():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.segment_hits["jira"] = [
        _make_quote(
            "seg-noise",
            "jira-automation",
            "November-01-2024",
            "Jira automation setup for task status updates.",
        ),
    ]
    store.segment_hits["tasks"] = [
        _make_quote(
            "seg-noise",
            "jira-automation",
            "November-01-2024",
            "Tasks updated automatically by automation rules.",
        ),
    ]
    store.segment_hits["completion target"] = [
        _make_quote(
            "seg-anchor",
            "segmentation-optimization",
            "November-01-2024",
            "Sprint completion target discussion.",
        ),
    ]
    store.segment_hits["target percentage"] = [
        _make_quote(
            "seg-anchor",
            "segmentation-optimization",
            "November-01-2024",
            "Sprint target percentage discussion.",
        ),
    ]
    store.segments["seg-noise"] = _make_segment(
        "seg-noise",
        "jira-automation",
        "November-01-2024",
        "User configured Jira automation rules for sprint tasks.",
    )
    store.segments["seg-anchor"] = _make_segment(
        "seg-anchor",
        "segmentation-optimization",
        "November-01-2024",
        "User is managing a Jira sprint with 17 tasks targeting 88% completion.",
    )

    result = resolver.remember_when(
        query="Jira tasks logged sprint completion target percentage",
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-10"},
        max_results=3,
        mode="state_at_time",
    )

    assert result["results"][0]["topic"] == "segmentation-optimization"
    assert "88%" in result["results"][0]["excerpt"]


def test_remember_when_state_at_time_prefers_value_bearing_fact_snapshot():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    store.fact_hits["tasks"] = [
        _make_fact("fact-anchor", "2024-11-01", "Sprint task count updated from 12 to 17 tasks"),
    ]
    store.fact_hits["logged"] = [
        _make_fact("fact-noise", "2024-11-05", "Logged performance data for threshold tuning"),
    ]
    store.fallback_facts = [
        _make_fact("fact-anchor", "2024-11-01", "Sprint task count updated from 12 to 17 tasks"),
        _make_fact("fact-target", "2024-11-01", "Sprint completion target set to 88%"),
        _make_fact("fact-noise", "2024-11-05", "Logged performance data for threshold tuning"),
    ]

    result = resolver.remember_when(
        query="Jira tasks logged sprint completion target percentage",
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-10"},
        max_results=3,
        mode="state_at_time",
    )

    assert result["mode"] == "state_at_time"
    assert result["target_date"] == "2024-11-05"
    assert result["state_anchor"]["source"] == "fact"
    assert result["facts_in_window"][0]["state_anchor"] is True
    assert result["facts_in_window"][0]["when"] == "2024-11-01"
    assert result["facts_in_window"][0]["date_distance_days"] == -4
    top_two = [item["what"] for item in result["facts_in_window"][:2]]
    assert any("17 tasks" in what for what in top_two)
    assert any("88%" in what for what in top_two)


def test_remember_when_state_at_time_surfaces_anchor_aligned_supporting_facts():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    anchor_summary = "User is managing a Jira sprint with 17 tasks targeting 88% completion."
    store.segment_hits["jira"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["tasks"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["completion target"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segments["seg-anchor"] = _make_segment(
        "seg-anchor",
        "segmentation-optimization",
        "November-01-2024",
        anchor_summary,
    )
    store.fact_hits["logged"] = [
        _make_fact("fact-noise", "2024-11-01", "User set up sprint tracking with 15 tasks and 90% completion rate goal"),
    ]
    store.fallback_facts = [
        _make_fact("fact-noise", "2024-11-01", "User set up sprint tracking with 15 tasks and 90% completion rate goal"),
        _make_fact("fact-tasks", "2024-11-01", "Sprint task count updated from 12 to 17 tasks"),
        _make_fact("fact-percent", "2024-11-01", "Sprint completion target set to 88%"),
    ]

    result = resolver.remember_when(
        query="Jira tasks logged sprint completion target percentage",
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-10"},
        max_results=3,
        mode="state_at_time",
    )

    top_three = [item["what"] for item in result["facts_in_window"][:3]]
    assert result["results"][0]["topic"] == "segmentation-optimization"
    assert any("17 tasks" in what for what in top_three)
    assert any("88%" in what for what in top_three)


def test_remember_when_state_at_time_separates_conflicting_value_bundles():
    store = FakeStore()
    search = FakeSearch()
    resolver = TemporalResolver(store=store, search_engine=search, config=_make_config())

    anchor_summary = (
        "User is managing a Jira sprint with 17 tasks targeting 88% completion "
        "while prioritizing caching, testing, deployment, and monitoring work."
    )
    conflict_summary = (
        "User created 14 dynamic resizing tasks in Jira and targeted 85% sprint completion."
    )
    store.segment_hits["jira"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["tasks"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["completion target"] = [
        _make_quote("seg-anchor", "segmentation-optimization", "November-01-2024", anchor_summary),
    ]
    store.segment_hits["logged"] = [
        _make_quote("seg-conflict", "query-latency-optimization", "November-05-2024", conflict_summary),
    ]
    store.segments["seg-anchor"] = _make_segment(
        "seg-anchor",
        "segmentation-optimization",
        "November-01-2024",
        anchor_summary,
    )
    store.segments["seg-conflict"] = _make_segment(
        "seg-conflict",
        "query-latency-optimization",
        "November-05-2024",
        conflict_summary,
    )
    store.fallback_facts = [
        _make_fact("fact-88", "2024-11-01", "Sprint completion target set to 88%", segment_ref="seg-anchor"),
        _make_fact("fact-17", "2024-11-01", "Sprint task count updated from 12 to 17 tasks", segment_ref="seg-anchor"),
        _make_fact("fact-90", "2024-11-01", "User set up sprint tracking with 15 tasks and 90% completion rate goal"),
        _make_fact("fact-85", "2024-11-05", "14 tasks created in Jira 9.6.0 targeting 85% sprint completion rate", segment_ref="seg-conflict"),
    ]

    result = resolver.remember_when(
        query="Jira tasks logged sprint completion target percentage",
        time_range={"kind": "between_dates", "start": "2024-11-01", "end": "2024-11-10"},
        max_results=4,
        mode="state_at_time",
    )

    assert result["chosen_state"]["effective_date"] == "2024-11-01"
    assert "17" in result["chosen_state"]["value_signals"]
    assert "88%" in result["chosen_state"]["value_signals"]
    kept_facts = [item["what"] for item in result["facts_in_window"]]
    assert any("17 tasks" in what for what in kept_facts)
    assert any("88%" in what for what in kept_facts)
    assert all("90%" not in what for what in kept_facts)
    assert all("85%" not in what for what in kept_facts)
    kept_topics = [item["topic"] for item in result["results"]]
    assert "query-latency-optimization" not in kept_topics
    conflicts = result["conflicting_candidates"]
    assert any(item.get("reason") == "conflicting_value_bundle" for item in conflicts)
    assert any("85%" in " ".join(item.get("value_signals", [])) for item in conflicts)
