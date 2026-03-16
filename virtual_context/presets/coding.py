"""Coding preset — tuned for software development sessions."""

from __future__ import annotations

from .base import Preset, register_preset

# ---------------------------------------------------------------------------
# Keyword fallback config (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

CODING_TAG_KEYWORDS: dict[str, list[str]] = {
    "architecture": [
        "architecture", "design", "pattern", "tradeoff", "microservice",
        "monolith", "separation of concerns", "coupling", "cohesion",
    ],
    "database": [
        "schema", "migration", "query", "SQL", "table", "column", "index",
        "foreign key", "ORM", "model", "JOIN", "SELECT",
    ],
    "auth": [
        "auth", "login", "JWT", "token", "session", "OAuth", "permission",
        "role", "RBAC", "password", "credential",
    ],
    "backend": [
        "endpoint", "route", "middleware", "controller", "service",
        "request", "response", "REST", "GraphQL", "handler",
    ],
    "frontend": [
        "component", "React", "Vue", "state", "props", "CSS", "render",
        "hook", "useEffect", "useState", "DOM", "layout",
    ],
    "debugging": [
        "bug", "error", "stack trace", "exception", "crash", "debug",
        "breakpoint", "log", "traceback", "undefined", "null",
        "test", "fixture", "mock", "assert", "coverage", "pytest", "jest",
    ],
    "infrastructure": [
        "Docker", "Kubernetes", "CI", "CD", "deploy", "pipeline", "nginx",
        "terraform", "AWS", "GCP", "container", "helm",
    ],
}

CODING_TAG_PATTERNS: dict[str, list[str]] = {
    "architecture": [
        "docs/adr/",
        "docs/design/",
        r"ARCHITECTURE\.md",
    ],
    "database": [
        r"\bCREATE TABLE\b",
        r"\bALTER TABLE\b",
        r"\bSELECT\s+.+\s+FROM\b",
        "migrations?/",
        r"\.sql\b",
    ],
    "auth": [
        "src/(auth|security|identity|iam)/",
        "middleware/(auth|session|jwt)",
    ],
    "debugging": [
        r"Traceback \(most recent",
        r"at .+\(\S+:\d+:\d+\)",
        r"Error: .+ at ",
        "panic: ",
        "FAILED tests/",
        "tests?/",
        "__tests__/",
        r"\.(test|spec)\.(ts|js|py|rb)\b",
        r"test_[a-z_]+\.py\b",
        r"conftest\.py",
    ],
    "infrastructure": [
        "Dockerfile",
        "docker-compose",
        r"\.github/workflows/",
        r"\.gitlab-ci",
        "terraform/",
    ],
}

# ---------------------------------------------------------------------------
# Tag rules
# ---------------------------------------------------------------------------

CODING_TAG_RULES: list[dict] = [
    {
        "match": "architecture*",
        "priority": 10,
        "ttl_days": None,
        "summary_prompt": (
            "Summarize architectural decisions, design patterns, and tradeoffs. "
            "Preserve ADR numbers, component names, and rationale."
        ),
    },
    {"match": "database*", "priority": 8, "ttl_days": None},
    {"match": "auth*", "priority": 8},
    {"match": "backend*", "priority": 7},
    {"match": "frontend*", "priority": 7},
    {"match": "debugging*", "priority": 7, "ttl_days": 7},
    {"match": "infrastructure*", "priority": 6},
    {"match": "*", "priority": 5, "ttl_days": 30},
]

# ---------------------------------------------------------------------------
# Full config dict
# ---------------------------------------------------------------------------

CODING_CONFIG: dict = {
    "version": "0.2",
    "storage_root": ".virtualcontext",
    "context_window": 120_000,
    "token_counter": "estimate",
    "paging": {
        "enabled": True,
        "auto_promote": True,
        "auto_evict": True,
        "max_tool_loops": 10,
    },
    "tag_generator": {
        "type": "llm",
        "provider": "ollama",
        "model": "qwen3:4b-instruct-2507-fp16",
        "max_tags": 10,
        "min_tags": 5,
        "max_tokens": 8192,
        "prompt_mode": "compact",
        "context_lookback_pairs": 5,
        "context_bleed_threshold": 0.1,
        "keyword_fallback": {
            "tag_keywords": CODING_TAG_KEYWORDS,
            "tag_patterns": CODING_TAG_PATTERNS,
        },
    },
    "tag_rules": CODING_TAG_RULES,
    "compaction": {
        "soft_threshold": 0.60,
        "hard_threshold": 0.80,
        "protected_recent_turns": 8,
        "overflow_buffer": 1.2,
        "summary_ratio": 0.15,
        "min_summary_tokens": 200,
        "max_summary_tokens": 2000,
        "max_concurrent_summaries": 4,
        "llm_token_overhead": 2000,
    },
    "summarization": {
        "provider": "ollama",
        "model": "qwen3:4b-instruct-2507-fp16",
        "max_tokens": 1000,
        "temperature": 0.3,
    },
    "providers": {
        "ollama": {
            "type": "generic_openai",
            "base_url": "http://127.0.0.1:11434/v1",
        },
    },
    "storage": {
        "backend": "sqlite",
        "sqlite": {"path": ".virtualcontext/store.db"},
    },
    "facts": {
        "graph_links": True,
    },
    "tool_output": {
        "enabled": True,
        "default_truncate_threshold": 8192,
    },
    "assembly": {
        "core_context_max_tokens": 18_000,
        "tag_context_max_tokens": 30_000,
        "facts_max_tokens": 20_000,
        "context_hint_enabled": True,
        "context_hint_max_tokens": 2000,
        "core_files": [],
    },
    "retrieval": {
        "skip_active_tags": True,
        "active_tag_lookback": 4,
        "inbound_tagger_type": "embedding",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_threshold": 0.3,
        "strategy_config": {
            "default": {
                "min_overlap": 1,
                "max_results": 10,
                "max_budget_fraction": 0.25,
                "include_related": True,
            },
        },
    },
    "telemetry": {
        "enabled": True,
        "models_file": "models.yaml",
    },
}

# ---------------------------------------------------------------------------
# YAML template
# ---------------------------------------------------------------------------

CODING_TEMPLATE = """\
# virtual-context configuration — coding preset
# Generated by: virtual-context init coding
# Docs: https://github.com/virtual-context/virtual-context
#
# Tuned for software development: more aggressive compaction (compact earlier
# to keep the window focused), more protected recent turns (code context is
# referenced more often), and coding-specific keyword fallbacks.

version: "0.2"
storage_root: ".virtualcontext"
context_window: 120000
token_counter: "estimate"

paging:
  enabled: true
  auto_promote: true
  auto_evict: true
  max_tool_loops: 10

tag_generator:
  type: "llm"
  provider: "ollama"
  model: "qwen3:4b-instruct-2507-fp16"
  max_tags: 10
  min_tags: 5
  max_tokens: 8192
  prompt_mode: "compact"
  context_lookback_pairs: 5
  context_bleed_threshold: 0.1
  keyword_fallback:
    tag_keywords:
      architecture:
        - architecture
        - design
        - pattern
        - tradeoff
        - microservice
        - monolith
      database:
        - schema
        - migration
        - query
        - SQL
        - table
        - index
        - ORM
      auth:
        - auth
        - login
        - JWT
        - token
        - session
        - OAuth
        - permission
      backend:
        - endpoint
        - route
        - middleware
        - controller
        - REST
        - GraphQL
      frontend:
        - component
        - React
        - Vue
        - state
        - props
        - CSS
        - hook
      debugging:
        - bug
        - error
        - stack trace
        - exception
        - crash
        - debug
        - traceback
        - test
        - fixture
        - mock
        - assert
        - pytest
      infrastructure:
        - Docker
        - Kubernetes
        - CI
        - CD
        - deploy
        - pipeline
        - terraform
    tag_patterns:
      architecture:
        - "docs/adr/"
        - "ARCHITECTURE\\\\.md"
      database:
        - "\\\\bCREATE TABLE\\\\b"
        - "\\\\bALTER TABLE\\\\b"
        - "\\\\bSELECT\\\\s+.+\\\\s+FROM\\\\b"
      debugging:
        - "Traceback \\\\(most recent"
        - "Error: .+ at "
        - "FAILED tests/"
        - "tests?/"
        - "conftest\\\\.py"
      infrastructure:
        - "Dockerfile"
        - "docker-compose"
        - "\\\\.github/workflows/"

tag_rules:
  - match: "architecture*"
    priority: 10
    ttl_days: null
    summary_prompt: |
      Summarize architectural decisions, design patterns, and tradeoffs.
      Preserve ADR numbers, component names, and rationale.

  - match: "database*"
    priority: 8
    ttl_days: null

  - match: "auth*"
    priority: 8

  - match: "backend*"
    priority: 7

  - match: "frontend*"
    priority: 7

  - match: "debugging*"
    priority: 7
    ttl_days: 7

  - match: "infrastructure*"
    priority: 6

  - match: "*"
    priority: 5
    ttl_days: 30

# Compaction — more aggressive than defaults for coding sessions
compaction:
  soft_threshold: 0.60          # default: 0.70 — compact earlier to keep window focused
  hard_threshold: 0.80          # default: 0.85
  protected_recent_turns: 8     # default: 6 — code context referenced more often
  overflow_buffer: 1.2
  summary_ratio: 0.15
  min_summary_tokens: 200
  max_summary_tokens: 2000
  max_concurrent_summaries: 4
  llm_token_overhead: 2000

summarization:
  provider: "ollama"
  model: "qwen3:4b-instruct-2507-fp16"
  max_tokens: 1000
  temperature: 0.3

providers:
  ollama:
    type: "generic_openai"
    base_url: "http://127.0.0.1:11434/v1"

storage:
  backend: "sqlite"
  sqlite:
    path: ".virtualcontext/store.db"

facts:
  graph_links: true

tool_output:
  enabled: true
  default_truncate_threshold: 8192

assembly:
  core_context_max_tokens: 18000
  tag_context_max_tokens: 30000
  facts_max_tokens: 20000
  context_hint_enabled: true
  context_hint_max_tokens: 2000
  core_files: []

retrieval:
  skip_active_tags: true
  active_tag_lookback: 4
  inbound_tagger_type: "embedding"
  embedding_model: "all-MiniLM-L6-v2"
  embedding_threshold: 0.3
  strategy_config:
    default:
      min_overlap: 1
      max_results: 10
      max_budget_fraction: 0.25
      include_related: true

telemetry:
  enabled: true
  models_file: "models.yaml"
"""

# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

coding_preset = Preset(
    name="coding",
    description="Coding preset — aggressive compaction, code-specific keyword fallbacks, "
                "tool output interception, fact graph enabled",
    config_dict=CODING_CONFIG,
    template=CODING_TEMPLATE,
)

register_preset(coding_preset)
