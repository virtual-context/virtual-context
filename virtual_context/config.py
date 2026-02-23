"""Configuration loading, validation, and defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .types import (
    AssemblerConfig,
    CompactorConfig,
    CostTrackingConfig,
    KeywordTagConfig,
    MonitorConfig,
    PagingConfig,
    ProxyConfig,
    ProxyInstanceConfig,
    RetrieverConfig,
    SegmenterConfig,
    StorageConfig,
    StrategyConfig,
    SummarizationConfig,
    TagGeneratorConfig,
    TagPromptRule,
    TagSplittingConfig,
    VirtualContextConfig,
)

CONFIG_FILENAMES = [
    "virtual-context.yaml",
    "virtual-context.yml",
    "virtual-context.json",
    "virtualcontext.yaml",
    "virtualcontext.yml",
    "virtualcontext.json",
]


def _discover_config() -> Path | None:
    """Search CWD then parent dirs up to home for a config file."""
    cwd = Path.cwd()
    home = Path.home()
    search = cwd
    while True:
        for name in CONFIG_FILENAMES:
            candidate = search / name
            if candidate.is_file():
                return candidate
        if search == home or search == search.parent:
            break
        search = search.parent
    return None


def _parse_tag_generator(raw: dict[str, Any]) -> TagGeneratorConfig:
    """Parse the tag_generator section."""
    keyword_raw = raw.get("keyword_fallback", {})
    keyword_fallback = None
    if keyword_raw:
        keyword_fallback = KeywordTagConfig(
            tag_keywords=keyword_raw.get("tag_keywords", {}),
            tag_patterns=keyword_raw.get("tag_patterns", {}),
        )

    # Pattern overrides: user-supplied list, or None to use defaults
    pattern_kwargs: dict = {}
    temporal_patterns_raw = raw.get("temporal_patterns")
    if temporal_patterns_raw is not None:
        pattern_kwargs["temporal_patterns"] = list(temporal_patterns_raw)

    # Tag splitting config
    split_raw = raw.get("tag_splitting", {})
    tag_splitting = TagSplittingConfig(
        enabled=split_raw.get("enabled", False),
        frequency_threshold=split_raw.get("frequency_threshold", 15),
        frequency_pct_threshold=split_raw.get("frequency_pct_threshold", 0.15),
        max_splits_per_turn=split_raw.get("max_splits_per_turn", 1),
    )

    return TagGeneratorConfig(
        type=raw.get("type", "keyword"),
        provider=raw.get("provider", ""),
        model=raw.get("model", ""),
        max_tags=raw.get("max_tags", 5),
        min_tags=raw.get("min_tags", 1),
        max_tokens=raw.get("max_tokens", 1000),
        prompt_mode=raw.get("prompt_mode", "detailed"),
        keyword_fallback=keyword_fallback,
        context_lookback_pairs=raw.get("context_lookback_pairs", 5),
        context_bleed_threshold=raw.get("context_bleed_threshold", 0.1),
        disable_thinking=raw.get("disable_thinking", False),
        temporal_heuristic_enabled=raw.get("temporal_heuristic_enabled", True),
        tag_splitting=tag_splitting,
        **pattern_kwargs,
    )


def _parse_tag_rules(raw_list: list[dict[str, Any]]) -> list[TagPromptRule]:
    """Parse the tag_rules section."""
    rules = []
    for entry in raw_list:
        rules.append(TagPromptRule(
            match=entry.get("match", "*"),
            ttl_days=entry.get("ttl_days"),
            priority=entry.get("priority", 5),
            summary_prompt=entry.get("summary_prompt"),
        ))
    return rules


def _parse_strategy_configs(raw: dict[str, Any]) -> dict[str, StrategyConfig]:
    """Parse strategy_config section."""
    configs = {}
    for name, conf in raw.items():
        configs[name] = StrategyConfig(
            min_overlap=conf.get("min_overlap", 1),
            max_results=conf.get("max_results", 10),
            max_budget_fraction=conf.get("max_budget_fraction", 0.25),
            include_related=conf.get("include_related", True),
        )
    if "default" not in configs:
        configs["default"] = StrategyConfig()
    return configs


def _build_config(raw: dict[str, Any]) -> VirtualContextConfig:
    """Build a VirtualContextConfig from a raw dict."""
    # Tag generator
    tag_gen_raw = raw.get("tag_generator", {})
    tag_generator = _parse_tag_generator(tag_gen_raw)

    # Tag rules
    tag_rules = _parse_tag_rules(raw.get("tag_rules", []))

    # Compaction / monitor
    compaction = raw.get("compaction", {})
    monitor_config = MonitorConfig(
        context_window=raw.get("context_window", 120_000),
        soft_threshold=compaction.get("soft_threshold", 0.70),
        hard_threshold=compaction.get("hard_threshold", 0.85),
        protected_recent_turns=compaction.get("protected_recent_turns", 6),
    )
    segmenter_config = SegmenterConfig()
    compactor_config = CompactorConfig(
        summary_ratio=compaction.get("summary_ratio", 0.15),
        min_summary_tokens=compaction.get("min_summary_tokens", 200),
        max_summary_tokens=compaction.get("max_summary_tokens", 2000),
        max_concurrent_summaries=compaction.get("max_concurrent_summaries", 4),
        overflow_buffer=compaction.get("overflow_buffer", 1.2),
        llm_token_overhead=compaction.get("llm_token_overhead", 800),
    )

    # Summarization
    summ_raw = raw.get("summarization", {})
    summarization = SummarizationConfig(
        provider=summ_raw.get("provider", "ollama"),
        model=summ_raw.get("model", "qwen3:4b-instruct-2507-fp16"),
        max_tokens=summ_raw.get("max_tokens", 1000),
        temperature=summ_raw.get("temperature", 0.3),
    )

    # Storage
    storage_raw = raw.get("storage", {})
    backend = storage_raw.get("backend", "sqlite")
    fs_raw = storage_raw.get("filesystem", {})
    sqlite_raw = storage_raw.get("sqlite", {})
    storage_root = raw.get("storage_root", ".virtualcontext")
    storage_config = StorageConfig(
        backend=backend,
        root=fs_raw.get("root", storage_root + "/store"),
        sqlite_path=sqlite_raw.get("path", storage_root + "/store.db"),
    )

    # Assembly
    assembly_raw = raw.get("assembly", {})
    assembler_config = AssemblerConfig(
        core_context_max_tokens=assembly_raw.get("core_context_max_tokens", 18_000),
        tag_context_max_tokens=assembly_raw.get("tag_context_max_tokens", 30_000),
        core_files=assembly_raw.get("core_files", []),
        recent_turns_always_included=assembly_raw.get("recent_turns_always_included", 3),
        context_hint_enabled=assembly_raw.get("context_hint_enabled", True),
        context_hint_max_tokens=assembly_raw.get("context_hint_max_tokens", 500),
    )

    # Retrieval
    retrieval_raw = raw.get("retrieval", {})
    strategy_raw = retrieval_raw.get("strategy_config", {})
    strategy_configs = _parse_strategy_configs(strategy_raw)
    retriever_config = RetrieverConfig(
        skip_active_tags=retrieval_raw.get("skip_active_tags", True),
        active_tag_lookback=retrieval_raw.get("active_tag_lookback", 4),
        tag_context_max_tokens=assembler_config.tag_context_max_tokens,
        strategy_configs=strategy_configs,
        anchorless_lookback=retrieval_raw.get("anchorless_lookback", 6),
        inbound_tagger_type=retrieval_raw.get("inbound_tagger_type", "embedding"),
        embedding_model=retrieval_raw.get("embedding_model", "all-MiniLM-L6-v2"),
        embedding_threshold=retrieval_raw.get("embedding_threshold", 0.3),
    )

    # Cost tracking
    cost_raw = raw.get("cost_tracking", {})
    cost_tracking = CostTrackingConfig(
        enabled=cost_raw.get("enabled", False),
        pricing=cost_raw.get("pricing", {}),
    )

    # Proxy settings
    proxy_raw = raw.get("proxy", {})
    instances_raw = proxy_raw.get("instances", [])
    instances = [
        ProxyInstanceConfig(
            port=inst.get("port", 5757),
            upstream=inst.get("upstream", ""),
            label=inst.get("label", ""),
            host=inst.get("host", "127.0.0.1"),
            config=inst.get("config", ""),
        )
        for inst in instances_raw
    ]
    proxy_config = ProxyConfig(
        request_log_dir=proxy_raw.get(
            "request_log_dir",
            os.path.join(storage_root, "request_log"),
        ),
        request_log_max_files=proxy_raw.get("request_log_max_files", 50),
        upstream_context_limit=proxy_raw.get("upstream_context_limit", 200_000),
        instances=instances,
    )

    # Paging settings
    paging_raw = raw.get("paging", {})
    paging_config = PagingConfig(
        enabled=paging_raw.get("enabled", False),
        autonomous_models=paging_raw.get("autonomous_models", [
            "opus", "sonnet", "gpt-4", "gpt-4o",
        ]),
        auto_promote=paging_raw.get("auto_promote", True),
        auto_evict=paging_raw.get("auto_evict", True),
    )

    cfg = VirtualContextConfig(
        version=raw.get("version", "0.2"),
        storage_root=storage_root,
        context_window=raw.get("context_window", 120_000),
        token_counter=raw.get("token_counter", "estimate"),
        tag_generator=tag_generator,
        tag_rules=tag_rules,
        monitor=monitor_config,
        segmenter=segmenter_config,
        compactor=compactor_config,
        retriever=retriever_config,
        assembler=assembler_config,
        summarization=summarization,
        storage=storage_config,
        cost_tracking=cost_tracking,
        paging=paging_config,
        proxy=proxy_config,
        providers=raw.get("providers", {}),
    )
    if "session_id" in raw:
        cfg.session_id = raw["session_id"]
    return cfg


def validate_config(config: VirtualContextConfig) -> list[str]:
    """Validate a config. Returns list of error strings (empty = valid)."""
    errors: list[str] = []

    # Tag generator validation
    if config.tag_generator.type not in ("llm", "keyword"):
        errors.append(
            f"tag_generator.type must be 'llm' or 'keyword', "
            f"got '{config.tag_generator.type}'"
        )

    if config.tag_generator.type == "llm":
        if not config.tag_generator.provider:
            errors.append("tag_generator.provider is required when type is 'llm'")
        if not config.tag_generator.model:
            errors.append("tag_generator.model is required when type is 'llm'")

    if config.tag_generator.max_tags < config.tag_generator.min_tags:
        errors.append(
            f"tag_generator.max_tags ({config.tag_generator.max_tags}) must be >= "
            f"min_tags ({config.tag_generator.min_tags})"
        )

    # Monitor thresholds
    if config.monitor.soft_threshold >= config.monitor.hard_threshold:
        errors.append(
            f"soft_threshold ({config.monitor.soft_threshold}) must be < "
            f"hard_threshold ({config.monitor.hard_threshold})"
        )

    if config.monitor.protected_recent_turns < 1:
        errors.append("protected_recent_turns must be >= 1")

    # Strategy configs
    for name, sc in config.retriever.strategy_configs.items():
        if sc.max_budget_fraction <= 0 or sc.max_budget_fraction > 1.0:
            errors.append(
                f"strategy_config[{name}].max_budget_fraction must be in (0, 1.0], "
                f"got {sc.max_budget_fraction}"
            )
        if sc.min_overlap < 1:
            errors.append(
                f"strategy_config[{name}].min_overlap must be >= 1, "
                f"got {sc.min_overlap}"
            )

    # Storage backend
    if config.storage.backend not in ("sqlite", "filesystem"):
        errors.append(
            f"storage.backend must be 'sqlite' or 'filesystem', "
            f"got '{config.storage.backend}'"
        )

    # Paging autonomous_models
    if not isinstance(config.paging.autonomous_models, list):
        errors.append(
            "paging.autonomous_models must be a list of model-name substrings"
        )

    # Proxy instances validation
    seen_ports: dict[str, int] = {}  # "host:port" -> index
    seen_labels: dict[str, int] = {}  # "label" -> index
    for i, inst in enumerate(config.proxy.instances):
        if not inst.upstream:
            errors.append(
                f"proxy.instances[{i}].upstream is required"
            )
        key = f"{inst.host}:{inst.port}"
        if key in seen_ports:
            errors.append(
                f"proxy.instances[{i}] duplicates {key} "
                f"(same as instances[{seen_ports[key]}])"
            )
        seen_ports[key] = i

        # Per-instance config file validation
        if inst.config:
            from pathlib import Path as _Path
            if not _Path(inst.config).is_file():
                errors.append(
                    f"proxy.instances[{i}].config file not found: {inst.config}"
                )

        # Label uniqueness
        if inst.label:
            if inst.label in seen_labels:
                errors.append(
                    f"proxy.instances[{i}].label '{inst.label}' duplicates "
                    f"instances[{seen_labels[inst.label]}]"
                )
            seen_labels[inst.label] = i

    # Check that summarization provider exists in providers
    if config.providers and config.summarization.provider not in config.providers:
        errors.append(
            f"Summarization provider '{config.summarization.provider}' "
            f"not found in providers section"
        )

    return errors


def load_config(
    config_path: str | Path | None = None,
    config_dict: dict | None = None,
) -> VirtualContextConfig:
    """Load config from dict, explicit path, or auto-discover."""
    if config_dict is not None:
        return _build_config(config_dict)

    if config_path is not None:
        path = Path(config_path)
    else:
        path = _discover_config()

    if path is None:
        # Return defaults
        return _build_config({})

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    if path.suffix == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text) or {}

    return _build_config(raw)
