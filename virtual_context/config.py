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
    DomainDef,
    MonitorConfig,
    RetrieverConfig,
    SegmenterConfig,
    StorageConfig,
    SummarizationConfig,
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

DEFAULT_GENERAL_DOMAIN = DomainDef(
    name="_general",
    description="General conversation",
    priority=1,
    retrieval_limit=2,
    retrieval_max_tokens=2000,
    ttl_days=14,
)


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


def _parse_domain(name: str, raw: dict[str, Any]) -> DomainDef:
    return DomainDef(
        name=name,
        description=raw.get("description", ""),
        keywords=raw.get("keywords", []),
        patterns=raw.get("patterns", []),
        priority=raw.get("priority", 5),
        summary_prompt=raw.get("summary_prompt"),
        retrieval_limit=raw.get("retrieval_limit", 3),
        retrieval_max_tokens=raw.get("retrieval_max_tokens", 5000),
        ttl_days=raw.get("ttl_days"),
    )


def _build_config(raw: dict[str, Any]) -> VirtualContextConfig:
    """Build a VirtualContextConfig from a raw dict."""
    # Domains
    domains: dict[str, DomainDef] = {}
    for name, dconf in raw.get("domains", {}).items():
        domains[name] = _parse_domain(name, dconf if isinstance(dconf, dict) else {})
    if "_general" not in domains:
        domains["_general"] = DEFAULT_GENERAL_DOMAIN

    # Classifier pipeline
    classifier_raw = raw.get("classifier", {})
    pipeline = classifier_raw.get("pipeline", [{"type": "keyword"}])
    min_confidence = classifier_raw.get("min_confidence", 0.3)

    # Compaction / monitor
    compaction = raw.get("compaction", {})
    monitor_config = MonitorConfig(
        context_window=raw.get("context_window", 120_000),
        soft_threshold=compaction.get("soft_threshold", 0.70),
        hard_threshold=compaction.get("hard_threshold", 0.85),
        protected_recent_turns=compaction.get("protected_recent_turns", 6),
    )
    segmenter_config = SegmenterConfig(min_confidence=min_confidence)
    compactor_config = CompactorConfig(
        summary_ratio=compaction.get("summary_ratio", 0.15),
        min_summary_tokens=compaction.get("min_summary_tokens", 200),
        max_summary_tokens=compaction.get("max_summary_tokens", 2000),
        max_concurrent_summaries=compaction.get("max_concurrent_summaries", 4),
        overflow_buffer=compaction.get("overflow_buffer", 1.2),
    )

    # Summarization
    summ_raw = raw.get("summarization", {})
    summarization = SummarizationConfig(
        provider=summ_raw.get("provider", "anthropic"),
        model=summ_raw.get("model", "claude-haiku-4-5"),
        max_tokens=summ_raw.get("max_tokens", 1000),
        temperature=summ_raw.get("temperature", 0.3),
    )

    # Storage
    storage_raw = raw.get("storage", {})
    backend = storage_raw.get("backend", "filesystem")
    fs_raw = storage_raw.get("filesystem", {})
    storage_config = StorageConfig(
        backend=backend,
        root=fs_raw.get("root", raw.get("storage_root", ".virtualcontext") + "/store"),
    )

    # Assembly
    assembly_raw = raw.get("assembly", {})
    assembler_config = AssemblerConfig(
        core_context_max_tokens=assembly_raw.get("core_context_max_tokens", 18_000),
        domain_context_max_tokens=assembly_raw.get("domain_context_max_tokens", 30_000),
        core_files=assembly_raw.get("core_files", []),
    )

    # Retrieval
    retrieval_raw = raw.get("retrieval", {})
    retriever_config = RetrieverConfig(
        deep_retrieve_threshold=retrieval_raw.get("deep_retrieve_threshold", 0.8),
        skip_active_domains=retrieval_raw.get("skip_active_domains", True),
        active_domain_lookback=retrieval_raw.get("active_domain_lookback", 4),
        domain_context_max_tokens=assembler_config.domain_context_max_tokens,
        domains=list(domains.values()),
        velocity_fallback=retrieval_raw.get("velocity_fallback", True),
        velocity_lookback=retrieval_raw.get("velocity_lookback", 10),
        velocity_threshold=retrieval_raw.get("velocity_threshold", 0.3),
    )

    return VirtualContextConfig(
        version=raw.get("version", "1.0"),
        storage_root=raw.get("storage_root", ".virtualcontext"),
        context_window=raw.get("context_window", 120_000),
        token_counter=raw.get("token_counter", "estimate"),
        domains=domains,
        classifier_pipeline=pipeline,
        monitor=monitor_config,
        segmenter=segmenter_config,
        compactor=compactor_config,
        retriever=retriever_config,
        assembler=assembler_config,
        summarization=summarization,
        storage=storage_config,
        providers=raw.get("providers", {}),
    )


def validate_config(config: VirtualContextConfig) -> list[str]:
    """Validate a config. Returns list of error strings (empty = valid)."""
    errors: list[str] = []

    if not config.domains:
        errors.append("At least one domain must be defined")

    if not config.classifier_pipeline:
        errors.append("At least one classifier must be in the pipeline")

    if config.monitor.soft_threshold >= config.monitor.hard_threshold:
        errors.append(
            f"soft_threshold ({config.monitor.soft_threshold}) must be < "
            f"hard_threshold ({config.monitor.hard_threshold})"
        )

    if config.monitor.protected_recent_turns < 1:
        errors.append("protected_recent_turns must be >= 1")

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
