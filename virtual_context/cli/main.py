"""CLI: virtual-context status, recall, compact, tags, config validate, init, telemetry, retrieve, transform."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

from ..config import load_config, validate_config
from ..presets import get_preset, list_presets
from ..storage.sqlite import SQLiteStore
from ..storage.filesystem import FilesystemStore


def _get_store(config_path: str | None = None):
    config = load_config(config_path)
    if config.storage.backend == "sqlite":
        return SQLiteStore(db_path=config.storage.sqlite_path), config
    return FilesystemStore(root=config.storage.root), config


def cmd_status(args):
    store, config = _get_store(args.config)
    tag_stats = store.get_all_tags()

    if not tag_stats:
        print("No stored segments yet.")
        return

    total_segments = sum(t.usage_count for t in tag_stats)
    total_full = sum(t.total_full_tokens for t in tag_stats)
    total_summary = sum(t.total_summary_tokens for t in tag_stats)

    print(f"Context Window: {config.context_window:,} tokens")
    print(f"Storage:        {config.storage.backend} ({config.storage.sqlite_path if config.storage.backend == 'sqlite' else config.storage.root})")
    print(f"Total Tags:     {len(tag_stats)}")
    print(f"Total Segments: {total_segments}")
    print(f"Full Tokens:    {total_full:,}")
    print(f"Summary Tokens: {total_summary:,}")
    if total_full > 0:
        print(f"Compression:    {total_summary / total_full:.1%}")
    print()

    print(f"{'Tag':<25} {'Segments':>8} {'Full Tokens':>12} {'Summary':>10} {'Oldest':>12} {'Newest':>12}")
    print("-" * 81)
    for t in tag_stats:
        oldest = t.oldest_segment.strftime("%Y-%m-%d") if t.oldest_segment else "n/a"
        newest = t.newest_segment.strftime("%Y-%m-%d") if t.newest_segment else "n/a"
        print(
            f"{t.tag:<25} {t.usage_count:>8} {t.total_full_tokens:>12,} "
            f"{t.total_summary_tokens:>10,} {oldest:>12} {newest:>12}"
        )


def cmd_tags(args):
    store, config = _get_store(args.config)
    tag_stats = store.get_all_tags()

    if not tag_stats:
        print("No tags yet. Compact some conversations first.")
        return

    print(f"{'Tag':<30} {'Count':>6} {'Summary Tokens':>15}")
    print("-" * 53)
    for t in tag_stats:
        print(f"{t.tag:<30} {t.usage_count:>6} {t.total_summary_tokens:>15,}")


def cmd_recall(args):
    store, config = _get_store(args.config)
    tag = args.tag
    limit = args.limit or 5

    summaries = store.get_summaries_by_tags(tags=[tag], limit=limit)

    if not summaries:
        print(f"No stored segments for tag: {tag}")
        return

    print(f"Tag: {tag} ({len(summaries)} segments)")
    print("=" * 60)

    for i, s in enumerate(summaries, 1):
        print(f"\n--- Segment {i} (ref: {s.ref[:8]}...) ---")
        print(f"Tags: {', '.join(s.tags)}")
        print(f"Created: {s.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Tokens: {s.summary_tokens} summary / {s.full_tokens} original")
        if s.metadata.entities:
            print(f"Entities: {', '.join(s.metadata.entities)}")
        if s.metadata.key_decisions:
            print(f"Decisions: {', '.join(s.metadata.key_decisions)}")
        print()
        print(s.summary)


def cmd_compact(args):
    from ..engine import VirtualContextEngine
    from ..types import Message

    engine = VirtualContextEngine(config_path=args.config)

    # Read conversation from stdin or file
    if args.input:
        text = Path(args.input).read_text()
    else:
        print("Reading conversation from stdin (Ctrl+D to end)...")
        text = sys.stdin.read()

    # Parse as JSON messages or plain text
    try:
        raw_messages = json.loads(text)
        messages = [
            Message(role=m["role"], content=m["content"])
            for m in raw_messages
        ]
    except (json.JSONDecodeError, KeyError):
        # Treat as alternating user/assistant
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        messages = []
        for i, line in enumerate(lines):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append(Message(role=role, content=line))

    if not messages:
        print("No messages to compact.")
        return

    report = engine.compact_manual(messages)
    if report:
        print(f"Compacted {report.segments_compacted} segments")
        print(f"Tokens freed: {report.tokens_freed:,}")
        print(f"Tags: {', '.join(report.tags)}")
    else:
        print("No compaction performed.")


def cmd_telemetry(args):
    from ..engine import VirtualContextEngine

    engine = VirtualContextEngine(config_path=args.config)
    ledger = engine.get_telemetry()
    total = ledger.total()
    by_comp = ledger.by_component()

    print("Session Telemetry Report")
    print("=" * 72)
    header = f"{'Component':<18} {'Calls':>6} {'Input':>10} {'Output':>10} {'Cost':>10} {'Time':>10}"
    print(header)
    print("-" * 72)

    for comp_name in sorted(by_comp):
        r = by_comp[comp_name]
        print(
            f"{comp_name:<18} {r.call_count:>6} {r.input_tokens:>10,} "
            f"{r.output_tokens:>10,} ${r.cost_usd:>9.4f} {r.duration_ms:>8.0f}ms"
        )

    print("-" * 72)
    print(
        f"{'TOTAL':<18} {total.call_count:>6} {total.input_tokens:>10,} "
        f"{total.output_tokens:>10,} ${total.cost_usd:>9.4f} {total.duration_ms:>8.0f}ms"
    )


# Legacy alias


def cmd_init(args):
    preset = get_preset(args.preset)
    if preset is None:
        available = ", ".join(p.name for p in list_presets())
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        if available:
            print(f"Available presets: {available}", file=sys.stderr)
        sys.exit(1)

    output = Path.cwd() / "virtual-context.yaml"
    if output.exists() and not args.force:
        print(f"Config file already exists: {output}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    output.write_text(preset.template)
    print(f"Created {output}")
    print(f"Preset: {preset.name} — {preset.description}")
    print()
    print("Next steps:")
    print("  1. Start your local LLM server (Ollama, llama.cpp, LM Studio, vLLM)")
    print("     or set a cloud provider in the config (anthropic, openai, gemini)")
    print("  2. Validate config:   virtual-context config validate")
    print("  3. List tags:         virtual-context tags")


def cmd_presets(args):
    action = getattr(args, "presets_action", None) or "list"

    if action == "list":
        presets = list_presets()
        if not presets:
            print("No presets registered.")
            return
        print(f"{'Name':<15} {'Description'}")
        print("-" * 60)
        for p in presets:
            print(f"{p.name:<15} {p.description}")

    elif action == "show":
        preset = get_preset(args.preset_name)
        if preset is None:
            available = ", ".join(p.name for p in list_presets())
            print(f"Unknown preset: {args.preset_name}", file=sys.stderr)
            if available:
                print(f"Available: {available}", file=sys.stderr)
            sys.exit(1)
        print(yaml.safe_dump(preset.config_dict, sort_keys=False))


def cmd_config_validate(args):
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    errors = validate_config(config)
    if errors:
        print("Config validation errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Config is valid.")
        print(f"  Tag generator: {config.tag_generator.type}")
        if config.tag_generator.type == "llm":
            print(f"  Provider: {config.tag_generator.provider} ({config.tag_generator.model})")
        print(f"  Tag rules: {len(config.tag_rules)}")
        print(f"  Context window: {config.context_window:,}")
        print(f"  Storage: {config.storage.backend}")


def cmd_aliases(args):
    from ..core.tag_canonicalizer import TagCanonicalizer

    store, config = _get_store(args.config)
    canonicalizer = TagCanonicalizer(store=store)
    canonicalizer.load()

    action = getattr(args, 'aliases_action', None) or 'list'

    if action == 'list':
        aliases = canonicalizer.get_aliases()
        if not aliases:
            print("No aliases registered.")
        else:
            for alias, canonical in sorted(aliases.items()):
                print(f"  {alias} → {canonical}")

    elif action == 'suggest':
        suggestions = canonicalizer.auto_detect_aliases()
        if not suggestions:
            print("No alias suggestions found.")
        else:
            print("Suggested aliases (by edit distance similarity):")
            for alias, canonical in suggestions:
                print(f"  {alias} → {canonical}")

    elif action == 'add':
        alias = args.alias
        canonical = args.canonical
        canonicalizer.register_alias(alias, canonical)
        print(f"Registered alias: {alias} → {canonical}")


def cmd_chat(args):
    import os

    try:
        from ..tui.app import run_chat
    except ImportError:
        print(
            "TUI dependencies not installed. Run: pip install virtual-context[tui]",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "No Anthropic API key found.\n"
            "Set ANTHROPIC_API_KEY env var or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    replay_prompts = None
    if args.replay:
        from ..tui.state import load_replay_prompts

        replay_path = Path(args.replay)
        if not replay_path.exists():
            print(f"Replay file not found: {replay_path}", file=sys.stderr)
            sys.exit(1)
        replay_prompts = load_replay_prompts(replay_path)
        if not replay_prompts:
            print(f"No prompts found in: {replay_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(replay_prompts)} prompts from {replay_path}")

    if args.headless:
        if not replay_prompts:
            print(
                "--headless requires --replay <file>",
                file=sys.stderr,
            )
            sys.exit(1)

        from ..tui.headless import HeadlessRunner

        runner = HeadlessRunner(
            config_path=args.config,
            api_key=api_key,
            model=args.model,
        )
        runner.run(replay_prompts)
        return

    run_chat(
        config_path=args.config,
        api_key=api_key,
        model=args.model,
        replay_prompts=replay_prompts,
    )


def _discover_config_path() -> Path | None:
    """Try to find an existing config file without creating one."""
    from ..config import _discover_config
    return _discover_config()


def _build_zero_config(upstream_url: str) -> dict | None:
    """Build a minimal config dict from an upstream URL for zero-config proxy mode.

    Returns None if the upstream can't be inferred (user should run onboard).
    """
    provider_name = _infer_provider_from_url(upstream_url)
    if not provider_name or provider_name not in _TAGGER_DEFAULTS:
        return None

    tag_provider, tag_model = _TAGGER_DEFAULTS[provider_name]

    # For local providers with no model specified, we can't auto-configure
    if not tag_model:
        return None

    # Build provider block
    tag_label, tag_base_url = _provider_defaults(tag_provider)
    api_key_env = _PROVIDER_KEY_ENVS.get(tag_provider, "")

    if tag_provider == "anthropic":
        provider_block = {tag_label: {"type": "anthropic"}}
        if api_key_env:
            provider_block[tag_label]["api_key_env"] = api_key_env
    elif tag_provider in ("openai", "gemini"):
        provider_block = {tag_label: {"type": "generic_openai", "base_url": tag_base_url}}
        if api_key_env:
            provider_block[tag_label]["api_key_env"] = api_key_env
    elif tag_provider == "openrouter":
        provider_block = {tag_label: {"type": "openrouter", "base_url": tag_base_url, "api_key_env": api_key_env or "OPENROUTER_API_KEY"}}
    elif tag_provider in ("local", "ollama"):
        provider_block = {tag_label: {"type": "generic_openai", "base_url": tag_base_url + "/v1"}}
    else:
        return None

    context_window = _UPSTREAM_CONTEXT_WINDOWS.get(provider_name, 200_000)

    # Write to ~/.virtualcontext/ so it persists for future runs
    vc_home = Path.home() / ".virtualcontext"
    vc_home.mkdir(parents=True, exist_ok=True)
    storage_root = str(vc_home)

    cfg = {
        "version": "0.2",
        "storage_root": storage_root,
        "context_window": context_window,
        "token_counter": "estimate",
        "tag_generator": {
            "type": "llm",
            "provider": tag_label,
            "model": tag_model,
            "max_tags": 5,
            "min_tags": 1,
        },
        "summarization": {
            "provider": tag_label,
            "model": tag_model,
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        "providers": provider_block,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": f"{storage_root}/store.db"},
        },
    }

    # Persist for future runs
    config_path = vc_home / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Auto-created config: {config_path} (tagger: {tag_provider}/{tag_model})")

    return cfg


def cmd_proxy(args):
    try:
        import uvicorn
        from ..proxy import create_app
    except ImportError:
        print("Run: pip install virtual-context[bridge]", file=sys.stderr)
        sys.exit(1)

    # Suppress CancelledError tracebacks on shutdown. Uvicorn force-cancels
    # SSE streaming responses after the graceful-shutdown timeout, which
    # triggers CancelledError inside Starlette internals. This is expected
    # and harmless — suppress the noisy traceback.
    import asyncio
    import logging as _logging

    class _SuppressCancelled(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            if record.exc_info:
                exc_type = record.exc_info[0]
                if exc_type is asyncio.CancelledError:
                    return False
            return True

    class _SuppressDashboardAccess(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "GET /dashboard/" in msg and "200" in msg:
                return False
            return True

    _logging.getLogger("uvicorn.error").addFilter(_SuppressCancelled())
    _logging.getLogger("uvicorn.access").addFilter(_SuppressDashboardAccess())

    # Check if multi-instance mode is configured
    from ..config import load_config as _load_config

    # Zero-config mode: if no config file exists and --upstream is provided,
    # auto-infer tagger provider from the upstream URL and build a config
    # in memory so `pip install && proxy --upstream` just works.
    config_path_for_proxy = args.config
    if not config_path_for_proxy and args.upstream:
        discovered = _discover_config_path()
        if discovered is None:
            config_dict = _build_zero_config(args.upstream)
            if config_dict:
                # Config was persisted to ~/.virtualcontext/config.yaml
                config_path_for_proxy = str(Path.home() / ".virtualcontext" / "config.yaml")
                _cfg = _load_config(config_dict=config_dict)
            else:
                print(
                    "Could not auto-configure tagger for this upstream. "
                    "Run: virtual-context onboard",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            _cfg = _load_config(config_path=str(discovered))
            config_path_for_proxy = str(discovered)
    else:
        _cfg = _load_config(config_path=config_path_for_proxy)
    instances = _cfg.proxy.instances

    if instances:
        # Multi-instance mode: ignore --upstream/--port/--host CLI args
        from ..proxy.multi import run_multi_instance

        print(f"Multi-instance proxy ({len(instances)} listeners):")
        asyncio.run(run_multi_instance(
            instances=instances,
            config_path=args.config,
        ))
    else:
        # Single-instance mode: --upstream is required
        if not args.upstream:
            print(
                "Error: --upstream is required in single-instance mode "
                "(or configure proxy.instances in config)",
                file=sys.stderr,
            )
            sys.exit(1)

        app = create_app(upstream=args.upstream, config_path=config_path_for_proxy)
        print(f"virtual-context proxy on {args.host}:{args.port} -> {args.upstream}")
        uvicorn.run(
            app, host=args.host, port=args.port, log_level="info",
            timeout_graceful_shutdown=2,
        )


def cmd_retrieve(args):
    from ..engine import VirtualContextEngine

    engine = VirtualContextEngine(config_path=args.config)

    active_tags = None
    if args.active_tags:
        active_tags = [t.strip() for t in args.active_tags.split(",") if t.strip()]

    result = engine.retrieve(args.message, active_tags=active_tags)

    output = {
        "tags_matched": result.tags_matched,
        "summaries": [
            {
                "ref": s.ref,
                "primary_tag": s.primary_tag,
                "tags": s.tags,
                "summary": s.summary,
                "summary_tokens": s.summary_tokens,
            }
            for s in result.summaries
        ],
        "total_tokens": result.total_tokens,
    }
    print(json.dumps(output, indent=2))


def cmd_transform(args):
    from ..engine import VirtualContextEngine

    engine = VirtualContextEngine(config_path=args.config)

    active_tags = None
    if args.active_tags:
        active_tags = [t.strip() for t in args.active_tags.split(",") if t.strip()]

    prepend_text = engine.transform(args.message, active_tags=active_tags, budget=args.budget)

    if prepend_text:
        print(prepend_text)
        sys.exit(0)
    else:
        sys.exit(2)


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{text}{suffix}: ").strip()
    if raw:
        return raw
    return default or ""


def _prompt_int(text: str, default: int, min_value: int = 1) -> int:
    while True:
        value = _prompt(text, str(default))
        try:
            parsed = int(value)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if parsed < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return parsed


def _prompt_yes_no(text: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    while True:
        raw = input(f"{text} [{default}]: ").strip().lower()
        if not raw:
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


def _prompt_choice(text: str, options: list[str], default: str | None = None) -> str:
    print(f"\n{text}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        print(f"  {i}) {opt}{marker}")
    while True:
        raw = input(f"Choice [1-{len(options)}]: ").strip()
        if not raw and default:
            return default
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}.")


_PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-opus-4-6"],
    "openai": ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o"],
    "gemini": ["gemini-2.0-flash", "gemini-2.5-flash-preview"],
    "local": ["qwen3:4b-instruct-2507-fp16", "llama3.1:8b", "mistral:7b"],
    "openrouter": ["google/gemini-2.0-flash", "qwen/qwen3-30b-a3b", "qwen/qwen3-32b"],
}

# Context window sizes for upstream providers — used to set correct defaults
_UPSTREAM_CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic": 1_000_000,
    "openai": 128_000,
    "gemini": 1_000_000,
    "local": 128_000,
    "openrouter": 200_000,
    "custom": 128_000,
}

# API key env var names per provider
_PROVIDER_KEY_ENVS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Auto-inferred tagger models per upstream provider (cheapest capable option)
_TAGGER_DEFAULTS: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic", "claude-haiku-4-5-20251001"),
    "openai": ("openai", "gpt-4.1-nano"),
    "gemini": ("gemini", "gemini-2.0-flash"),
    "openrouter": ("openrouter", "google/gemini-2.0-flash"),
    "local": ("local", ""),
}


def _infer_provider_from_url(url: str) -> str | None:
    """Infer the upstream provider name from a URL."""
    url_lower = url.lower()
    if "api.anthropic.com" in url_lower:
        return "anthropic"
    if "api.openai.com" in url_lower:
        return "openai"
    if "generativelanguage.googleapis.com" in url_lower:
        return "gemini"
    if "openrouter.ai" in url_lower:
        return "openrouter"
    if "127.0.0.1" in url_lower or "localhost" in url_lower:
        return "local"
    return None


def _prompt_tagging_provider(
    upstream_url: str | None = None,
) -> tuple[str, str]:
    """Prompt for tagging/summarization provider and model.

    If upstream_url is provided, auto-infers a default and asks for confirmation.
    """
    inferred_provider = None
    inferred_model = None

    if upstream_url:
        upstream_name = _infer_provider_from_url(upstream_url)
        if upstream_name and upstream_name in _TAGGER_DEFAULTS:
            inferred_provider, inferred_model = _TAGGER_DEFAULTS[upstream_name]

    if inferred_provider and inferred_model:
        print(f"\n  Auto-selected tagger: {inferred_provider} / {inferred_model}")
        if _prompt_yes_no("Use this for tagging and summarization?", default_yes=True):
            return inferred_provider, inferred_model
        print()  # blank line before full picker

    provider = _prompt_choice(
        "Tagging/summarization provider:",
        ["anthropic", "openai", "gemini", "local", "openrouter", "custom"],
        default=inferred_provider or "local",
    )
    models = _PROVIDER_MODELS.get(provider)
    if models:
        choices = models + ["custom model..."]
        model = _prompt_choice(f"{provider} model:", choices, default=models[0])
        if model == "custom model...":
            model = _prompt("Model name", "")
    else:
        model = _prompt("Model name", "")
    return provider, model


def _provider_defaults(provider: str) -> tuple[str, str]:
    if provider == "anthropic":
        return "anthropic", "https://api.anthropic.com"
    if provider == "openai":
        return "openai", "https://api.openai.com/v1"
    if provider == "gemini":
        return "gemini", "https://generativelanguage.googleapis.com"
    if provider == "local":
        return "local", "http://127.0.0.1:11434"
    if provider == "openrouter":
        return "openrouter", "https://openrouter.ai/api/v1"
    # Backwards compat: old configs may still say "ollama"
    if provider == "ollama":
        return "local", "http://127.0.0.1:11434"
    return "custom", "https://api.example.com"


def _write_instance_config(
    base_dir: Path, label: str, provider: str, model: str, inbound_tagger_type: str,
    *,
    context_window: int = 200_000,
    host: str = "127.0.0.1",
    port: int = 5757,
    dashboard_token: str = "",
    tag_api_key_env: str = "",
) -> str:
    """Generate a standalone YAML config for one proxy instance.

    Creates isolated storage at ``~/.virtual-context/<label>/store.db``
    and writes config to ``<base_dir>/virtual-context-proxy-<label>.yaml``.
    Returns the path to the written file.
    """
    storage_root = str(Path.home() / ".virtual-context" / label)
    provider_label, base_url = _provider_defaults(provider)
    provider_block: dict = {}
    if provider in ("local", "ollama"):
        provider_block = {provider_label: {"type": "generic_openai", "base_url": base_url + "/v1"}}
    elif provider == "anthropic":
        prov = {"type": "anthropic"}
        if tag_api_key_env:
            prov["api_key_env"] = tag_api_key_env
        provider_block = {provider_label: prov}
    elif provider == "openai":
        prov = {"type": "generic_openai", "base_url": base_url}
        if tag_api_key_env:
            prov["api_key_env"] = tag_api_key_env
        provider_block = {provider_label: prov}
    elif provider == "openrouter":
        provider_block = {provider_label: {"type": "openrouter", "base_url": base_url, "api_key_env": tag_api_key_env or "OPENROUTER_API_KEY"}}
    else:
        provider_block = {provider_label: {"type": "generic_openai", "base_url": base_url}}

    cfg: dict = {
        "version": "0.2",
        "storage_root": storage_root,
        "context_window": context_window,
        "token_counter": "estimate",
        "tag_generator": {
            "type": "llm" if provider != "keyword" else "keyword",
            "provider": provider_label,
            "model": model,
            "max_tags": 5,
            "min_tags": 1,
        },
        "summarization": {
            "provider": provider_label,
            "model": model,
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        "providers": provider_block,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": f"{storage_root}/store.db"},
        },
        "retrieval": {
            "inbound_tagger_type": inbound_tagger_type,
        },
    }

    out_path = base_dir / f"virtual-context-proxy-{label}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out_path)


def _check_provider_reachable(provider: str) -> bool:
    """Quick check if a tagging provider is reachable."""
    if provider in ("local", "ollama"):
        try:
            import urllib.request
            # Try OpenAI-compatible /v1/models endpoint (works with Ollama, llama.cpp, vLLM, LM Studio)
            urllib.request.urlopen("http://127.0.0.1:11434/v1/models", timeout=3)
            return True
        except Exception:
            try:
                # Fallback: Ollama native endpoint
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
                return True
            except Exception:
                return False
    return True  # Remote providers — can't cheaply validate without a key


def _generate_dashboard_token() -> str:
    import secrets
    return secrets.token_urlsafe(24)


def _run_instance_wizard() -> tuple[list[dict], list[str]]:
    """Interactive wizard for proxy instance setup.

    Returns:
        (instances, config_paths): instances list (dicts for YAML) and
        list of written per-instance config file paths.
    """
    # --- Upstream first (so we can auto-infer tagger) ---
    print("\n--- Upstream Provider ---")
    first_provider = _prompt_choice(
        "Upstream provider:",
        ["anthropic", "openai", "gemini", "custom"],
        default="anthropic",
    )
    _first_label, first_upstream = _provider_defaults(first_provider)
    first_upstream = _prompt("Upstream URL", first_upstream)

    # --- Tagging & Summarization (auto-inferred from upstream) ---
    print("\n--- Tagging & Summarization ---")
    tag_provider, tag_model = _prompt_tagging_provider(upstream_url=first_upstream)

    # Validate tagging provider is reachable
    if not _check_provider_reachable(tag_provider):
        print(f"\n  WARNING: {tag_provider} does not appear to be running.")
        if tag_provider in ("local", "ollama"):
            print("  Start your local LLM server (Ollama, llama.cpp, LM Studio, vLLM)")
        if not _prompt_yes_no("Continue anyway?", default_yes=False):
            sys.exit(0)

    # API key for tagging provider
    tag_api_key_env = ""
    tag_key_env = _PROVIDER_KEY_ENVS.get(tag_provider)
    if tag_key_env:
        existing = os.environ.get(tag_key_env, "")
        if existing:
            print(f"\n  Found {tag_key_env} in environment.")
            tag_api_key_env = tag_key_env
        else:
            print(f"\n  The {tag_provider} tagger needs an API key.")
            print(f"  Set {tag_key_env} in your environment before starting the proxy.")
            tag_api_key_env = tag_key_env

    inbound_tagger_type = _prompt_choice(
        "Inbound tagging mode:",
        ["embedding", "llm", "keyword"],
        default="embedding",
    )

    # Client selection — determines post-wizard instructions
    client = _prompt_choice(
        "What client will you use with this proxy?",
        ["Claude Code", "Claude Desktop", "Other / REST API"],
        default="Claude Code",
    )

    # Dashboard token
    print("\n--- Security ---")
    dashboard_token = _generate_dashboard_token()
    print(f"  Generated dashboard token: {dashboard_token}")
    print("  (protects dashboard mutating endpoints)")

    multi = _prompt_yes_no("\nConfigure multiple proxy instances?", default_yes=False)
    count = 1
    if multi:
        count = _prompt_int("How many proxy instances?", default=2, min_value=1)

    instances: list[dict] = []
    config_paths: list[str] = []
    used_ports: set[int] = set()
    base_dir = Path.home() / ".virtual-context"
    base_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        print(f"\n--- Instance {i + 1}/{count} ---")
        if i == 0:
            # First instance uses the upstream we already asked about
            provider = first_provider
            default_label, _ = _provider_defaults(provider)
            label = _prompt("Label", default_label)
            upstream = first_upstream
        else:
            provider = _prompt_choice(
                "Upstream provider:",
                ["anthropic", "openai", "gemini", "custom"],
                default="openai",
            )
            default_label, default_upstream = _provider_defaults(provider)
            label = _prompt("Label", default_label)
            upstream = _prompt("Upstream URL", default_upstream)

        # Context window — default based on upstream provider
        default_cw = _UPSTREAM_CONTEXT_WINDOWS.get(provider, 200_000)
        context_window = _prompt_int("Context window (tokens)", default=default_cw, min_value=8_000)

        default_port = 5757 + i
        while True:
            port = _prompt_int("Port", default=default_port, min_value=1)
            if port in used_ports:
                print("That port is already used in this setup.")
                continue
            used_ports.add(port)
            break

        host = _prompt("Host", "127.0.0.1")

        # Per-instance tagger override
        use_different_tagger = False
        inst_tag_provider = tag_provider
        inst_tag_model = tag_model
        inst_tag_key_env = tag_api_key_env
        if count > 1:
            use_different_tagger = _prompt_yes_no(
                f"Use a different tagger provider for '{label}'?", default_yes=False,
            )

        if use_different_tagger:
            inst_tag_provider, inst_tag_model = _prompt_tagging_provider()
            inst_tag_key_env = _PROVIDER_KEY_ENVS.get(inst_tag_provider, "")

        # Write per-instance config
        cfg_path = _write_instance_config(
            base_dir, label, inst_tag_provider, inst_tag_model, inbound_tagger_type,
            context_window=context_window,
            host=host,
            port=port,
            dashboard_token=dashboard_token,
            tag_api_key_env=inst_tag_key_env,
        )
        config_paths.append(cfg_path)

        instances.append(
            {
                "port": port,
                "upstream": upstream,
                "label": label,
                "host": host,
                "config": cfg_path,
                "context_window": context_window,
                "dashboard_token": dashboard_token,
            }
        )

    # Review screen
    print("\n--- Review ---")
    for inst in instances:
        print(f"  [{inst['label']}] {inst['host']}:{inst['port']} -> {inst['upstream']}")
        print(f"    context window: {inst['context_window']:,} tokens")
        print(f"    config: {inst['config']}")
    print()

    if not _prompt_yes_no("Proceed with this configuration?", default_yes=True):
        print("Aborted.")
        sys.exit(0)

    # Post-wizard summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    vc_home = Path.home() / ".virtual-context"
    print(f"\n  Config files:  {vc_home}/")
    print(f"  Storage:       {vc_home}/<label>/")
    print(f"  Dashboard:     http://{instances[0]['host']}:{instances[0]['port']}/dashboard")

    # API key reminders
    needed_keys = set()
    if tag_key_env and not os.environ.get(tag_key_env, ""):
        needed_keys.add(tag_key_env)
    if needed_keys:
        print("\n  API keys needed (set before starting proxy):")
        for k in sorted(needed_keys):
            print(f"    export {k}=\"your-key-here\"")

    # Client-specific instructions
    inst = instances[0]
    if "Claude Code" in client:
        print(f"\n  Claude Code setup:")
        print(f"    Set this in your environment or .claude/settings.json:")
        print(f"    ANTHROPIC_BASE_URL=http://{inst['host']}:{inst['port']}/")
        print(f"\n    Disable auto-compact in Claude Code:")
        print(f"    Type /config -> Auto-compact -> disable")
        print(f"\n    Set VC_DASHBOARD_TOKEN in your environment:")
        print(f"    export VC_DASHBOARD_TOKEN=\"{dashboard_token}\"")
    elif "Desktop" in client:
        print(f"\n  Claude Desktop setup:")
        print(f"    In Claude Desktop settings, set the API base URL to:")
        print(f"    http://{inst['host']}:{inst['port']}/")
    else:
        print(f"\n  REST API / Custom client:")
        print(f"    Proxy endpoint: http://{inst['host']}:{inst['port']}/")
        print(f"    Forward your LLM API requests through this proxy.")

    print(f"\n  Logs:")
    print(f"    Proxy: tail -f ~/Library/Logs/virtual-context.log")
    print(f"    Errors: tail -f ~/Library/Logs/virtual-context.err.log")
    print()

    return instances, config_paths


def _apply_proxy_instances(config_path: Path, instances: list[dict]) -> None:
    raw = yaml.safe_load(config_path.read_text()) or {}
    proxy = raw.get("proxy") or {}
    proxy["instances"] = instances
    raw["proxy"] = proxy
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))


def _proxy_command(config_path: Path, upstream: str | None) -> str:
    import shlex
    cmd = f'virtual-context -c {shlex.quote(str(config_path))} proxy'
    if upstream:
        cmd += f" --upstream {shlex.quote(upstream)}"
    return cmd


def _install_launchd_daemon(
    config_path: Path, upstream: str | None, start: bool,
    *, dashboard_token: str = "",
) -> None:
    from xml.sax.saxutils import escape as _xml_escape

    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / "io.virtualcontext.proxy.plist"
    cmd = _proxy_command(config_path, upstream)
    log_path = Path.home() / "Library" / "Logs" / "virtual-context.log"
    err_path = Path.home() / "Library" / "Logs" / "virtual-context.err.log"

    # Build optional EnvironmentVariables block
    env_block = ""
    if dashboard_token:
        env_block = f""" <key>EnvironmentVariables</key>
  <dict>
    <key>VC_DASHBOARD_TOKEN</key>
    <string>{_xml_escape(dashboard_token)}</string>
  </dict>
"""

    # Escape XML special characters to prevent XML injection
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>io.virtualcontext.proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>{_xml_escape(cmd)}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{_xml_escape(str(log_path))}</string>
  <key>StandardErrorPath</key>
  <string>{_xml_escape(str(err_path))}</string>
{env_block}</dict>
</plist>
"""
    plist_path.write_text(plist)
    print(f"Wrote LaunchAgent: {plist_path}")
    if not start:
        print("Start manually:")
        print(f"  launchctl load {plist_path}")
        print("  launchctl start io.virtualcontext.proxy")
        return

    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    subprocess.run(["launchctl", "start", "io.virtualcontext.proxy"], check=True)
    print("Daemon installed and started (launchd).")


def _install_systemd_user_daemon(config_path: Path, upstream: str | None, start: bool) -> None:
    if not (Path("/bin/systemctl").exists() or Path("/usr/bin/systemctl").exists()):
        print("systemd not found. Skipping daemon install.")
        return

    import shutil

    user_dir = Path.home() / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    service_path = user_dir / "virtual-context.service"

    # Resolve the executable path to avoid shell injection
    vc_exe = shutil.which("virtual-context") or "virtual-context"
    exec_parts = [vc_exe, "-c", str(config_path), "proxy"]
    if upstream:
        exec_parts.extend(["--upstream", upstream])
    exec_start = " ".join(exec_parts)

    service_text = f"""[Unit]
Description=virtual-context proxy
After=network-online.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    service_path.write_text(service_text)
    print(f"Wrote systemd user unit: {service_path}")
    if not start:
        print("Start manually:")
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable --now virtual-context")
        return

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "virtual-context"], check=True)
    print("Daemon installed and started (systemd --user).")


def _install_windows_task_daemon(config_path: Path, upstream: str | None, start: bool) -> None:
    import base64

    task_name = "virtual-context-proxy"
    cmd = _proxy_command(config_path, upstream)
    # Encode the command as base64 UTF-16LE for PowerShell -EncodedCommand
    # This prevents command injection through special characters
    encoded = base64.b64encode(cmd.encode("utf-16-le")).decode("ascii")
    task_cmd = (
        f'powershell.exe -NoProfile -WindowStyle Hidden -EncodedCommand {encoded}'
    )
    subprocess.run(
        [
            "schtasks",
            "/create",
            "/f",
            "/sc",
            "ONLOGON",
            "/tn",
            task_name,
            "/tr",
            task_cmd,
        ],
        check=True,
    )
    print(f"Scheduled task installed: {task_name}")
    if not start:
        print(f"Start manually: schtasks /run /tn {task_name}")
        return
    subprocess.run(["schtasks", "/run", "/tn", task_name], check=True)
    print("Daemon installed and started (Task Scheduler).")


def cmd_onboard(args):
    config_path = Path(args.config) if args.config else (Path.cwd() / "virtual-context.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        preset = get_preset(args.preset)
        if preset is None:
            available = ", ".join(p.name for p in list_presets())
            print(f"Unknown preset: {args.preset}", file=sys.stderr)
            if available:
                print(f"Available presets: {available}", file=sys.stderr)
            sys.exit(1)
        config_path.write_text(preset.template)
        print(f"Created config: {config_path}")
    else:
        print(f"Using existing config: {config_path}")

    try:
        config = load_config(str(config_path))
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    errors = validate_config(config)
    if errors:
        print("Config validation errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Config is valid.")

    daemon_requested = bool(args.install_daemon)
    # Wizard runs by default in a TTY; --no-wizard disables it
    no_wizard = getattr(args, "no_wizard", False)
    interactive = bool(not no_wizard and sys.stdin.isatty())
    selected_upstream = args.upstream

    _wizard_dashboard_token = ""
    if interactive:
        instances, _config_paths = _run_instance_wizard()
        _apply_proxy_instances(config_path, instances)
        print(f"Updated {config_path} with {len(instances)} proxy instance(s).")
        # In wizard mode, use the first instance's config for daemon install
        if _config_paths:
            config_path = Path(_config_paths[0])
        # Extract upstream and dashboard token from first instance
        if instances:
            selected_upstream = instances[0].get("upstream")
            _wizard_dashboard_token = instances[0].get("dashboard_token", "")
        else:
            selected_upstream = None
        if not daemon_requested:
            daemon_requested = _prompt_yes_no("Install daemon/service now?", default_yes=True)

    if not daemon_requested:
        print("Onboarding complete.")
        print("Start proxy manually:")
        if selected_upstream:
            print(f"  virtual-context -c {config_path} proxy --upstream {selected_upstream}")
        else:
            print(f"  virtual-context -c {config_path} proxy")
        return

    system = platform.system().lower()
    try:
        if system == "darwin":
            _install_launchd_daemon(
                config_path, selected_upstream, start=not args.no_start,
                dashboard_token=_wizard_dashboard_token,
            )
        elif system == "linux":
            _install_systemd_user_daemon(config_path, selected_upstream, start=not args.no_start)
        elif system == "windows":
            _install_windows_task_daemon(config_path, selected_upstream, start=not args.no_start)
        else:
            print(f"Unsupported platform for daemon install: {system}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Daemon installation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Onboarding complete.")


def _daemon_platform() -> str:
    return platform.system().lower()


def _daemon_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "io.virtualcontext.proxy.plist"


def _daemon_systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "virtual-context.service"


def _resolve_daemon_config(args) -> Path:
    """Resolve the config path for daemon install, creating one if missing.

    Priority: explicit ``-c`` flag → ``~/.virtualcontext/config.yaml`` (auto-created
    from agentic preset with absolute storage paths so the daemon works from any CWD).

    If no config exists and stdin is a TTY, prompts the user to run onboard first.
    """
    if hasattr(args, "config") and args.config:
        return Path(args.config)

    import yaml

    home_vc = Path.home() / ".virtualcontext"
    home_vc.mkdir(exist_ok=True)
    config_path = home_vc / "config.yaml"
    if not config_path.exists():
        if sys.stdin.isatty():
            print("No config found. Running onboard setup first...")
            print("  (run 'virtual-context onboard' for the full guided setup)\n")
        preset = get_preset("agentic")
        cfg = yaml.safe_load(preset.template)
        cfg["storage_root"] = str(home_vc)
        cfg["storage"]["sqlite"]["path"] = str(home_vc / "store.db")
        config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"Created config: {config_path} (preset: agentic)")
    return config_path


def cmd_daemon(args):
    import time

    action = args.daemon_action
    system = _daemon_platform()

    # Handle install separately — it creates the daemon definition
    if action == "install":
        config_path = _resolve_daemon_config(args)
        upstream = getattr(args, "upstream", None)
        start = not getattr(args, "no_start", False)
        try:
            if system == "darwin":
                _install_launchd_daemon(config_path, upstream, start=start)
            elif system == "linux":
                _install_systemd_user_daemon(config_path, upstream, start=start)
            elif system == "windows":
                _install_windows_task_daemon(config_path, upstream, start=start)
            else:
                print(f"Unsupported platform for daemon install: {system}", file=sys.stderr)
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Daemon installation failed: {e}", file=sys.stderr)
            sys.exit(1)
        return

    _not_installed_msg = "Run 'virtual-context daemon install' first."

    try:
        if system == "darwin":
            label = "io.virtualcontext.proxy"
            plist_path = _daemon_plist_path()

            if action != "uninstall" and not plist_path.exists():
                print(f"Daemon not installed (no plist at {plist_path}).")
                print(_not_installed_msg)
                sys.exit(1)

            if action == "status":
                result = subprocess.run(
                    ["launchctl", "list", label],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print("Daemon is not loaded.")
            elif action == "start":
                subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=True)
                subprocess.run(["launchctl", "start", label], check=True)
                print("Daemon started (launchd).")
            elif action == "stop":
                subprocess.run(["launchctl", "stop", label], check=False)
                subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
                print("Daemon stopped (launchd).")
            elif action == "restart":
                subprocess.run(["launchctl", "stop", label], check=False)
                subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
                time.sleep(1)
                subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=True)
                subprocess.run(["launchctl", "start", label], check=True)
                print("Daemon restarted (launchd).")
            elif action == "uninstall":
                subprocess.run(["launchctl", "stop", label], check=False)
                subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
                if plist_path.exists():
                    plist_path.unlink()
                    print(f"Removed {plist_path}")
                else:
                    print("Daemon was not installed.")

        elif system == "linux":
            unit = "virtual-context"
            unit_path = _daemon_systemd_unit_path()

            if action != "uninstall" and not unit_path.exists():
                print(f"Daemon not installed (no unit at {unit_path}).")
                print(_not_installed_msg)
                sys.exit(1)

            if action == "status":
                subprocess.run(["systemctl", "--user", "status", unit, "--no-pager"], check=False)
            elif action == "start":
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                subprocess.run(["systemctl", "--user", "enable", "--now", unit], check=True)
                print("Daemon started (systemd).")
            elif action == "stop":
                subprocess.run(["systemctl", "--user", "stop", unit], check=False)
                print("Daemon stopped (systemd).")
            elif action == "restart":
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                subprocess.run(["systemctl", "--user", "restart", unit], check=True)
                print("Daemon restarted (systemd).")
            elif action == "uninstall":
                subprocess.run(["systemctl", "--user", "disable", "--now", unit], check=False)
                if unit_path.exists():
                    unit_path.unlink()
                    print(f"Removed {unit_path}")
                else:
                    print("Daemon was not installed.")
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)

        elif system == "windows":
            task = "virtual-context-proxy"

            if action != "uninstall":
                probe = subprocess.run(
                    ["schtasks", "/query", "/tn", task],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    print(f"Daemon not installed (no scheduled task '{task}').")
                    print(_not_installed_msg)
                    sys.exit(1)

            if action == "status":
                subprocess.run(["schtasks", "/query", "/tn", task, "/v", "/fo", "LIST"], check=False)
            elif action == "start":
                subprocess.run(["schtasks", "/run", "/tn", task], check=True)
                print("Daemon started (Task Scheduler).")
            elif action == "stop":
                subprocess.run(["schtasks", "/end", "/tn", task], check=False)
                print("Daemon stopped (Task Scheduler).")
            elif action == "restart":
                subprocess.run(["schtasks", "/end", "/tn", task], check=False)
                time.sleep(1)
                subprocess.run(["schtasks", "/run", "/tn", task], check=True)
                print("Daemon restarted (Task Scheduler).")
            elif action == "uninstall":
                subprocess.run(["schtasks", "/delete", "/tn", task, "/f"], check=False)
                print(f"Removed scheduled task '{task}'.")

        else:
            print(f"Unsupported platform: {system}", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Daemon action failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="virtual-context",
        description="Virtual memory for LLM conversation context management",
    )
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__import__('virtual_context').__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # status
    subparsers.add_parser("status", help="Show tag stats and token usage")

    # tags
    subparsers.add_parser("tags", help="List all tags in the store")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Recall context by tag")
    recall_parser.add_argument("tag", help="Tag to recall")
    recall_parser.add_argument("--limit", "-n", type=int, default=5, help="Max segments")

    # compact
    compact_parser = subparsers.add_parser("compact", help="Trigger manual compaction")
    compact_parser.add_argument("--input", "-i", help="Input file (JSON messages)")

    # init
    init_parser = subparsers.add_parser("init", help="Generate config from a preset")
    init_parser.add_argument("preset", help="Preset name (e.g. 'coding')")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")

    # onboard
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Guided setup: create/validate config, optionally install daemon",
    )
    onboard_parser.add_argument(
        "--preset",
        default="agentic",
        help="Preset used when creating a new config (default: agentic)",
    )
    onboard_parser.add_argument(
        "--install-daemon",
        action="store_true",
        help="Install OS daemon/service for proxy",
    )
    onboard_parser.add_argument(
        "--no-start",
        action="store_true",
        help="Install daemon but do not start it",
    )
    onboard_parser.add_argument(
        "--upstream",
        "-u",
        default=None,
        help="Optional upstream URL for single-instance proxy mode",
    )
    onboard_parser.add_argument(
        "--wizard",
        action="store_true",
        help="(deprecated, wizard now runs by default in TTY)",
    )
    onboard_parser.add_argument(
        "--no-wizard",
        action="store_true",
        help="Skip interactive wizard, use preset defaults",
    )

    # telemetry (cost-report is a deprecated alias)
    subparsers.add_parser("telemetry", help="Show conversation telemetry report")
    subparsers.add_parser("cost-report", help="Show conversation cost report (alias for telemetry)")

    # retrieve
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve context for a message")
    retrieve_parser.add_argument("--message", "-m", required=True, help="Inbound message")
    retrieve_parser.add_argument("--active-tags", help="Comma-separated currently active tags")

    # transform
    transform_parser = subparsers.add_parser("transform", help="Retrieve + assemble context block")
    transform_parser.add_argument("--message", "-m", required=True, help="Inbound message")
    transform_parser.add_argument("--active-tags", help="Comma-separated currently active tags")
    transform_parser.add_argument("--budget", type=int, help="Token budget override")

    # aliases
    aliases_parser = subparsers.add_parser("aliases", help="Manage tag aliases")
    aliases_sub = aliases_parser.add_subparsers(dest="aliases_action")

    # aliases list (default)
    aliases_sub.add_parser("list", help="Show all aliases")

    # aliases suggest
    aliases_sub.add_parser("suggest", help="Auto-detect potential aliases")

    # aliases add
    aliases_add_parser = aliases_sub.add_parser("add", help="Register a tag alias")
    aliases_add_parser.add_argument("alias", help="Alias tag name")
    aliases_add_parser.add_argument("canonical", help="Canonical tag name")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive TUI chat with virtual-context")
    chat_parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Anthropic model")
    chat_parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    chat_parser.add_argument(
        "--replay",
        metavar="FILE",
        help="Replay prompts from a vc-session.json or a text file (one prompt per line)",
    )
    chat_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run replay without TUI (requires --replay)",
    )

    # proxy
    proxy_parser = subparsers.add_parser("proxy", help="Start HTTP proxy for LLM enrichment")
    proxy_parser.add_argument(
        "--upstream", "-u", default=None,
        help="Upstream provider URL (e.g., https://api.anthropic.com). "
             "Required for single-instance mode; ignored when proxy.instances is configured.",
    )
    proxy_parser.add_argument("--port", "-p", type=int, default=5757)
    proxy_parser.add_argument("--host", default="127.0.0.1")

    # presets
    presets_parser = subparsers.add_parser("presets", help="List or inspect config presets")
    presets_sub = presets_parser.add_subparsers(dest="presets_action")
    presets_sub.add_parser("list", help="List all available presets")
    presets_show_parser = presets_sub.add_parser("show", help="Show a preset's config as YAML")
    presets_show_parser.add_argument("preset_name", help="Preset name to show")

    # daemon
    daemon_parser = subparsers.add_parser("daemon", help="Manage proxy daemon/service")
    daemon_parser.add_argument(
        "daemon_action",
        choices=["install", "status", "start", "stop", "restart", "uninstall"],
        help="Daemon action",
    )
    daemon_parser.add_argument(
        "--upstream",
        help="Upstream LLM API base URL (for install)",
    )
    daemon_parser.add_argument(
        "--no-start",
        action="store_true",
        help="Install daemon but do not start it (for install)",
    )

    # import
    import_parser = subparsers.add_parser("import", help="Import conversation history from exports")
    import_parser.add_argument(
        "--provider", "-p",
        required=True,
        choices=["chatgpt", "claude", "devin", "grok"],
        help="Export provider format",
    )
    import_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file path",
    )

    # config validate
    config_parser = subparsers.add_parser("config", help="Config operations")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("validate", help="Validate config file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "onboard":
        cmd_onboard(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "tags":
        cmd_tags(args)
    elif args.command == "recall":
        cmd_recall(args)
    elif args.command == "compact":
        cmd_compact(args)
    elif args.command in ("telemetry", "cost-report"):
        cmd_telemetry(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "transform":
        cmd_transform(args)
    elif args.command == "aliases":
        cmd_aliases(args)
    elif args.command == "proxy":
        cmd_proxy(args)
    elif args.command == "presets":
        cmd_presets(args)
    elif args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "import":
        from virtual_context.cli.import_cmd import run_import
        sys.exit(run_import(args))
    elif args.command == "config":
        if args.config_command == "validate":
            cmd_config_validate(args)
        else:
            print("Usage: virtual-context config validate")
            sys.exit(1)


if __name__ == "__main__":
    main()
