"""CLI: virtual-context status, recall, compact, tags, config validate, init, cost-report, retrieve, transform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    """Show current tag stats and token usage."""
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
    """List all tags in the store."""
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
    """Retrieve and display context for a tag."""
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
    """Trigger manual compaction."""
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


def cmd_cost_report(args):
    """Show session cost report."""
    from ..engine import VirtualContextEngine

    engine = VirtualContextEngine(config_path=args.config)
    summary = engine.get_cost_report()

    print("Session Cost Report")
    print("=" * 40)
    print(f"Retrievals:      {summary.total_retrievals}")
    print(f"Compactions:     {summary.total_compactions}")
    print(f"Tag Generations: {summary.total_tag_generations}")
    print(f"Input Tokens:    {summary.total_input_tokens:,}")
    print(f"Output Tokens:   {summary.total_output_tokens:,}")
    print(f"Est. Cost:       ${summary.estimated_cost_usd:.4f}")


def cmd_init(args):
    """Generate a config file from a preset."""
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
    print("  1. Install Ollama:    brew install ollama && ollama pull qwen3:4b-instruct-2507-fp16")
    print("  2. Start Ollama:      ollama serve")
    print("  3. Validate config:   virtual-context config validate")
    print("  4. List tags:         virtual-context tags")


def cmd_config_validate(args):
    """Validate config file."""
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
    """Manage tag aliases."""
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
    """Launch interactive TUI chat."""
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


def cmd_retrieve(args):
    """Retrieve context for a message."""
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
    """Retrieve + assemble context block."""
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


def main():
    parser = argparse.ArgumentParser(
        prog="virtual-context",
        description="Virtual memory for LLM session context management",
    )
    parser.add_argument("--config", "-c", help="Path to config file")

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

    # cost-report
    subparsers.add_parser("cost-report", help="Show session cost report")

    # retrieve
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve context for a message")
    retrieve_parser.add_argument("--message", "-m", required=True, help="Inbound message")
    retrieve_parser.add_argument("--active-tags", help="Comma-separated active tags to skip")

    # transform
    transform_parser = subparsers.add_parser("transform", help="Retrieve + assemble context block")
    transform_parser.add_argument("--message", "-m", required=True, help="Inbound message")
    transform_parser.add_argument("--active-tags", help="Comma-separated active tags to skip")
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
    elif args.command == "cost-report":
        cmd_cost_report(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "transform":
        cmd_transform(args)
    elif args.command == "aliases":
        cmd_aliases(args)
    elif args.command == "config":
        if args.config_command == "validate":
            cmd_config_validate(args)
        else:
            print("Usage: virtual-context config validate")
            sys.exit(1)


if __name__ == "__main__":
    main()
