"""CLI: virtual-context status, recall, compact, domains, config validate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..config import load_config, validate_config
from ..storage.filesystem import FilesystemStore


def _get_store(config_path: str | None = None):
    config = load_config(config_path)
    return FilesystemStore(root=config.storage.root), config


async def cmd_status(args):
    """Show current domain stats and token usage."""
    store, config = _get_store(args.config)
    domains = await store.list_domains()

    if not domains:
        print("No stored segments yet.")
        return

    total_segments = sum(d.segment_count for d in domains)
    total_full = sum(d.total_full_tokens for d in domains)
    total_summary = sum(d.total_summary_tokens for d in domains)

    print(f"Context Window: {config.context_window:,} tokens")
    print(f"Storage Root:   {config.storage.root}")
    print(f"Total Segments: {total_segments}")
    print(f"Full Tokens:    {total_full:,}")
    print(f"Summary Tokens: {total_summary:,}")
    if total_full > 0:
        print(f"Compression:    {total_summary / total_full:.1%}")
    print()

    print(f"{'Domain':<20} {'Segments':>8} {'Full Tokens':>12} {'Summary':>10} {'Oldest':>12} {'Newest':>12}")
    print("-" * 76)
    for d in domains:
        oldest = d.oldest_segment.strftime("%Y-%m-%d") if d.oldest_segment else "n/a"
        newest = d.newest_segment.strftime("%Y-%m-%d") if d.newest_segment else "n/a"
        print(
            f"{d.domain:<20} {d.segment_count:>8} {d.total_full_tokens:>12,} "
            f"{d.total_summary_tokens:>10,} {oldest:>12} {newest:>12}"
        )


async def cmd_domains(args):
    """List configured domains."""
    config = load_config(args.config)
    print(f"{'Domain':<20} {'Priority':>8} {'Keywords':>8} {'Patterns':>8} {'Retrieval Limit':>15}")
    print("-" * 62)
    for name, domain in sorted(config.domains.items(), key=lambda x: x[1].priority, reverse=True):
        print(
            f"{name:<20} {domain.priority:>8} {len(domain.keywords):>8} "
            f"{len(domain.patterns):>8} {domain.retrieval_limit:>15}"
        )


async def cmd_recall(args):
    """Retrieve and display context for a domain."""
    store, config = _get_store(args.config)
    domain = args.domain
    limit = args.limit or 5

    summaries = await store.get_summaries(domain=domain, limit=limit)

    if not summaries:
        print(f"No stored segments for domain: {domain}")
        return

    print(f"Domain: {domain} ({len(summaries)} segments)")
    print("=" * 60)

    for i, s in enumerate(summaries, 1):
        print(f"\n--- Segment {i} (ref: {s.ref[:8]}...) ---")
        print(f"Created: {s.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Tokens: {s.summary_tokens} summary / {s.full_tokens} original")
        if s.metadata.entities:
            print(f"Entities: {', '.join(s.metadata.entities)}")
        if s.metadata.key_decisions:
            print(f"Decisions: {', '.join(s.metadata.key_decisions)}")
        print()
        print(s.summary)


async def cmd_compact(args):
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

    report = await engine.compact_manual(messages)
    if report:
        print(f"Compacted {report.segments_compacted} segments")
        print(f"Tokens freed: {report.tokens_freed:,}")
        print(f"Domains: {', '.join(report.domains)}")
    else:
        print("No compaction performed.")


async def cmd_config_validate(args):
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
        print(f"  Domains: {len(config.domains)}")
        print(f"  Classifiers: {len(config.classifier_pipeline)}")
        print(f"  Context window: {config.context_window:,}")
        print(f"  Storage: {config.storage.backend} ({config.storage.root})")


def main():
    parser = argparse.ArgumentParser(
        prog="virtual-context",
        description="Virtual memory for LLM session context management",
    )
    parser.add_argument("--config", "-c", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command")

    # status
    subparsers.add_parser("status", help="Show domain stats and token usage")

    # domains
    subparsers.add_parser("domains", help="List configured domains")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Recall domain context")
    recall_parser.add_argument("domain", help="Domain to recall")
    recall_parser.add_argument("--limit", "-n", type=int, default=5, help="Max segments")

    # compact
    compact_parser = subparsers.add_parser("compact", help="Trigger manual compaction")
    compact_parser.add_argument("--input", "-i", help="Input file (JSON messages)")

    # config validate
    config_parser = subparsers.add_parser("config", help="Config operations")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("validate", help="Validate config file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "domains":
        asyncio.run(cmd_domains(args))
    elif args.command == "recall":
        asyncio.run(cmd_recall(args))
    elif args.command == "compact":
        asyncio.run(cmd_compact(args))
    elif args.command == "config":
        if args.config_command == "validate":
            asyncio.run(cmd_config_validate(args))
        else:
            print("Usage: virtual-context config validate")
            sys.exit(1)


if __name__ == "__main__":
    main()
