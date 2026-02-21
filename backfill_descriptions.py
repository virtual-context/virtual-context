#!/usr/bin/env python3
"""One-time backfill: generate 1-line descriptions for existing tag summaries.

Usage:
    ANTHROPIC_API_KEY=sk-... python backfill_descriptions.py [--config CONFIG] [--dry-run]
"""
import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from virtual_context.config import load_config
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.storage.filesystem import FilesystemStore
from virtual_context.types import TagSummary


DESCRIPTION_PROMPT = """\
Given this topic tag and its full summary, generate a 1-line description
(max 20 words) that captures: who is involved, what is being discussed,
and the most distinctive detail. This will be shown as a topic label.

Tag: {tag}

Summary:
{summary}

Respond with ONLY the description text, no quotes, no JSON, no explanation."""


def build_provider(config):
    """Build the Anthropic provider from config."""
    from virtual_context.providers.anthropic import AnthropicProvider
    provider_config = config.providers.get("anthropic", {})
    api_key = os.environ.get(
        provider_config.get("api_key_env", "ANTHROPIC_API_KEY"), ""
    )
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return AnthropicProvider(
        api_key=api_key,
        model=provider_config.get("model", "claude-haiku-4-5-20251001"),
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill tag summary descriptions")
    parser.add_argument("-c", "--config", default="virtual-context-haiku-tagger.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    config = load_config(args.config)
    provider = build_provider(config)

    # Build store
    storage_cfg = config.storage
    if storage_cfg.backend == "sqlite":
        store = SQLiteStore(storage_cfg.sqlite_path)
    else:
        store = FilesystemStore(storage_cfg.root)

    summaries = store.get_all_tag_summaries()
    need_backfill = [ts for ts in summaries if not ts.description]
    print(f"Total tag summaries: {len(summaries)}")
    print(f"Need backfill: {len(need_backfill)}")

    if args.dry_run:
        for ts in need_backfill:
            print(f"  [{ts.tag}] summary: {ts.summary[:80]}...")
        return

    for i, ts in enumerate(need_backfill):
        prompt = DESCRIPTION_PROMPT.format(
            tag=ts.tag,
            summary=ts.summary[:1500],  # cap to avoid huge prompts
        )
        try:
            description = provider.complete(
                system="You generate concise topic labels. Output only the label text.",
                user=prompt,
                max_tokens=100,
            ).strip()
            # Strip quotes if LLM wraps in them
            if description.startswith('"') and description.endswith('"'):
                description = description[1:-1]
            ts.description = description
            store.save_tag_summary(ts)
            print(f"  [{i+1}/{len(need_backfill)}] {ts.tag}: {description}")
        except Exception as e:
            print(f"  [{i+1}/{len(need_backfill)}] {ts.tag}: FAILED - {e}")
        time.sleep(0.1)  # rate limit courtesy

    print(f"\nDone. Backfilled {len(need_backfill)} descriptions.")


if __name__ == "__main__":
    main()
