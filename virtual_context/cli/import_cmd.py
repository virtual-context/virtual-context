"""Import command handler for virtual-context CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.import_adapters import get_adapter
from virtual_context.import_adapters.loader import load_from_path


def run_import(args: argparse.Namespace) -> int:
    """Execute the import command.

    Args:
        args: Parsed command-line arguments with provider, input, config, compact.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    try:
        adapter = get_adapter(args.provider)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load base config
    config_path = args.config if hasattr(args, "config") and args.config else None
    base_config = load_config(config_path=config_path)

    total_turns = 0
    files_processed = 0
    files_skipped = 0

    for conversation_id, messages in load_from_path(input_path, adapter):
        if not messages:
            files_skipped += 1
            continue

        print(f"Importing {len(messages)} messages from conversation {conversation_id[:8]}...")

        # Override conversation_id on the loaded config so engine writes to the
        # correct conversation row without discarding the user's storage settings.
        base_config.conversation_id = conversation_id
        engine = VirtualContextEngine(config=base_config)

        def progress(done: int, total: int, entry: object) -> None:
            print(f"  Ingested {done}/{total} turns", end="\r")

        turns_ingested = engine.ingest_history(messages, progress_callback=progress)
        total_turns += turns_ingested
        files_processed += 1
        print(f"  Imported {turns_ingested} turns")

        # Optional compaction
        if getattr(args, "compact", False):
            print("  Running compaction...")
            compact_count = 0
            while engine.compact_manual(messages):
                compact_count += 1
            print(f"  Compaction complete ({compact_count} rounds)")

    print(f"\nSummary: {files_processed} files processed, {total_turns} total turns imported")
    if files_skipped:
        print(f"  ({files_skipped} files skipped due to errors)")

    return 0
