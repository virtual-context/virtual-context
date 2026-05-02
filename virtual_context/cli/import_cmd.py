"""Import command handler for virtual-context CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.import_adapters import get_adapter


def run_import(args: argparse.Namespace) -> int:
    """Execute the import command.

    Args:
        args: Parsed command-line arguments with provider, input, config.

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

    # Load config from path or auto-discover
    config_path = args.config if hasattr(args, "config") and args.config else None
    config = load_config(config_path=config_path)

    if not input_path.is_file():
        print(f"Error: Input must be a file: {input_path}", file=sys.stderr)
        return 1

    try:
        data = json.loads(input_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}", file=sys.stderr)
        return 1

    messages = adapter.extract_messages(data)
    conversation_id = adapter.extract_conversation_id(data)

    if not messages:
        print(f"Warning: No messages found in {input_path}", file=sys.stderr)
        return 0

    print(f"Importing {len(messages)} messages from {adapter.name} export...")

    # Override conversation_id on the loaded config so engine writes to the
    # correct conversation row without discarding the user's storage settings.
    config.conversation_id = conversation_id

    engine = VirtualContextEngine(config=config)

    def progress(done: int, total: int, entry: object) -> None:
        print(f"  Ingested {done}/{total} turns", end="\r")

    turns_ingested = engine.ingest_history(messages, progress_callback=progress)
    print(f"\nSuccessfully imported {turns_ingested} turns from {input_path.name}")

    return 0
