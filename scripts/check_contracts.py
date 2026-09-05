"""Run the small offline contract gate with bounded time and concurrency."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/test_admin_setup_contracts.py",
    "tests/test_benchmark_cache_manifest.py",
    "tests/test_context_budget_contract.py",
    "tests/test_fact_lifecycle_contracts.py",
    "tests/test_fact_audit_upgrade.py",
    "tests/test_fact_query_constraints.py",
    "tests/test_paging_transactions.py",
    "tests/test_proxy_review_invariants.py",
    "tests/test_storage_upsert_integrity.py",
    "tests/test_native_semantic_search.py",
    "tests/test_canonical_pagination_identity.py",
    "tests/test_pgvector_storage_shapes.py",
    "tests/test_relational_contracts.py",
    "tests/test_rendered_memory_paging.py",
    "tests/test_streaming_semantic_search.py",
    "tests/test_filesystem_embedding_pages.py",
    "tests/test_pending_compaction_reads.py",
    "tests/test_proxy_durable_continuation.py",
    "tests/test_proxy_handler_public_text.py",
    "tests/test_compaction_community_services.py",
    "tests/test_context_evaluation.py",
    "tests/test_context_resource_harness.py",
    "tests/test_storage_domain_contracts.py",
    "tests/test_storage_bounded_contracts.py",
    "tests/test_pending_exchange_guard_lookup.py",
    "tests/test_postgres_read_index_migration.py",
    "tests/test_sqlite_maintenance_cli.py",
    "tests/test_sqlite_reconcile_transactions.py",
]


def main() -> int:
    lock = Path("/tmp/vc-suite-slot.lock")
    try:
        lock.mkdir()
    except FileExistsError:
        print("Test slot occupied; contract check deferred.", file=sys.stderr)
        return 75
    try:
        if hasattr(os, "nice"):
            os.nice(19)
        env = os.environ.copy()
        for name in ("DATABASE_URL", "VC_TEST_POSTGRES_URL"):
            env.pop(name, None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        deadline = time.monotonic() + 85
        commands = [
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "virtual_context",
                "scripts/check_contracts.py",
                "benchmarks/longmemeval/cache_manifest.py",
                "benchmarks/context_contracts",
            ],
            [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "--timeout=25", *TESTS],
        ]
        for command in commands:
            process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
            try:
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                print("Contract check stopped; time limit or interruption.", file=sys.stderr)
                return 124
            if code:
                return code
        return 0
    finally:
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
