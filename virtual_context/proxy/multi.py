"""Multi-instance proxy: spawn N uvicorn listeners sharing one engine/store."""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from ..engine import VirtualContextEngine
from ..types import ProxyInstanceConfig
from .metrics import ProxyMetrics
from .server import create_app

logger = logging.getLogger(__name__)


async def run_multi_instance(
    instances: list[ProxyInstanceConfig],
    config_path: str | None = None,
    engine: VirtualContextEngine | None = None,
    metrics: ProxyMetrics | None = None,
) -> None:
    """Start multiple uvicorn servers sharing a single engine and metrics.

    Args:
        instances: List of ProxyInstanceConfig (port, upstream, label, host).
        config_path: Path to virtual-context config file (used to create engine if not provided).
        engine: Shared engine.  Created from config_path if None.
        metrics: Shared metrics.  Created if None.
    """
    if engine is None:
        engine = VirtualContextEngine(config_path=config_path)
    if metrics is None:
        metrics = ProxyMetrics(
            context_window=engine.config.monitor.context_window,
        )

    servers: list[uvicorn.Server] = []
    for inst in instances:
        app = create_app(
            upstream=inst.upstream,
            config_path=config_path,
            shared_engine=engine,
            shared_metrics=metrics,
            instance_label=inst.label,
        )
        config = uvicorn.Config(
            app,
            host=inst.host,
            port=inst.port,
            log_level="info",
            timeout_graceful_shutdown=2,
        )
        servers.append(uvicorn.Server(config))
        label = inst.label or inst.upstream
        print(
            f"  [{label}] {inst.host}:{inst.port} -> {inst.upstream}",
            flush=True,
        )

    print(f"Starting {len(servers)} proxy instance(s)...", flush=True)
    await asyncio.gather(*(s.serve() for s in servers))
