"""Tests for multi-instance proxy (Phase 6)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.config import load_config
from virtual_context.types import ProxyInstanceConfig
from virtual_context.proxy.multi import run_multi_instance
from virtual_context.proxy.metrics import ProxyMetrics


class TestRunMultiInstance:
    """Test run_multi_instance creates correct servers."""

    def test_creates_servers_for_each_instance(self):
        """Each instance config gets its own uvicorn.Server."""
        instances = [
            ProxyInstanceConfig(port=5757, upstream="https://api.anthropic.com", label="anthropic"),
            ProxyInstanceConfig(port=5758, upstream="https://api.openai.com/v1", label="openai"),
        ]

        created_servers = []

        class FakeServer:
            def __init__(self, config):
                self.config = config
                created_servers.append(self)

            async def serve(self):
                pass

        class FakeConfig:
            def __init__(self, app, host, port, log_level, timeout_graceful_shutdown):
                self.app = app
                self.host = host
                self.port = port

        with patch("virtual_context.proxy.multi.uvicorn") as mock_uv:
            mock_uv.Config = FakeConfig
            mock_uv.Server = FakeServer

            asyncio.run(run_multi_instance(
                instances=instances,
                config_path=None,
            ))

        assert len(created_servers) == 2
        assert created_servers[0].config.port == 5757
        assert created_servers[1].config.port == 5758

    def test_shared_engine_passed_to_all(self):
        """All instances share the same engine and metrics."""
        instances = [
            ProxyInstanceConfig(port=5757, upstream="https://api.anthropic.com", label="a"),
            ProxyInstanceConfig(port=5758, upstream="https://api.openai.com/v1", label="b"),
        ]

        apps_created = []

        def tracking_create_app(**kwargs):
            apps_created.append(kwargs)
            app = MagicMock()
            app.title = "test"
            app.state = MagicMock()
            return app

        class FakeServer:
            def __init__(self, config):
                pass

            async def serve(self):
                pass

        class FakeConfig:
            def __init__(self, app, **kw):
                pass

        with (
            patch("virtual_context.proxy.multi.create_app", side_effect=tracking_create_app),
            patch("virtual_context.proxy.multi.uvicorn") as mock_uv,
        ):
            mock_uv.Config = FakeConfig
            mock_uv.Server = FakeServer

            engine = MagicMock()
            engine.config.monitor.context_window = 120_000
            metrics = ProxyMetrics(context_window=120_000)

            asyncio.run(run_multi_instance(
                instances=instances,
                config_path=None,
                engine=engine,
                metrics=metrics,
            ))

        assert len(apps_created) == 2
        # Both got the same engine and metrics
        assert apps_created[0]["shared_engine"] is engine
        assert apps_created[1]["shared_engine"] is engine
        assert apps_created[0]["shared_metrics"] is metrics
        assert apps_created[1]["shared_metrics"] is metrics
        # Labels passed through
        assert apps_created[0]["instance_label"] == "a"
        assert apps_created[1]["instance_label"] == "b"

    def test_creates_engine_when_not_provided(self):
        """When engine is None, run_multi_instance creates one from config."""
        instances = [
            ProxyInstanceConfig(port=5757, upstream="https://api.anthropic.com", label="test"),
        ]

        class FakeServer:
            def __init__(self, config):
                pass

            async def serve(self):
                pass

        class FakeConfig:
            def __init__(self, app, **kw):
                pass

        with (
            patch("virtual_context.proxy.multi.create_app") as mock_create,
            patch("virtual_context.proxy.multi.uvicorn") as mock_uv,
        ):
            mock_create.return_value = MagicMock()
            mock_uv.Config = FakeConfig
            mock_uv.Server = FakeServer

            asyncio.run(run_multi_instance(
                instances=instances,
                config_path=None,
            ))

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("shared_engine") is not None

    def test_per_instance_config_gets_own_engine(self, tmp_path):
        """Instance with config field gets its own engine, not the shared one."""
        import yaml

        # Write a per-instance config file
        inst_cfg = {
            "version": "0.2",
            "storage_root": str(tmp_path / "inst_store"),
            "storage": {
                "backend": "sqlite",
                "sqlite": {"path": str(tmp_path / "inst.db")},
            },
        }
        inst_cfg_path = tmp_path / "inst.yaml"
        inst_cfg_path.write_text(yaml.safe_dump(inst_cfg))

        instances = [
            ProxyInstanceConfig(
                port=5757, upstream="https://api.anthropic.com",
                label="isolated", config=str(inst_cfg_path),
            ),
        ]

        apps_created = []

        def tracking_create_app(**kwargs):
            apps_created.append(kwargs)
            app = MagicMock()
            app.title = "test"
            app.state = MagicMock()
            return app

        class FakeServer:
            def __init__(self, config):
                pass
            async def serve(self):
                pass

        class FakeConfig:
            def __init__(self, app, **kw):
                pass

        shared_engine = MagicMock()
        shared_engine.config.monitor.context_window = 120_000

        with (
            patch("virtual_context.proxy.multi.create_app", side_effect=tracking_create_app),
            patch("virtual_context.proxy.multi.uvicorn") as mock_uv,
        ):
            mock_uv.Config = FakeConfig
            mock_uv.Server = FakeServer

            asyncio.run(run_multi_instance(
                instances=instances,
                config_path=None,
                engine=shared_engine,
            ))

        assert len(apps_created) == 1
        # Should NOT be the shared engine — it should be a new one
        assert apps_created[0]["shared_engine"] is not shared_engine
        assert apps_created[0]["config_path"] == str(inst_cfg_path)

    def test_instance_without_config_uses_shared_engine(self):
        """Instance without config field uses the shared engine."""
        instances = [
            ProxyInstanceConfig(
                port=5757, upstream="https://api.anthropic.com",
                label="shared", config="",
            ),
        ]

        apps_created = []

        def tracking_create_app(**kwargs):
            apps_created.append(kwargs)
            app = MagicMock()
            app.title = "test"
            app.state = MagicMock()
            return app

        class FakeServer:
            def __init__(self, config):
                pass
            async def serve(self):
                pass

        class FakeConfig:
            def __init__(self, app, **kw):
                pass

        shared_engine = MagicMock()
        shared_engine.config.monitor.context_window = 120_000
        shared_metrics = ProxyMetrics(context_window=120_000)

        with (
            patch("virtual_context.proxy.multi.create_app", side_effect=tracking_create_app),
            patch("virtual_context.proxy.multi.uvicorn") as mock_uv,
        ):
            mock_uv.Config = FakeConfig
            mock_uv.Server = FakeServer

            asyncio.run(run_multi_instance(
                instances=instances,
                config_path=None,
                engine=shared_engine,
                metrics=shared_metrics,
            ))

        assert len(apps_created) == 1
        assert apps_created[0]["shared_engine"] is shared_engine
        assert apps_created[0]["shared_metrics"] is shared_metrics

    def test_mixed_shared_and_isolated(self, tmp_path):
        """One shared + one isolated instance get correct engine assignment."""
        import yaml

        inst_cfg = {
            "version": "0.2",
            "storage_root": str(tmp_path / "iso"),
            "storage": {
                "backend": "sqlite",
                "sqlite": {"path": str(tmp_path / "iso.db")},
            },
        }
        inst_cfg_path = tmp_path / "iso.yaml"
        inst_cfg_path.write_text(yaml.safe_dump(inst_cfg))

        instances = [
            ProxyInstanceConfig(
                port=5757, upstream="https://api.anthropic.com",
                label="shared_inst", config="",
            ),
            ProxyInstanceConfig(
                port=5758, upstream="https://api.openai.com/v1",
                label="isolated_inst", config=str(inst_cfg_path),
            ),
        ]

        apps_created = []

        def tracking_create_app(**kwargs):
            apps_created.append(kwargs)
            app = MagicMock()
            app.title = "test"
            app.state = MagicMock()
            return app

        class FakeServer:
            def __init__(self, config):
                pass
            async def serve(self):
                pass

        class FakeConfig:
            def __init__(self, app, **kw):
                pass

        shared_engine = MagicMock()
        shared_engine.config.monitor.context_window = 120_000
        shared_metrics = ProxyMetrics(context_window=120_000)

        with (
            patch("virtual_context.proxy.multi.create_app", side_effect=tracking_create_app),
            patch("virtual_context.proxy.multi.uvicorn") as mock_uv,
        ):
            mock_uv.Config = FakeConfig
            mock_uv.Server = FakeServer

            asyncio.run(run_multi_instance(
                instances=instances,
                config_path="/master.yaml",
                engine=shared_engine,
                metrics=shared_metrics,
            ))

        assert len(apps_created) == 2
        # First instance (no config) → shared engine
        assert apps_created[0]["shared_engine"] is shared_engine
        assert apps_created[0]["shared_metrics"] is shared_metrics
        # Second instance (with config) → isolated engine
        assert apps_created[1]["shared_engine"] is not shared_engine
        assert apps_created[1]["shared_metrics"] is not shared_metrics


class TestCLIMultiInstance:
    """Test CLI dispatches correctly between single and multi-instance."""

    def test_single_instance_requires_upstream(self):
        """Without --upstream and no instances, CLI errors out."""
        from virtual_context.cli.main import cmd_proxy

        args = MagicMock()
        args.upstream = None
        args.config = None
        args.host = "127.0.0.1"
        args.port = 5757

        cfg = load_config(config_dict={})

        with pytest.raises(SystemExit) as exc_info:
            with patch("virtual_context.config.load_config", return_value=cfg):
                cmd_proxy(args)

        assert exc_info.value.code == 1

    def test_multi_instance_calls_run_multi(self):
        """With instances configured, multi-instance mode is triggered."""
        from virtual_context.cli.main import cmd_proxy

        args = MagicMock()
        args.upstream = None
        args.config = None
        args.host = "127.0.0.1"
        args.port = 5757

        cfg = load_config(config_dict={
            "proxy": {
                "instances": [
                    {"port": 5757, "upstream": "https://api.anthropic.com", "label": "anthropic"},
                ],
            },
        })

        with (
            patch("virtual_context.config.load_config", return_value=cfg),
            patch("virtual_context.proxy.multi.run_multi_instance") as mock_rmi,
        ):
            # asyncio.run calls the coroutine, so we need to handle it
            async def fake_run(*a, **kw):
                pass
            mock_rmi.return_value = fake_run()

            # The cmd_proxy function imports asyncio locally and calls asyncio.run
            # We need to let it actually run
            cmd_proxy(args)

            mock_rmi.assert_called_once()
            call_kwargs = mock_rmi.call_args
            assert len(call_kwargs.kwargs["instances"]) == 1
            assert call_kwargs.kwargs["instances"][0].label == "anthropic"


class TestDashboardInstanceLabel:
    """Phase 7: Instance label in dashboard settings and HTML."""

    def test_settings_includes_label(self, tmp_path):
        """GET /dashboard/settings includes instance_label when set."""
        from starlette.testclient import TestClient
        from virtual_context.proxy.server import create_app
        from virtual_context.engine import VirtualContextEngine

        config = load_config(config_dict={
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "storage_root": str(tmp_path),
        })
        engine = VirtualContextEngine(config=config)
        metrics = ProxyMetrics(context_window=120_000)

        app = create_app(
            upstream="https://api.anthropic.com",
            shared_engine=engine,
            shared_metrics=metrics,
            instance_label="anthropic",
        )
        client = TestClient(app)
        resp = client.get("/dashboard/settings")
        data = resp.json()
        assert data["instance_label"] == "anthropic"

    def test_settings_no_label_when_empty(self, tmp_path):
        """GET /dashboard/settings omits instance_label when not set."""
        from starlette.testclient import TestClient
        from virtual_context.proxy.server import create_app
        from virtual_context.engine import VirtualContextEngine

        config = load_config(config_dict={
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "storage_root": str(tmp_path),
        })
        engine = VirtualContextEngine(config=config)
        metrics = ProxyMetrics(context_window=120_000)

        app = create_app(
            upstream="https://api.anthropic.com",
            shared_engine=engine,
            shared_metrics=metrics,
        )
        client = TestClient(app)
        resp = client.get("/dashboard/settings")
        data = resp.json()
        assert "instance_label" not in data

    def test_dashboard_html_has_label_span(self):
        """Dashboard HTML includes the instance-label span."""
        from virtual_context.proxy.dashboard import get_dashboard_html
        html = get_dashboard_html()
        assert 'id="instance-label"' in html

    def test_app_title_includes_label(self, tmp_path):
        """App title includes instance label in brackets."""
        from virtual_context.proxy.server import create_app
        from virtual_context.engine import VirtualContextEngine

        config = load_config(config_dict={
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "storage_root": str(tmp_path),
        })
        engine = VirtualContextEngine(config=config)
        metrics = ProxyMetrics(context_window=120_000)

        app = create_app(
            upstream="https://api.anthropic.com",
            shared_engine=engine,
            shared_metrics=metrics,
            instance_label="gemini",
        )
        assert app.title == "virtual-context proxy [gemini]"
