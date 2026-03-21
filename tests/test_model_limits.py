# tests/test_model_limits.py
"""Tests for model-aware upstream context limit resolution."""
from virtual_context.model_limits import resolve_upstream_limit


class TestResolveUpstreamLimit:
    def test_instance_override_takes_precedence(self):
        assert resolve_upstream_limit("claude-opus-4-6", instance_limit=500_000) == 500_000

    def test_global_override_when_no_instance(self):
        assert resolve_upstream_limit("claude-opus-4-6", global_limit=300_000) == 300_000

    def test_instance_beats_global(self):
        assert resolve_upstream_limit("gpt-4o", instance_limit=100_000, global_limit=200_000) == 100_000

    def test_anthropic_opus(self):
        assert resolve_upstream_limit("claude-opus-4-6") == 1_000_000

    def test_anthropic_sonnet(self):
        assert resolve_upstream_limit("claude-sonnet-4-6") == 200_000

    def test_anthropic_haiku(self):
        assert resolve_upstream_limit("claude-haiku-4-5-20251001") == 200_000

    def test_gpt5(self):
        assert resolve_upstream_limit("gpt-5") == 1_000_000

    def test_gpt5_4(self):
        assert resolve_upstream_limit("gpt-5.4") == 1_000_000

    def test_gpt4o(self):
        assert resolve_upstream_limit("gpt-4o") == 128_000

    def test_gpt4_1(self):
        assert resolve_upstream_limit("gpt-4.1") == 1_000_000

    def test_o3(self):
        assert resolve_upstream_limit("o3") == 200_000

    def test_o4_mini(self):
        assert resolve_upstream_limit("o4-mini") == 200_000

    def test_gemini_25_pro(self):
        assert resolve_upstream_limit("gemini-2.5-pro") == 1_000_000

    def test_gemini_20_flash(self):
        assert resolve_upstream_limit("gemini-2.0-flash") == 1_000_000

    def test_deepseek(self):
        assert resolve_upstream_limit("deepseek-r1") == 128_000

    def test_llama4(self):
        assert resolve_upstream_limit("llama-4-scout") == 128_000

    def test_openrouter_prefix_stripped(self):
        assert resolve_upstream_limit("anthropic/claude-opus-4-6") == 1_000_000
        assert resolve_upstream_limit("openai/gpt-4o") == 128_000

    def test_unknown_model_fallback(self):
        assert resolve_upstream_limit("some-unknown-model-v3") == 200_000

    def test_empty_model_fallback(self):
        assert resolve_upstream_limit("") == 200_000


class TestConfigParsing:
    def test_instance_upstream_limit_parsed(self):
        from virtual_context.config import load_config
        import tempfile, os, yaml
        cfg_data = {
            "proxy": {
                "instances": [
                    {"port": 5757, "upstream": "https://api.anthropic.com",
                     "upstream_context_limit": 1_000_000},
                    {"port": 5758, "upstream": "https://api.openai.com",
                     "upstream_context_limit": 128_000},
                ]
            },
            "providers": {"default": {"provider": "ollama", "model": "test"}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_data, f)
            path = f.name
        try:
            config = load_config(path, validate=False)
            assert config.proxy.instances[0].upstream_context_limit == 1_000_000
            assert config.proxy.instances[1].upstream_context_limit == 128_000
        finally:
            os.unlink(path)

    def test_global_default_is_zero(self):
        from virtual_context.types import ProxyConfig
        cfg = ProxyConfig()
        assert cfg.upstream_context_limit == 0

    def test_instance_default_is_zero(self):
        from virtual_context.types import ProxyInstanceConfig
        cfg = ProxyInstanceConfig()
        assert cfg.upstream_context_limit == 0
