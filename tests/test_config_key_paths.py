"""Every config path a message names must be one the loader reads.

A validation message that names the wrong section is worse than no message:
the operator follows it, writes a block that loads cleanly, and is read by
nothing. Nothing errors, the setting stays at its default, and they believe
it took effect. For a safety gate that means believing a check is running
when the pipeline is simply off.
"""

from __future__ import annotations

import re
import pathlib

import pytest
import yaml

from virtual_context.config import _MISLEADING_TOP_LEVEL_BLOCKS, load_config

CONFIG_SRC = pathlib.Path(
    "virtual_context/config.py"
).read_text()


class TestNamedPathsMatchTheLoader:
    def test_every_dotted_path_in_a_message_names_a_real_section(self):
        """The sweep that found segmenter.* after engagement.*."""
        sections = set(re.findall(r'raw\.get\("([a-z_]+)"', CONFIG_SRC))
        named = set(re.findall(
            r'\b([a-z_]+)\.([a-z_0-9]+) (?:is required|must be)', CONFIG_SRC,
        ))
        assert named, "sweep found no dotted paths — the pattern has drifted"
        wrong = sorted(
            f"{head}.{tail}" for head, tail in named if head not in sections
        )
        assert wrong == [], f"messages name sections the loader never reads: {wrong}"

    def test_the_engagement_settings_are_named_by_their_real_path(self):
        assert "assembly.engagement_fidelity_judge_model is required" in CONFIG_SRC
        assert "engagement.fidelity_judge_model is required" not in CONFIG_SRC


class TestAMisleadingBlockIsRefused:
    """Silently discarding a block someone wrote is what hid this."""

    @pytest.mark.parametrize("block", sorted(_MISLEADING_TOP_LEVEL_BLOCKS))
    def test_it_raises_rather_than_being_ignored(self, block, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({block: {"anything": 1}}))
        with pytest.raises(ValueError, match=f"'{block}:'"):
            load_config(str(path))

    def test_the_error_says_where_the_settings_actually_live(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(
            {"engagement": {"fidelity_judge_model": "some/model"}},
        ))
        with pytest.raises(ValueError, match="assembly"):
            load_config(str(path))

    def test_a_config_without_the_block_still_loads(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"assembly": {"engagement_enabled": False}}))
        assert load_config(str(path)) is not None

    def test_the_real_spelling_is_read(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"assembly": {
            "engagement_enabled": True,
            "engagement_fidelity_judge_model": "some/model",
        }}))
        config = load_config(str(path))
        assert config.assembler.engagement_enabled is True
        assert config.assembler.engagement_fidelity_judge_model == "some/model"

    def test_a_top_level_block_the_loader_does_read_is_untouched(self, tmp_path):
        """The refusal is a short list, not a blanket unknown-key check.

        The shipped production config carries a `telemetry:` block that this
        loader reads nothing from. A blanket rule would refuse a config that
        works in production today, so the list stays explicit.
        """
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"telemetry": {"enabled": True}}))
        assert load_config(str(path)) is not None
