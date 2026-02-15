"""Preset registry: register, lookup, and list presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Preset:
    name: str
    description: str
    config_dict: dict
    template: str


_PRESETS: dict[str, Preset] = {}


def register_preset(preset: Preset) -> None:
    """Register a preset by name."""
    _PRESETS[preset.name] = preset


def get_preset(name: str) -> Preset | None:
    """Return a preset by name, or None if not found."""
    return _PRESETS.get(name)


def list_presets() -> list[Preset]:
    """Return all registered presets."""
    return list(_PRESETS.values())
