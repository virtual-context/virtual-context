"""Presets — ready-to-use config templates for common use cases."""

from .base import get_preset, list_presets  # noqa: F401

# Import presets to trigger registration
from . import coding  # noqa: F401
from . import general  # noqa: F401
