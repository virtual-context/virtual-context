"""ModelCatalog: load model pricing from a YAML file."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    canonical_name: str
    provider: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int
    aliases: list[str] = field(default_factory=list)


class ModelCatalog:
    """Centralized model pricing loaded from a YAML file."""

    def __init__(self, path: str) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._alias_map: dict[str, str] = {}
        self._load(path)

    @classmethod
    def default(cls) -> ModelCatalog:
        """Load the default models.yaml shipped alongside the package."""
        here = os.path.dirname(os.path.abspath(__file__))
        # core/ -> virtual_context/ -> project root
        root = os.path.dirname(os.path.dirname(here))
        path = os.path.join(root, "models.yaml")
        return cls(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Model catalog not found at %s — costs will be zero", path)
            return
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        models_raw = raw.get("models", {})
        for name, info in models_raw.items():
            model = ModelInfo(
                canonical_name=name,
                provider=info.get("provider", ""),
                input_per_mtok=float(info.get("input_per_mtok", 0.0)),
                output_per_mtok=float(info.get("output_per_mtok", 0.0)),
                context_window=int(info.get("context_window", 0)),
                aliases=info.get("aliases", []),
            )
            self._models[name] = model
            for alias in model.aliases:
                self._alias_map[alias.lower()] = name

    def _resolve(self, model_name: str) -> ModelInfo | None:
        # 1. Exact match
        if model_name in self._models:
            return self._models[model_name]
        # 2. Alias match
        canonical = self._alias_map.get(model_name.lower())
        if canonical:
            return self._models[canonical]
        # 3. Substring match (e.g. "haiku" in "claude-haiku-4-5-20251001")
        name_lower = model_name.lower()
        for key, model in self._models.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return model
        return None

    def get_pricing(self, model_name: str) -> tuple[float, float]:
        """Return (input_per_mtok, output_per_mtok) for a model."""
        model = self._resolve(model_name)
        if model is None:
            logger.debug("Unknown model '%s' — returning zero pricing", model_name)
            return (0.0, 0.0)
        return (model.input_per_mtok, model.output_per_mtok)

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given token count."""
        inp_rate, out_rate = self.get_pricing(model_name)
        return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000

    def get_context_window(self, model_name: str) -> int:
        """Return context window size for a model, or 0 if unknown."""
        model = self._resolve(model_name)
        return model.context_window if model else 0
