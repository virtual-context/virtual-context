"""Shared embedding provider — single model load shared across engine components."""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Owns one embedding model, shared across SemanticSearchManager,
    EmbeddingTagGenerator, and any other consumer.

    Three construction modes:
    - Injected: EmbeddingProvider(embed_fn=my_fn) — the callable is the model.
    - Standalone: EmbeddingProvider(model_name=...) — lazy local load on first
      use.
    - Disabled: EmbeddingProvider(disabled=True) — embeddings are permanently
      off. ``get_embed_fn`` returns None without ever attempting a local
      load. This is a distinct state, not a failure: a host that must never
      load a local model uses it so that no consumer can interpret an absent
      callable as permission to construct one.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        *,
        disabled: bool = False,
    ) -> None:
        if disabled and embed_fn is not None:
            raise ValueError("disabled=True cannot be combined with embed_fn")
        self._model_name = model_name
        self._embed_fn: Callable[[list[str]], list[list[float]]] | None = embed_fn
        self._loaded = embed_fn is not None
        self._load_failed = False
        self._disabled = disabled

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def disabled(self) -> bool:
        return self._disabled

    def get_embed_fn(self) -> Callable[[list[str]], list[list[float]]] | None:
        """Return the embed function, lazy-loading the model on first call.

        Returns None permanently when the provider was constructed disabled,
        and returns None if sentence-transformers is not installed or the
        load fails.
        """
        if self._disabled:
            return None
        if self._loaded:
            return self._embed_fn
        if self._load_failed:
            return None

        try:
            from sentence_transformers import SentenceTransformer

            old_stderr = sys.stderr
            try:
                sys.stderr = open(os.devnull, "w")
                model = SentenceTransformer(self._model_name)
            finally:
                try:
                    sys.stderr.close()
                except Exception:
                    pass
                sys.stderr = old_stderr

            def embed(texts: list[str]) -> list[list[float]]:
                return model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False,
                ).tolist()

            self._embed_fn = embed
            self._loaded = True
            logger.info("EmbeddingProvider: loaded model %s", self._model_name)
            return self._embed_fn

        except ImportError:
            logger.debug("sentence-transformers not installed, embeddings disabled")
            self._load_failed = True
            return None
        except Exception:
            logger.debug("Failed to load embedding model %s", self._model_name, exc_info=True)
            self._load_failed = True
            return None
