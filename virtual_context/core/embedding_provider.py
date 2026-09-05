"""Shared embedding provider — single model load shared across engine components."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable

logger = logging.getLogger(__name__)


# One loaded sentence-transformer per (process, model_name). Every consumer
# instance used to perform its own lazy load, so a process holding many
# engines — a pooled multi-tenant host, a test worker — paid a full torch
# model load per instance and ratcheted resident memory with every one.
# Only SUCCESSFUL loads are cached: a transient failure stays a per-caller
# condition rather than poisoning the process, and the ImportError-disabled
# path keeps its per-instance semantics.
_PROCESS_MODEL_CACHE: dict[str, Callable[[list[str]], list[list[float]]]] = {}
_PROCESS_MODEL_LOCK = threading.Lock()


def get_shared_embed_fn(
    model_name: str,
) -> Callable[[list[str]], list[list[float]]] | None:
    """Return the process-wide embed callable for *model_name*.

    Double-checked locking bounds concurrent first-touch to exactly one
    load; the losers wait only for the load window. Encoding is NOT
    serialized: the model is used for inference only and the returned
    closure is stateless. ``None`` means the load failed (package missing
    or model unavailable) and nothing was cached.
    """
    fn = _PROCESS_MODEL_CACHE.get(model_name)
    if fn is not None:
        return fn
    with _PROCESS_MODEL_LOCK:
        fn = _PROCESS_MODEL_CACHE.get(model_name)
        if fn is not None:
            return fn
        fn = _load_sentence_transformer(model_name)
        if fn is not None:
            _PROCESS_MODEL_CACHE[model_name] = fn
        return fn


def _load_sentence_transformer(
    model_name: str,
) -> Callable[[list[str]], list[list[float]]] | None:
    try:
        from sentence_transformers import SentenceTransformer

        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, "w")
            model = SentenceTransformer(model_name)
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

        logger.info("EmbeddingProvider: loaded model %s", model_name)
        return embed
    except ImportError:
        logger.info('Local embeddings unavailable; install "virtual-context[embeddings]" to enable them')
        return None
    except Exception:
        logger.debug(
            "Failed to load embedding model %s", model_name, exc_info=True,
        )
        return None


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

        fn = get_shared_embed_fn(self._model_name)
        if fn is None:
            self._load_failed = True
            return None
        self._embed_fn = fn
        self._loaded = True
        return fn
