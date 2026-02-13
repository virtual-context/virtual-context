"""LLM Provider base protocol and error class.

The LLMProvider Protocol is defined in types.py.
This module re-exports for convenience.
"""

from ..types import LLMProvider, LLMProviderError

__all__ = ["LLMProvider", "LLMProviderError"]
