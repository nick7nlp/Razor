"""Registered adapters; model support requires layout and forward validation."""
from __future__ import annotations

from .base import MoEAdapter, MoEBlock, MoEContext, RouterSpec, register, get_adapter, list_adapters
from . import qwen as _qwen  # noqa: F401
from . import glm as _glm  # noqa: F401
from . import hunyuan as _hunyuan  # noqa: F401
from . import deepseek as _deepseek  # noqa: F401
from . import generic as _generic  # noqa: F401

__all__ = ["MoEAdapter", "MoEBlock", "MoEContext", "RouterSpec", "register", "get_adapter", "list_adapters"]
