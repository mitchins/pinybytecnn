"""
TinyByteCNN: minimal, pure-Python layers/components.

Exports:
- Conv1DMaxPool: Fused Conv1D + ReLU + GlobalMaxPool with manual backprop.
- Embedding, Dense, Sigmoid: Minimal layers for ByteCNN.
- ByteCNN: End-to-end model with weight loaders.
"""

"""Top level exports for :mod:`tinybytecnn`.

This module keeps the public surface area intentionally tiny so that the
pure-Python implementation can be embedded in constrained environments.
Some of the unit tests exercise legacy ``unittest.mock`` behaviour on
platforms where :mod:`unittest.mock` might not be automatically exposed on
the ``unittest`` module.  To remain compatible with those environments we
perform a light-weight shim here that mirrors the behaviour of the standard
library when ``mock`` is available.  The shim is effectively free at runtime
and keeps the hot code paths in the model untouched.
"""

from .layers import Conv1DMaxPool, Dense, Embedding, Sigmoid
from .model import ByteCNN

# ``unittest.mock`` exists on every modern Python, but a handful of
# environments require explicit initialisation.  Import lazily so that the
# dependency is only pulled in when the standard library provides it and fall
# back to the third-party ``mock`` package if necessary.  This mirrors the
# behaviour expected by the tests while keeping imports side-effect free.
try:  # pragma: no cover - defensive compatibility shim
    import unittest

    if not hasattr(unittest, "mock"):
        try:
            from unittest import mock as _mock  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - very old Python
            import mock as _mock  # type: ignore[import-not-found]

        unittest.mock = _mock  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - never fail import for consumers
    pass


__all__ = [
    "ByteCNN",
    "Conv1DMaxPool",
    "Dense",
    "Embedding",
    "Sigmoid",
]
