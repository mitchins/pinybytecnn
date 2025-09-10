"""
TinyByteCNN: minimal, pure-Python layers/components.

Exports:
- Conv1DMaxPool: Fused Conv1D + ReLU + GlobalMaxPool with manual backprop.
- Embedding, Dense, Sigmoid: Minimal layers for ByteCNN.
- ByteCNN: End-to-end model with weight loaders.
"""

from .layers import Conv1DMaxPool, Dense, Embedding, Sigmoid
from .model import ByteCNN


__all__ = [
    "ByteCNN",
    "Conv1DMaxPool",
    "Dense",
    "Embedding",
    "Sigmoid",
]
