"""
Fused Conv1D + ReLU + GlobalMaxPool (manual backprop, pure Python).

Interface
---------
Conv1DMaxPool(in_dim, out_channels, kernel_size, bias=True)

Forward:
  - Input x: list[list[float]] with shape [seq_len][in_dim]
  - Returns: list[float] with shape [out_channels]
    (global max over time after ReLU per filter)

Backward:
  - Input grad_out: list[float] with shape [out_channels]
  - Returns: list[list[float]] with shape [seq_len][in_dim] (grad w.r.t input)

Notes
-----
- This layer computes, for each filter f and time t:
      z_f(t) = b_f + sum_{i=0..k-1} dot(W_f[i], x[t+i])
      a_f(t) = relu(z_f(t))
  and outputs y_f = max_t a_f(t)

- Backprop only flows to the argmax time index (per filter) and only if the
  pre-activation z_f(t*) > 0 (due to ReLU). Derivatives are computed directly
  from the convolution definition.

- This implementation keeps only the argmax index and its pre-activation per
  filter, avoiding storing the full convolution map.
"""

import random
from typing import Optional


def _zeros_like_weights(weights: list[list[list[float]]]) -> list[list[list[float]]]:
    return [[[0.0 for _ in row] for row in kernel] for kernel in weights]


def _zeros_2d(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


class Conv1DMaxPool:
    def __init__(
        self,
        in_dim: int | None = None,
        out_channels: int = 0,
        kernel_size: int = 0,
        padding: int = 0,
        bias: bool = True,
        init_scale: float = 0.02,
        *,
        in_channels: int | None = None,
    ):
        """Create the fused Conv1D → ReLU → GlobalMaxPool layer.

        Historically the class accepted ``in_dim`` as the first positional
        argument.  Some of the newer tests exercise an ``in_channels`` keyword
        (to match the configuration dictionaries used elsewhere).  To remain
        backwards compatible we accept both names and validate that, when both
        are provided, they agree.
        """

        if in_dim is None:
            if in_channels is None:
                raise ValueError("in_dim or in_channels must be provided")
            in_dim = in_channels
        elif in_channels is not None and in_channels != in_dim:
            raise ValueError("in_dim and in_channels must match when both provided")

        if in_dim <= 0 or out_channels <= 0 or kernel_size <= 0:
            raise ValueError("in_dim, out_channels, kernel_size must be positive")

        self.in_dim = in_dim
        self.in_channels = in_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = bias

        # Weights: [out_channels][kernel_size][in_dim]
        s = init_scale
        self.weight: list[list[list[float]]] = [
            [[random.uniform(-s, s) for _ in range(in_dim)] for _ in range(kernel_size)]
            for _ in range(out_channels)
        ]
        self.weight_grad: list[list[list[float]]] = _zeros_like_weights(self.weight)

        # Bias: [out_channels]
        self.bias: list[float] | None = [0.0 for _ in range(out_channels)] if bias else None
        self.bias_grad: list[float] | None = [0.0 for _ in range(out_channels)] if bias else None

        # Cache for backward
        self._x: list[list[float]] | None = None
        self._argmax_t: list[int] | None = None
        self._argmax_pre_relu: list[float] | None = None

    def zero_grad(self) -> None:
        # Reset parameter grads to zero
        wg = self.weight_grad
        for f in range(self.out_channels):
            k = self.kernel_size
            for i in range(k):
                row = wg[f][i]
                for c in range(self.in_dim):
                    row[c] = 0.0
        if self.use_bias and self.bias_grad is not None:
            for f in range(self.out_channels):
                self.bias_grad[f] = 0.0

    def forward(self, x: list[list[float]]) -> list[float]:
        # Basic validation
        if not x or not isinstance(x, list) or not isinstance(x[0], list):
            raise ValueError("Input x must be a non-empty list[list[float]]")
        seq_len = len(x)
        if len(x[0]) != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {len(x[0])}")
        if seq_len + 2 * self.padding < self.kernel_size:
            raise ValueError("Effective sequence too short for the given kernel_size and padding")

        in_dim = self.in_dim
        k = self.kernel_size
        F = self.out_channels
        W = self.weight
        b = self.bias

        # Same-length conv when padding = k//2; otherwise general padded conv.
        # We compute an output position for every input time step (0..seq_len-1)
        Tvalid = seq_len
        # Track per-filter argmax and value
        max_vals = [-1e30 for _ in range(F)]
        argmax_t = [0 for _ in range(F)]
        pre_relu_at_max = [-1e30 for _ in range(F)]

        # Iterate time positions and filters, compute ReLU(conv), track max
        for t_out in range(Tvalid):
            # Corresponding leftmost index in input is t = t_out - padding
            t_left = t_out - self.padding
            for f in range(F):
                # conv dot product over kernel window and in_dim
                z = b[f] if (b is not None) else 0.0
                wf = W[f]
                # Unroll loops carefully
                for i in range(k):
                    t_in = t_left + i
                    if t_in < 0 or t_in >= seq_len:
                        # zero padding
                        continue
                    xi = x[t_in]
                    wfi = wf[i]
                    # dot
                    s = 0.0
                    for c in range(in_dim):
                        s += wfi[c] * xi[c]
                    z += s
                # ReLU
                a = z if z > 0.0 else 0.0
                if a > max_vals[f]:
                    max_vals[f] = a
                    argmax_t[f] = t_out
                    pre_relu_at_max[f] = z

        # Cache for backward
        self._x = x
        self._argmax_t = argmax_t
        self._argmax_pre_relu = pre_relu_at_max

        return max_vals

    def backward(self, grad_out: list[float]) -> list[list[float]]:
        if self._x is None or self._argmax_t is None or self._argmax_pre_relu is None:
            raise RuntimeError("Must call forward before backward")

        x = self._x
        in_dim = self.in_dim
        k = self.kernel_size
        F = self.out_channels
        W = self.weight
        WG = self.weight_grad
        bG = self.bias_grad
        argmax_t = self._argmax_t
        pre_relu = self._argmax_pre_relu

        seq_len = len(x)
        # Gradient w.r.t input
        dx = _zeros_2d(seq_len, in_dim)

        # For each filter, backprop only along argmax time if z>0
        for f in range(F):
            t_star = argmax_t[f]
            z_star = pre_relu[f]
            g = grad_out[f]
            if z_star <= 0.0:
                # ReLU off -> no gradient flows
                continue
            # dL/dz at t_star equals upstream grad (since max-pool selected that index)
            dz = g

            # Accumulate bias grad
            if bG is not None:
                bG[f] += dz

            # Window [t_star : t_star + k)
            wf = W[f]
            wgf = WG[f]
            for i in range(k):
                t_in = (t_star - self.padding) + i
                if t_in < 0 or t_in >= seq_len:
                    continue
                xi = x[t_in]
                wfi = wf[i]
                wgfi = wgf[i]
                for c in range(in_dim):
                    # dL/dW = dz * x
                    wgfi[c] += dz * xi[c]
                    # dL/dx = dz * W
                    dx[t_in][c] += dz * wfi[c]

        return dx

    # Optional helper if an optimizer expects parameters access
    def parameters(self):
        return {
            "weight": self.weight,
            "weight_grad": self.weight_grad,
            "bias": self.bias,
            "bias_grad": self.bias_grad,
        }


class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        if vocab_size <= 0 or dim <= 0:
            raise ValueError("vocab_size and dim must be positive")
        import random

        s = 0.02
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight: list[list[float]] = [
            [random.uniform(-s, s) for _ in range(dim)] for _ in range(vocab_size)
        ]

    def forward(self, idxs: list[int]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in idxs:
            if i < 0 or i >= self.vocab_size:
                raise ValueError(f"index {i} out of range [0,{self.vocab_size})")
            out.append(self.weight[i])
        return out


class Dense:
    def __init__(self, in_dim: int, out_dim: int):
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        import random

        s = 1.0 / (in_dim**0.5)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight: list[list[float]] = [
            [random.uniform(-s, s) for _ in range(in_dim)] for _ in range(out_dim)
        ]
        self.bias: list[float] = [0.0 for _ in range(out_dim)]

    def forward(self, x: list[float]) -> list[float]:
        if len(x) != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {len(x)}")
        out = [0.0 for _ in range(self.out_dim)]
        for o in range(self.out_dim):
            s = self.bias[o]
            wrow = self.weight[o]
            for i in range(self.in_dim):
                s += wrow[i] * x[i]
            out[o] = s
        return out


class Sigmoid:
    def forward(self, x: list[float]) -> list[float]:
        import math

        return [1.0 / (1.0 + math.exp(-xi)) for xi in x]
