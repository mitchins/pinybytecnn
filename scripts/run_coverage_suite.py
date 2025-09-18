"""Minimal coverage harness for Codecov uploads."""

from __future__ import annotations

import json
from pathlib import Path

from tinybytecnn.multi_layer_optimized import (
    BatchNorm1D,
    FusedMultiConv1D,
    MultiLayerByteCNN,
    OptimizedDense,
    OptimizedEmbedding,
)
from tinybytecnn.layers import Conv1DMaxPool


def exercise_embedding() -> list[list[float]]:
    emb = OptimizedEmbedding(16, 4)
    for idx in range(4):
        row = emb.weight[idx]
        for col in range(4):
            row[col] = (idx + 1) * 0.05 * (col + 1)
    return emb.forward([0, 1, 2, 3], max_len=4)


def exercise_batch_norm(vectors: list[list[float]]) -> list[list[float]]:
    bn = BatchNorm1D(len(vectors[0]))
    weight = [1.1 for _ in range(bn.num_features)]
    bias = [0.1 for _ in range(bn.num_features)]
    running_mean = [0.0 for _ in range(bn.num_features)]
    running_var = [1.0 for _ in range(bn.num_features)]
    bn.load_parameters(weight, bias, running_mean, running_var)
    return bn.forward(vectors)


def exercise_conv_block(vectors: list[list[float]]) -> list[float]:
    conv = FusedMultiConv1D(
        [
            {"in_channels": 4, "out_channels": 6, "kernel_size": 3},
            {"in_channels": 6, "out_channels": 5, "kernel_size": 3},
        ],
        max_seq_len=len(vectors),
    )

    for layer_idx, conf in enumerate(conv.layers_config):
        out_ch = conf["out_channels"]
        in_ch = conf["in_channels"]
        kernel = conf["kernel_size"]
        weights = [
            [[0.02 * (i + j + k) for _ in range(in_ch)] for k in range(kernel)]
            for i in range(out_ch)
        ]
        biases = [0.01 * i for i in range(out_ch)]
        conv.set_weights(layer_idx, weights, biases)

    return conv.forward(vectors)


def exercise_dense(vector: list[float]) -> list[float]:
    dense = OptimizedDense(len(vector), 3)
    for row_idx, row in enumerate(dense.weight):
        for col_idx in range(len(vector)):
            row[col_idx] = 0.05 * (row_idx + 1) * (col_idx + 1)
    dense.bias = [0.05 * i for i in range(3)]
    return dense.forward(vector)


def exercise_model() -> None:
    model = MultiLayerByteCNN.create_2layer_32kb(max_len=32)
    model.predict("Codecov minimal run", strategy="truncate")
    model.forward_indices([1, 2, 3, 4, 5])
    weights_path = Path("tests/bytecnn_10k_reference.json")
    weights_data = json.loads(weights_path.read_text())
    model.load_weights_from_dict(weights_data)
    model.predict("Codecov minimal run", strategy="truncate")


def exercise_conv1d_max_pool() -> None:
    layer = Conv1DMaxPool(in_channels=2, out_channels=2, kernel_size=3)
    layer.weight = [
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.2, 0.1], [0.0, -0.1], [0.1, 0.0]],
    ]
    if layer.bias is not None:
        layer.bias = [0.1, -0.1]
    _ = layer.forward([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


def main() -> None:
    embeddings = exercise_embedding()
    normalized = exercise_batch_norm(embeddings)
    pooled = exercise_conv_block(normalized)
    dense_out = exercise_dense(pooled)
    assert len(dense_out) == 3
    exercise_model()
    exercise_conv1d_max_pool()


if __name__ == "__main__":
    main()
