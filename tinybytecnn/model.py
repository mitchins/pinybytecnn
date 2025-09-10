from typing import Any, Optional

from .layers import Conv1DMaxPool, Dense, Embedding, Sigmoid
from .utils import relu_vec, text_to_fixed_bytes


class ByteCNN:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        conv_filters: int,
        conv_kernel_size: int,
        hidden_dim: int,
        output_dim: int = 1,
        max_len: int = 128,
        padding: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.padding = padding if padding is not None else (conv_kernel_size // 2)

        self.embed = Embedding(vocab_size, embed_dim)
        self.conv = Conv1DMaxPool(embed_dim, conv_filters, conv_kernel_size, padding=self.padding)
        self.fc1 = Dense(conv_filters, hidden_dim)
        self.fc2 = Dense(hidden_dim, output_dim)
        self.sigmoid = Sigmoid()

    def forward_indices(self, idxs: list[int]) -> float:
        x = self.embed.forward(idxs)  # [T, D]
        x = self.conv.forward(x)  # [F]
        x = self.fc1.forward(x)  # [H]
        x = relu_vec(x)
        x = self.fc2.forward(x)  # [1]
        y = self.sigmoid.forward(x)[0]
        return y

    def predict(self, text: str, strategy: str = "truncate") -> float:
        if strategy == "truncate":
            idxs = text_to_fixed_bytes(text, max_len=self.max_len)
            return self.forward_indices(idxs)
        elif strategy in ["average", "attention"]:
            return self.predict_sliding_window(text, strategy)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Use 'truncate', 'average', or 'attention'"
            )

    def predict_sliding_window(self, text: str, strategy: str) -> float:
        """Handle long texts with sliding window strategies."""
        from .utils import text_to_bytes

        # Convert text to bytes
        byte_data = text_to_bytes(text)

        # If text fits in single window, use direct prediction
        if len(byte_data) <= self.max_len:
            return self.predict(text, strategy="truncate")

        # Create sliding windows
        window_size = self.max_len
        stride = window_size // 2  # 50% overlap
        windows = []

        for start in range(0, len(byte_data) - window_size + 1, stride):
            window = byte_data[start : start + window_size]
            windows.append(window)

        # Handle final partial window if needed
        if len(byte_data) % stride != 0:
            final_window = byte_data[-window_size:]
            if len(final_window) == window_size:
                windows.append(final_window)

        # Get predictions for each window
        predictions = []
        for window in windows:
            # Convert bytes back to indices for model
            idxs = list(window)  # bytes are already 0-255 integers
            # Pad to max_len if needed
            while len(idxs) < self.max_len:
                idxs.append(0)
            pred = self.forward_indices(idxs)
            predictions.append(pred)

        # Aggregate predictions based on strategy
        if strategy == "average":
            return sum(predictions) / len(predictions)
        elif strategy == "attention":
            # Attention-weighted: higher predictions get more weight
            import math

            # Convert predictions to logits for attention weighting
            logits = [math.log(p / (1 - p + 1e-8)) for p in predictions]
            # Softmax attention weights
            max_logit = max(logits)
            exp_logits = [math.exp(logit - max_logit) for logit in logits]
            sum_exp = sum(exp_logits)
            attention_weights = [e / sum_exp for e in exp_logits]
            # Weighted average of original predictions
            weighted_pred = sum(w * p for w, p in zip(attention_weights, predictions, strict=False))
            return weighted_pred

        raise ValueError(f"Unknown sliding window strategy: {strategy}")

    # -------------------------
    # Weight loading utilities
    # -------------------------
    @staticmethod
    def _transpose_conv_weight(w: list[list[list[float]]]) -> list[list[list[float]]]:
        # Input w shape: [out_channels][in_channels][kernel]
        # Output to Conv1DMaxPool expected: [out_channels][kernel][in_channels]
        OC = len(w)
        IC = len(w[0]) if OC > 0 else 0
        K = len(w[0][0]) if IC > 0 else 0
        out = [[[0.0 for _ in range(IC)] for _ in range(K)] for _ in range(OC)]
        for oc in range(OC):
            for ic in range(IC):
                for k in range(K):
                    out[oc][k][ic] = w[oc][ic][k]
        return out

    @classmethod
    def from_weight_dict(cls, weights: dict[str, Any], max_len: int = 128) -> "ByteCNN":
        # Expect keys like the provided demo: embedding, conv1_weight, conv1_bias,
        # classifier_weight, classifier_bias, output_weight, output_bias
        emb = weights["embedding"]  # [vocab, embed_dim]
        vocab_size = len(emb)
        embed_dim = len(emb[0])

        conv_w = weights["conv1_weight"]  # [out, in, k]
        conv_b = weights["conv1_bias"]  # [out]
        conv_filters = len(conv_w)
        in_channels = len(conv_w[0])
        conv_k = len(conv_w[0][0])
        if in_channels != embed_dim:
            raise ValueError(f"conv in_channels {in_channels} != embed_dim {embed_dim}")

        cls_w = weights["classifier_weight"]  # [out, in]
        cls_b = weights["classifier_bias"]  # [out]
        hidden_dim = len(cls_b)
        cls_in = len(cls_w[0]) if hidden_dim > 0 else 0
        if cls_in != conv_filters:
            raise ValueError(f"classifier in_dim {cls_in} != conv_filters {conv_filters}")

        out_w = weights["output_weight"]  # [1, hidden]
        out_b = weights["output_bias"]  # [1]
        output_dim = len(out_b)
        if output_dim != 1:
            raise ValueError("Only binary output supported (output_dim=1)")
        out_in = len(out_w[0])
        if out_in != hidden_dim:
            raise ValueError(f"output in_dim {out_in} != hidden_dim {hidden_dim}")

        model = cls(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            conv_filters=conv_filters,
            conv_kernel_size=conv_k,
            hidden_dim=hidden_dim,
            output_dim=1,
            max_len=max_len,
            padding=conv_k // 2,
        )

        # Load weights
        model.embed.weight = [list(map(float, row)) for row in emb]
        # conv: transpose to [out][k][in]
        cw = cls._transpose_conv_weight(conv_w)
        model.conv.weight = cw
        model.conv.bias = [float(v) for v in conv_b] if model.conv.use_bias else None

        model.fc1.weight = [list(map(float, row)) for row in cls_w]
        model.fc1.bias = [float(v) for v in cls_b]

        model.fc2.weight = [list(map(float, row)) for row in out_w]
        model.fc2.bias = [float(v) for v in out_b]

        return model

    @staticmethod
    def load_weights_json(path: str) -> dict[str, Any]:
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # If file includes model_info, keep it but pass through
        return data

    @staticmethod
    def load_weights_npz(path: str) -> dict[str, Any]:
        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError("NumPy is required to load NPZ weights; use JSON instead") from e
        npz = np.load(path, allow_pickle=True)
        out: dict[str, Any] = {}
        for k in npz.files:
            v = npz[k]
            if hasattr(v, "tolist"):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out
