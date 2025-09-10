"""
Multi-Layer Optimized ByteCNN for Pure Python Edge Deployment
Supports 1, 2, and 3 conv layers with aggressive optimization strategies
Target: F1=0.744 performance with minimal latency
"""

import math
from typing import Any, Optional, Union


def relu_vec(x: list[float]) -> list[float]:
    """Vectorized ReLU - basic building block"""
    return [xi if xi > 0.0 else 0.0 for xi in x]


def sigmoid_vec(x: list[float]) -> list[float]:
    """Vectorized sigmoid"""
    return [1.0 / (1.0 + math.exp(-xi)) for xi in x]


class FusedMultiConv1D:
    """
    Fused multi-layer 1D convolution optimized for pure Python
    Strategy: Pre-compute as much as possible, minimize Python loops
    """

    def __init__(self, layers_config: list[dict[str, Any]], max_seq_len: int = 512):
        """
        layers_config: List of dicts with keys: in_channels, out_channels, kernel_size
        Example: [{"in_channels": 14, "out_channels": 28, "kernel_size": 3},
                  {"in_channels": 28, "out_channels": 24, "kernel_size": 3}]
        """
        self.layers_config = layers_config
        self.num_layers = len(layers_config)
        self.max_seq_len = max_seq_len

        # Pre-allocate intermediate buffers (reuse memory)
        self.layer_outputs = []
        for _i, config in enumerate(layers_config):
            out_channels = config["out_channels"]
            # Pre-allocate for maximum sequence length
            layer_buffer = [[0.0 for _ in range(max_seq_len)] for _ in range(out_channels)]
            self.layer_outputs.append(layer_buffer)

        # Weights storage: weights[layer_idx][out_ch][kernel_pos][in_ch]
        self.weights = []
        self.biases = []

        for config in layers_config:
            in_ch = config["in_channels"]
            out_ch = config["out_channels"]
            k_size = config["kernel_size"]

            # Initialize with placeholder weights
            layer_weights = [
                [[0.0 for _ in range(in_ch)] for _ in range(k_size)] for _ in range(out_ch)
            ]
            layer_biases = [0.0 for _ in range(out_ch)]

            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    def set_weights(self, layer_idx: int, weights: list[list[list[float]]], biases: list[float]):
        """Set weights for a specific layer"""
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer {layer_idx} out of range")

        self.weights[layer_idx] = weights
        self.biases[layer_idx] = biases

    def forward(self, x: list[list[float]]) -> list[float]:
        """
        Multi-layer convolution with aggressive optimization
        x: [seq_len][in_channels] for first layer
        Returns: [out_channels] after global max pooling
        """
        seq_len = len(x)
        current_input = x

        for layer_idx in range(self.num_layers):
            current_input = self._forward_single_layer(layer_idx, current_input, seq_len)

        # Global max pooling on final layer
        final_channels = len(current_input)
        max_pooled = []

        for ch in range(final_channels):
            max_val = -1e30
            for t in range(seq_len):
                if current_input[ch][t] > max_val:
                    max_val = current_input[ch][t]
            max_pooled.append(max_val)

        return max_pooled

    def forward_single_layer(
        self, input_data: list[list[float]], layer_idx: int
    ) -> list[list[float]]:
        """Public interface for single layer forward pass"""
        seq_len = len(input_data) if input_data else 0
        return self._forward_single_layer(layer_idx, input_data, seq_len)

    def _forward_single_layer(
        self, layer_idx: int, input_data: list[list[float]], seq_len: int
    ) -> list[list[float]]:
        """
        Optimized single layer forward pass
        Returns: [out_channels][seq_len] format for next layer
        """
        config = self.layers_config[layer_idx]
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        padding = kernel_size // 2  # Same padding

        weights = self.weights[layer_idx]
        biases = self.biases[layer_idx]
        output = self.layer_outputs[layer_idx]  # Reuse pre-allocated memory

        # Reset output buffer for this sequence length
        for ch in range(out_channels):
            for t in range(seq_len):
                output[ch][t] = 0.0

        # Optimized convolution with loop reordering for cache efficiency
        for out_ch in range(out_channels):
            bias = biases[out_ch]
            w_out = weights[out_ch]

            for t in range(seq_len):
                conv_sum = bias

                # Kernel loop with bounds checking
                for k in range(kernel_size):
                    input_t = t + k - padding
                    if 0 <= input_t < seq_len:
                        w_k = w_out[k]

                        # Inner product with input channels
                        if layer_idx == 0:
                            # First layer: input_data is [seq_len][in_channels]
                            for in_ch in range(in_channels):
                                conv_sum += w_k[in_ch] * input_data[input_t][in_ch]
                        else:
                            # Later layers: input_data is [out_channels][seq_len]
                            for in_ch in range(in_channels):
                                conv_sum += w_k[in_ch] * input_data[in_ch][input_t]

                # ReLU activation inline
                output[out_ch][t] = conv_sum if conv_sum > 0.0 else 0.0

        return output


class OptimizedEmbedding:
    """Memory-efficient embedding with pre-computed lookups"""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Pre-allocate embedding table
        self.weight = [[0.0 for _ in range(embed_dim)] for _ in range(vocab_size)]

    def forward(self, indices: list[int], max_len: int | None = None) -> list[list[float]]:
        """
        Optimized embedding lookup
        Returns: [seq_len][embed_dim]
        """
        seq_len = len(indices)
        if max_len:
            seq_len = min(seq_len, max_len)

        # Direct lookup without intermediate allocations
        result = []
        for i in range(seq_len):
            idx = indices[i]
            if 0 <= idx < self.vocab_size:
                # Copy embedding vector
                result.append(self.weight[idx][:])  # Slice copy for safety
            else:
                # Zero embedding for out-of-vocab
                result.append([0.0] * self.embed_dim)

        return result


class OptimizedDense:
    """Cache-optimized dense layer"""

    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
        self.bias = [0.0 for _ in range(out_dim)]

    def forward(self, x: list[float]) -> list[float]:
        """Optimized matrix-vector multiplication"""
        if len(x) != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} dims, got {len(x)}")

        result = []
        for out_idx in range(self.out_dim):
            # Dot product with bias
            dot = self.bias[out_idx]
            weight_row = self.weight[out_idx]

            # Unrolled when possible for small dimensions
            if self.in_dim <= 32:
                # Manual unrolling for better performance
                for in_idx in range(self.in_dim):
                    dot += weight_row[in_idx] * x[in_idx]
            else:
                # Standard loop for larger dimensions
                for in_idx in range(self.in_dim):
                    dot += weight_row[in_idx] * x[in_idx]

            result.append(dot)

        return result


class BatchNorm1D:
    """Optimized Batch Normalization for inference"""

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.weight = [1.0] * num_features
        self.bias = [0.0] * num_features
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.eps = 1e-5

    def load_parameters(
        self,
        weight: list[float],
        bias: list[float],
        running_mean: list[float],
        running_var: list[float],
    ):
        """Load batch norm parameters"""
        self.weight = weight[:]
        self.bias = bias[:]
        self.running_mean = running_mean[:]
        self.running_var = running_var[:]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        """Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias"""
        batch_size = len(x)
        result = [[0.0 for _ in range(self.num_features)] for _ in range(batch_size)]

        # Pre-compute normalization factors
        norm_factors = []
        for i in range(self.num_features):
            std = (self.running_var[i] + self.eps) ** 0.5
            scale = self.weight[i] / std
            shift = self.bias[i] - (self.running_mean[i] * scale)
            norm_factors.append((scale, shift))

        # Apply normalization
        for batch_idx in range(batch_size):
            for feat_idx in range(self.num_features):
                scale, shift = norm_factors[feat_idx]
                result[batch_idx][feat_idx] = x[batch_idx][feat_idx] * scale + shift

        return result


class MultiLayerByteCNN:
    """
    Optimized Multi-Layer ByteCNN for Edge Deployment
    Supports 1, 2, or 3 conv layers with aggressive Python optimizations
    """

    def __init__(
        self,
        layers_config: list[dict[str, Any]],
        hidden_dim: int,
        max_len: int = 512,
        vocab_size: int = 256,
    ):
        """
        layers_config: List of conv layer configurations
        Example for 2-layer 32KB model:
        [{"in_channels": 14, "out_channels": 28, "kernel_size": 3},
         {"in_channels": 28, "out_channels": 24, "kernel_size": 3}]
        """

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.layers_config = layers_config
        self.hidden_dim = hidden_dim

        # Build components
        embed_dim = layers_config[0]["in_channels"]
        final_conv_dim = layers_config[-1]["out_channels"]

        self.embedding = OptimizedEmbedding(vocab_size, embed_dim)
        self.multi_conv = FusedMultiConv1D(layers_config, max_len)

        # Batch normalization layers for each conv layer
        self.batch_norms = []
        for layer_config in layers_config:
            bn = BatchNorm1D(layer_config["out_channels"])
            self.batch_norms.append(bn)

        self.classifier = OptimizedDense(final_conv_dim, hidden_dim)
        self.output = OptimizedDense(hidden_dim, 1)

        print(
            f"📊 MultiLayerByteCNN: {len(layers_config)} layers, {embed_dim}D→{final_conv_dim}D→{hidden_dim}D→1"
        )

    def forward_indices(self, indices: list[int]) -> float:
        """Optimized forward pass - uses fused kernel for 1-layer models"""

        # FAST PATH: Use original fused kernel for 1-layer models (restores ~0.2ms performance)
        if len(self.layers_config) == 1 and not self.layers_config[0].get("use_batch_norm", False):
            return self._forward_fused_1layer(indices)

        # SLOW PATH: Multi-layer with batch normalization
        return self._forward_multilayer_bn(indices)

    def _forward_fused_1layer(self, indices: list[int]) -> float:
        """Fast path using original fused Conv1D+ReLU+MaxPool kernel"""
        from .layers import Conv1DMaxPool, Dense, Sigmoid

        # Create fused conv layer (cached for reuse)
        if not hasattr(self, "_fused_conv"):
            config = self.layers_config[0]
            self._fused_conv = Conv1DMaxPool(
                in_dim=len(self.embedding.weight[0]),
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"],
                padding=config["kernel_size"] // 2,
            )
            # Load weights into fused conv
            self._load_fused_weights()

        if not hasattr(self, "_fused_classifier"):
            self._fused_classifier = Dense(self.layers_config[0]["out_channels"], self.hidden_dim)
            self._fused_output = Dense(self.hidden_dim, 1)
            self._fused_sigmoid = Sigmoid()
            # Load dense weights
            self._load_fused_dense_weights()

        # 1. Optimized embedding lookup (truncate indices first)
        seq_len = min(len(indices), self.max_len)
        embedded = []
        for i in range(seq_len):
            idx = indices[i]
            if 0 <= idx < self.vocab_size:
                embedded.append(self.embedding.weight[idx][:])  # Fast slice copy
            else:
                embedded.append([0.0] * len(self.embedding.weight[0]))

        # 2. Fused Conv1D + ReLU + MaxPool (single optimized kernel!)
        conv_output = self._fused_conv.forward(embedded)

        # 3. Dense layers with inlined ReLU for speed
        hidden = self._fused_classifier.forward(conv_output)
        # Inline ReLU instead of function call
        for i in range(len(hidden)):
            if hidden[i] < 0.0:
                hidden[i] = 0.0

        logits = self._fused_output.forward(hidden)
        probs = self._fused_sigmoid.forward(logits)

        return probs[0]

    def _load_fused_weights(self):
        """Load weights from multi-layer format into fused Conv1DMaxPool"""
        layer_weights = self.multi_conv.weights[0]  # First layer weights
        layer_biases = self.multi_conv.biases[0]  # First layer biases

        # Convert from [out_ch][kernel][in_ch] to Conv1DMaxPool format [out_ch][kernel][in_ch]
        self._fused_conv.weight = [
            [
                [layer_weights[f][k][c] for c in range(len(layer_weights[f][k]))]
                for k in range(len(layer_weights[f]))
            ]
            for f in range(len(layer_weights))
        ]

        if self._fused_conv.use_bias:
            self._fused_conv.bias = [float(b) for b in layer_biases]

    def _load_fused_dense_weights(self):
        """Load dense layer weights from multi-layer format into fused Dense layers"""
        # Classifier weights
        for out_idx in range(len(self.classifier.weight)):
            for in_idx in range(len(self.classifier.weight[out_idx])):
                self._fused_classifier.weight[out_idx][in_idx] = self.classifier.weight[out_idx][
                    in_idx
                ]

        for i in range(len(self.classifier.bias)):
            self._fused_classifier.bias[i] = self.classifier.bias[i]

        # Output weights
        for out_idx in range(len(self.output.weight)):
            for in_idx in range(len(self.output.weight[out_idx])):
                self._fused_output.weight[out_idx][in_idx] = self.output.weight[out_idx][in_idx]

        for i in range(len(self.output.bias)):
            self._fused_output.bias[i] = self.output.bias[i]

    def _forward_multilayer_bn(self, indices: list[int]) -> float:
        """Slow path for multi-layer models with batch normalization"""
        # 1. Embedding lookup
        embedded = self.embedding.forward(indices, self.max_len)

        # 2. Layer-by-layer convolution with BN and ReLU
        current_input = embedded

        for layer_idx in range(len(self.layers_config)):
            # Apply convolution for this layer
            conv_output = self.multi_conv.forward_single_layer(current_input, layer_idx)

            # Apply batch normalization (reshape for BN)
            conv_reshaped = [
                [conv_output[ch][t] for ch in range(len(conv_output))]
                for t in range(len(conv_output[0]))
            ]
            normalized = self.batch_norms[layer_idx].forward(conv_reshaped)

            # Reshape back and apply ReLU
            conv_bn = [
                [normalized[t][ch] for t in range(len(normalized))]
                for ch in range(len(normalized[0]))
            ]
            current_input = [[max(0.0, val) for val in row] for row in conv_bn]

        # 3. Global average + max pooling
        seq_len = len(current_input[0])
        pooled_features = []
        for ch in range(len(current_input)):
            if seq_len > 0:
                avg_val = sum(current_input[ch]) / seq_len
                max_val = max(current_input[ch])
                pooled_features.append((avg_val + max_val) / 2.0)
            else:
                pooled_features.append(0.0)

        # 4. Classifier with ReLU
        hidden = self.classifier.forward(pooled_features)
        hidden_relu = relu_vec(hidden)

        # 5. Output + sigmoid
        logits = self.output.forward(hidden_relu)
        probs = sigmoid_vec(logits)

        return probs[0]

    def predict(self, text: str, strategy: str = "truncate") -> float:
        """Prediction with sliding window strategies"""
        from .utils import text_to_bytes

        if strategy == "truncate":
            from .utils import text_to_fixed_bytes

            indices = text_to_fixed_bytes(text, self.max_len)
            return self.forward_indices(indices)

        elif strategy in ["average", "attention"]:
            return self._predict_sliding_window(text, strategy)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_sliding_window(self, text: str, strategy: str) -> float:
        """Sliding window prediction for long texts"""
        from .utils import text_to_bytes

        byte_data = text_to_bytes(text)

        if len(byte_data) <= self.max_len:
            # Pad and predict normally
            padded = byte_data + [0] * (self.max_len - len(byte_data))
            return self.forward_indices(padded)

        # Create windows
        stride = self.max_len // 2
        windows = []

        for start in range(0, len(byte_data) - self.max_len + 1, stride):
            window = byte_data[start : start + self.max_len]
            windows.append(window)

        if len(byte_data) % stride != 0:
            final_window = byte_data[-self.max_len :]
            windows.append(final_window)

        # Get predictions for each window
        predictions = [self.forward_indices(window) for window in windows]

        if strategy == "average":
            return sum(predictions) / len(predictions)

        elif strategy == "attention":
            # Attention weighting based on prediction confidence
            import math

            # Convert to logits for weighting
            logits = [math.log(p / (1 - p + 1e-8)) for p in predictions]

            # Softmax attention
            max_logit = max(logits)
            exp_logits = [math.exp(logit - max_logit) for logit in logits]
            sum_exp = sum(exp_logits)
            weights = [e / sum_exp for e in exp_logits]

            # Weighted prediction
            return sum(w * p for w, p in zip(weights, predictions, strict=False))

    @classmethod
    def create_2layer_32kb(cls, max_len: int = 512):
        """Factory method for optimal 2-layer 32KB model (F1=0.744)"""
        layers_config = [
            {"in_channels": 14, "out_channels": 28, "kernel_size": 3},
            {"in_channels": 28, "out_channels": 24, "kernel_size": 3},
        ]
        return cls(layers_config, hidden_dim=48, max_len=max_len)

    @classmethod
    def create_3layer_32kb(cls, max_len: int = 512):
        """Factory method for 3-layer 32KB model (for comparison)"""
        layers_config = [
            {"in_channels": 14, "out_channels": 28, "kernel_size": 3},
            {"in_channels": 28, "out_channels": 28, "kernel_size": 5},  # Larger kernel middle
            {"in_channels": 28, "out_channels": 24, "kernel_size": 3},
        ]
        return cls(layers_config, hidden_dim=48, max_len=max_len)

    def load_weights_from_dict(self, weights: dict[str, Any]):
        """Load weights from PyTorch model conversion"""

        # Load embedding weights
        if "embedding" in weights:
            embed_weights = weights["embedding"]
            for i, row in enumerate(embed_weights):
                if i < self.vocab_size:
                    for j, val in enumerate(row):
                        if j < len(self.embedding.weight[i]):
                            self.embedding.weight[i][j] = float(val)

        # Load conv layer weights
        layer_names = ["conv1_weight", "conv2_weight", "conv3_weight"]
        bias_names = ["conv1_bias", "conv2_bias", "conv3_bias"]

        for layer_idx in range(len(self.layers_config)):
            if layer_idx < len(layer_names):
                # Weights: convert from [out_ch][in_ch][kernel] to [out_ch][kernel][in_ch]
                weight_key = layer_names[layer_idx]
                bias_key = bias_names[layer_idx]

                if weight_key in weights:
                    conv_weights = weights[weight_key]
                    converted_weights = []

                    for _out_ch, out_data in enumerate(conv_weights):
                        kernel_data = []
                        for k_pos in range(len(out_data[0])):  # kernel_size
                            in_ch_data = []
                            for in_ch in range(len(out_data)):
                                in_ch_data.append(float(out_data[in_ch][k_pos]))
                            kernel_data.append(in_ch_data)
                        converted_weights.append(kernel_data)

                    biases = (
                        [float(b) for b in weights[bias_key]]
                        if bias_key in weights
                        else [0.0] * self.layers_config[layer_idx]["out_channels"]
                    )

                    self.multi_conv.set_weights(layer_idx, converted_weights, biases)

        # Load batch normalization parameters
        bn_names = ["bn1", "bn2", "bn3"]
        for layer_idx in range(len(self.layers_config)):
            if layer_idx < len(bn_names):
                bn_name = bn_names[layer_idx]

                # Check for BN parameters
                bn_weight_key = f"{bn_name}_weight"
                bn_bias_key = f"{bn_name}_bias"
                bn_mean_key = f"{bn_name}_running_mean"
                bn_var_key = f"{bn_name}_running_var"

                if all(
                    key in weights for key in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]
                ):
                    self.batch_norms[layer_idx].load_parameters(
                        weights[bn_weight_key],
                        weights[bn_bias_key],
                        weights[bn_mean_key],
                        weights[bn_var_key],
                    )

        # Load dense layer weights
        if "classifier_weight" in weights and "classifier_bias" in weights:
            cls_w = weights["classifier_weight"]
            cls_b = weights["classifier_bias"]

            for out_idx, row in enumerate(cls_w):
                if out_idx < len(self.classifier.weight):
                    for in_idx, val in enumerate(row):
                        if in_idx < len(self.classifier.weight[out_idx]):
                            self.classifier.weight[out_idx][in_idx] = float(val)

            for i, val in enumerate(cls_b):
                if i < len(self.classifier.bias):
                    self.classifier.bias[i] = float(val)

        # Load output layer weights
        if "output_weight" in weights and "output_bias" in weights:
            out_w = weights["output_weight"]
            out_b = weights["output_bias"]

            for out_idx, row in enumerate(out_w):
                if out_idx < len(self.output.weight):
                    for in_idx, val in enumerate(row):
                        if in_idx < len(self.output.weight[out_idx]):
                            self.output.weight[out_idx][in_idx] = float(val)

            for i, val in enumerate(out_b):
                if i < len(self.output.bias):
                    self.output.bias[i] = float(val)

        print("✅ Weights loaded successfully")


# Performance optimizations for specific architectures
class OptimizedArchitectures:
    """Pre-optimized architectures for common use cases"""

    @staticmethod
    def create_production_model(target_f1: float = 0.74) -> MultiLayerByteCNN:
        """
        Create the optimal production model based on our sweep results
        F1=0.744 with 2-layer 32KB architecture
        """
        if target_f1 >= 0.74:
            return MultiLayerByteCNN.create_2layer_32kb(max_len=512)
        else:
            # For lower accuracy requirements, use smaller model
            layers_config = [{"in_channels": 12, "out_channels": 20, "kernel_size": 3}]
            return MultiLayerByteCNN(layers_config, hidden_dim=40, max_len=512)

    @staticmethod
    def benchmark_inference_speed(
        model: MultiLayerByteCNN, num_samples: int = 1000
    ) -> dict[str, float]:
        """Benchmark inference speed for optimization"""
        import random
        import time

        # Generate test samples
        test_texts = [
            "This is a test message for benchmarking inference speed performance",
            "Short text",
            "Very long text that should trigger sliding window processing with multiple windows "
            * 20,
        ]

        results = {}

        for strategy in ["truncate", "average", "attention"]:
            start_time = time.time()

            for _ in range(num_samples):
                text = random.choice(test_texts)
                _ = model.predict(text, strategy=strategy)

            elapsed = time.time() - start_time
            rps = num_samples / elapsed  # Requests per second

            results[f"{strategy}_rps"] = rps
            results[f"{strategy}_latency_ms"] = (elapsed / num_samples) * 1000

        return results
