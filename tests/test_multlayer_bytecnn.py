#!/usr/bin/env python3
"""
Comprehensive unit tests for PinyByteCNN multi-layer optimized architecture
Targeting ≥85% coverage with maximum raw Python performance validation
"""

import unittest
import sys
import os
import json
import time
import math
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tinybytecnn.multi_layer_optimized import (
    MultiLayerByteCNN,
    FusedMultiConv1D, 
    OptimizedEmbedding,
    OptimizedDense,
    BatchNorm1D,
    relu_vec,
    sigmoid_vec
)


class TestActivationFunctions(unittest.TestCase):
    """Test core activation functions for correctness and performance"""
    
    def test_relu_vec_basic(self):
        """Test ReLU activation with basic inputs"""
        # Basic functionality
        result = relu_vec([1.0, -2.0, 0.0, 3.5, -0.1])
        expected = [1.0, 0.0, 0.0, 3.5, 0.0]
        self.assertEqual(result, expected)
        
        # Edge cases
        self.assertEqual(relu_vec([]), [])
        self.assertEqual(relu_vec([0.0]), [0.0])
        self.assertEqual(relu_vec([-1.0]), [0.0])
        self.assertEqual(relu_vec([float('inf')]), [float('inf')])
    
    def test_relu_vec_performance(self):
        """Test ReLU performance on large vectors"""
        # Performance test with 10K elements
        large_vec = [i - 5000 for i in range(10000)]  # Mix of pos/neg
        
        start_time = time.time()
        result = relu_vec(large_vec)
        elapsed = time.time() - start_time
        
        # Should be very fast (< 10ms for raw Python)
        self.assertLess(elapsed, 0.01)
        self.assertEqual(len(result), 10000)
        
        # Verify correctness on subset
        for i in range(0, 100, 10):
            expected = max(0.0, large_vec[i])
            self.assertEqual(result[i], expected)
    
    def test_sigmoid_vec_basic(self):
        """Test sigmoid activation with basic inputs"""
        result = sigmoid_vec([0.0, 1.0, -1.0, 2.0])
        
        # Check approximate values (sigmoid properties)
        self.assertAlmostEqual(result[0], 0.5, places=6)  # sigmoid(0) = 0.5
        self.assertGreater(result[1], 0.7)  # sigmoid(1) ≈ 0.73
        self.assertLess(result[2], 0.3)     # sigmoid(-1) ≈ 0.27
        self.assertGreater(result[3], 0.8)  # sigmoid(2) ≈ 0.88
    
    def test_sigmoid_vec_edge_cases(self):
        """Test sigmoid with extreme values"""
        result = sigmoid_vec([100.0, -100.0, 0.0])
        
        # Large positive should approach 1.0
        self.assertGreater(result[0], 0.99)
        # Large negative should approach 0.0
        self.assertLess(result[1], 0.01)
        # Zero should be exactly 0.5
        self.assertEqual(result[2], 0.5)


class TestOptimizedEmbedding(unittest.TestCase):
    """Test optimized embedding layer"""
    
    def setUp(self):
        self.vocab_size = 256
        self.embed_dim = 14
        self.embedding = OptimizedEmbedding(self.vocab_size, self.embed_dim)
    
    def test_initialization(self):
        """Test embedding layer initialization"""
        self.assertEqual(self.embedding.vocab_size, self.vocab_size)
        self.assertEqual(self.embedding.embed_dim, self.embed_dim)
        self.assertEqual(len(self.embedding.weight), self.vocab_size)
        self.assertEqual(len(self.embedding.weight[0]), self.embed_dim)
        
        # Check zero initialization
        self.assertEqual(self.embedding.weight[0][0], 0.0)
        self.assertEqual(self.embedding.weight[255][13], 0.0)
    
    def test_forward_basic(self):
        """Test embedding lookup functionality"""
        # Set some test weights
        self.embedding.weight[65] = [1.0, 2.0, 3.0] + [0.0] * 11  # 'A'
        self.embedding.weight[66] = [4.0, 5.0, 6.0] + [0.0] * 11  # 'B'
        
        # Test lookup
        indices = [65, 66, 0]
        result = self.embedding.forward(indices, max_len=10)
        
        self.assertEqual(len(result), 3)  # seq_len
        self.assertEqual(len(result[0]), self.embed_dim)
        
        # Check values
        self.assertEqual(result[0][:3], [1.0, 2.0, 3.0])
        self.assertEqual(result[1][:3], [4.0, 5.0, 6.0])
        self.assertEqual(result[2][:3], [0.0, 0.0, 0.0])
    
    def test_forward_performance(self):
        """Test embedding lookup performance"""
        # Long sequence
        indices = list(range(100)) * 5  # 500 elements
        
        start_time = time.time()
        result = self.embedding.forward(indices, max_len=512)
        elapsed = time.time() - start_time
        
        # Should be very fast for raw Python
        self.assertLess(elapsed, 0.01)
        self.assertEqual(len(result), 500)
    
    def test_forward_edge_cases(self):
        """Test embedding edge cases"""
        # Empty sequence
        result = self.embedding.forward([], max_len=10)
        self.assertEqual(len(result), 0)
        
        # Single element
        result = self.embedding.forward([100], max_len=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), self.embed_dim)
        
        # Out of bounds (should handle gracefully)
        with self.assertRaises(IndexError):
            self.embedding.forward([300], max_len=10)  # > vocab_size


class TestOptimizedDense(unittest.TestCase):
    """Test optimized dense layer"""
    
    def setUp(self):
        self.in_dim = 24
        self.out_dim = 48
        self.dense = OptimizedDense(self.in_dim, self.out_dim)
    
    def test_initialization(self):
        """Test dense layer initialization"""
        self.assertEqual(self.dense.in_dim, self.in_dim)
        self.assertEqual(self.dense.out_dim, self.out_dim)
        self.assertEqual(len(self.dense.weight), self.out_dim)
        self.assertEqual(len(self.dense.weight[0]), self.in_dim)
        self.assertEqual(len(self.dense.bias), self.out_dim)
        
        # Check zero initialization
        self.assertEqual(self.dense.weight[0][0], 0.0)
        self.assertEqual(self.dense.bias[0], 0.0)
    
    def test_forward_basic(self):
        """Test dense layer forward pass"""
        # Set test weights: identity-like
        for i in range(min(self.in_dim, self.out_dim)):
            self.dense.weight[i][i] = 1.0
        
        # Test input
        x = [1.0] * self.in_dim
        result = self.dense.forward(x)
        
        self.assertEqual(len(result), self.out_dim)
        
        # First 24 outputs should be 1.0 (identity mapping)
        for i in range(self.in_dim):
            self.assertAlmostEqual(result[i], 1.0, places=6)
        
        # Remaining outputs should be 0.0
        for i in range(self.in_dim, self.out_dim):
            self.assertEqual(result[i], 0.0)
    
    def test_forward_with_bias(self):
        """Test dense layer with bias"""
        # Set bias values
        for i in range(self.out_dim):
            self.dense.bias[i] = 0.5
        
        x = [0.0] * self.in_dim  # Zero input
        result = self.dense.forward(x)
        
        # All outputs should be bias values
        for val in result:
            self.assertEqual(val, 0.5)
    
    def test_forward_performance(self):
        """Test dense layer performance"""
        # Large dense layer for performance test
        large_dense = OptimizedDense(512, 256)
        x = [1.0] * 512
        
        start_time = time.time()
        for _ in range(100):  # 100 forward passes
            result = large_dense.forward(x)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 0.1)  # 100 passes in < 100ms
    
    def test_forward_dimension_mismatch(self):
        """Test dimension validation"""
        with self.assertRaises(ValueError):
            self.dense.forward([1.0] * (self.in_dim + 1))
        
        with self.assertRaises(ValueError):
            self.dense.forward([1.0] * (self.in_dim - 1))


class TestBatchNorm1D(unittest.TestCase):
    """Test batch normalization layer"""
    
    def setUp(self):
        self.num_features = 28
        self.bn = BatchNorm1D(self.num_features)
    
    def test_initialization(self):
        """Test batch norm initialization"""
        self.assertEqual(self.bn.num_features, self.num_features)
        self.assertEqual(len(self.bn.weight), self.num_features)
        self.assertEqual(len(self.bn.bias), self.num_features)
        self.assertEqual(len(self.bn.running_mean), self.num_features)
        self.assertEqual(len(self.bn.running_var), self.num_features)
        
        # Check default values
        self.assertEqual(self.bn.weight[0], 1.0)
        self.assertEqual(self.bn.bias[0], 0.0)
        self.assertEqual(self.bn.running_mean[0], 0.0)
        self.assertEqual(self.bn.running_var[0], 1.0)
    
    def test_load_parameters(self):
        """Test loading batch norm parameters"""
        weight = [2.0] * self.num_features
        bias = [0.5] * self.num_features
        mean = [1.0] * self.num_features
        var = [4.0] * self.num_features
        
        self.bn.load_parameters(weight, bias, mean, var)
        
        self.assertEqual(self.bn.weight, weight)
        self.assertEqual(self.bn.bias, bias)
        self.assertEqual(self.bn.running_mean, mean)
        self.assertEqual(self.bn.running_var, var)
    
    def test_forward_identity(self):
        """Test batch norm with identity parameters"""
        # Default parameters should apply identity transformation
        batch_size = 2
        x = [[1.0] * self.num_features, [2.0] * self.num_features]
        
        result = self.bn.forward(x)
        
        self.assertEqual(len(result), batch_size)
        self.assertEqual(len(result[0]), self.num_features)
        
        # With default params (mean=0, var=1, weight=1, bias=0):
        # output = (x - 0) / sqrt(1 + 1e-5) * 1 + 0 ≈ x
        for i in range(batch_size):
            for j in range(self.num_features):
                self.assertAlmostEqual(result[i][j], x[i][j], places=4)
    
    def test_forward_normalization(self):
        """Test batch norm normalization behavior"""
        # Set specific parameters
        self.bn.running_mean = [2.0] * self.num_features
        self.bn.running_var = [4.0] * self.num_features
        self.bn.weight = [2.0] * self.num_features
        self.bn.bias = [1.0] * self.num_features
        
        # Input that should be normalized
        x = [[4.0] * self.num_features]  # Single batch
        
        result = self.bn.forward(x)
        
        # Expected: (4 - 2) / sqrt(4 + 1e-5) * 2 + 1 ≈ 2 * 2 + 1 = 5
        for j in range(self.num_features):
            self.assertAlmostEqual(result[0][j], 5.0, places=3)
    
    def test_forward_performance(self):
        """Test batch norm performance"""
        # Large batch for performance test
        batch_size = 100
        x = [[1.0] * self.num_features for _ in range(batch_size)]
        
        start_time = time.time()
        result = self.bn.forward(x)
        elapsed = time.time() - start_time
        
        self.assertLess(elapsed, 0.01)  # Should be fast
        self.assertEqual(len(result), batch_size)


class TestFusedMultiConv1D(unittest.TestCase):
    """Test fused multi-layer convolution"""
    
    def setUp(self):
        self.layers_config = [
            {"in_channels": 14, "out_channels": 28, "kernel_size": 3},
            {"in_channels": 28, "out_channels": 24, "kernel_size": 3}
        ]
        self.max_seq_len = 512
        self.conv = FusedMultiConv1D(self.layers_config, self.max_seq_len)
    
    def test_initialization(self):
        """Test multi-conv initialization"""
        self.assertEqual(self.conv.layers_config, self.layers_config)
        self.assertEqual(self.conv.max_seq_len, self.max_seq_len)
        self.assertEqual(self.conv.num_layers, 2)
        self.assertEqual(len(self.conv.weights), 2)
        self.assertEqual(len(self.conv.biases), 2)
        self.assertEqual(len(self.conv.layer_outputs), 2)
        
        # Check pre-allocated buffers
        self.assertEqual(len(self.conv.layer_outputs[0]), 28)  # out_channels layer 1
        self.assertEqual(len(self.conv.layer_outputs[1]), 24)  # out_channels layer 2
        self.assertEqual(len(self.conv.layer_outputs[0][0]), self.max_seq_len)
    
    def test_set_weights(self):
        """Test weight setting functionality"""
        # Create test weights for layer 0
        layer_0_weights = [
            [[1.0, 2.0, 3.0] for _ in range(3)]  # kernel_size=3
            for _ in range(28)  # out_channels
        ]
        layer_0_biases = [0.5] * 28
        
        self.conv.set_weights(0, layer_0_weights, layer_0_biases)
        
        self.assertEqual(len(self.conv.weights[0]), 28)
        self.assertEqual(len(self.conv.weights[0][0]), 3)  # kernel_size
        self.assertEqual(len(self.conv.weights[0][0][0]), 14)  # in_channels
        self.assertEqual(self.conv.biases[0], layer_0_biases)
    
    def test_set_weights_invalid_layer(self):
        """Test weight setting with invalid layer index"""
        with self.assertRaises(ValueError):
            self.conv.set_weights(5, [], [])
    
    def test_forward_single_layer(self):
        """Test single layer forward pass"""
        # Simple test input
        seq_len = 10
        input_data = [[1.0] * 14 for _ in range(seq_len)]  # [seq_len][in_channels]
        
        # Set some test weights (identity-like)
        test_weights = [
            [[1.0 if k == 1 else 0.0 for _ in range(14)] for k in range(3)]
            for _ in range(28)
        ]
        test_biases = [0.0] * 28
        self.conv.set_weights(0, test_weights, test_biases)
        
        result = self.conv.forward_single_layer(input_data, 0)
        
        self.assertEqual(len(result), 28)  # out_channels
        self.assertEqual(len(result[0]), seq_len)  # seq_len preserved
    
    def test_forward_full_pipeline(self):
        """Test full multi-layer forward pass"""
        # Create realistic input
        seq_len = 50
        input_data = [[0.1 * i * j for j in range(14)] for i in range(seq_len)]
        
        # Set random-like weights
        for layer_idx in range(2):
            config = self.layers_config[layer_idx]
            out_ch = config["out_channels"]
            in_ch = config["in_channels"]
            kernel = config["kernel_size"]
            
            weights = [
                [[0.1 * (i + j + k) for _ in range(in_ch)] for k in range(kernel)]
                for i in range(out_ch)
            ]
            biases = [0.01 * i for i in range(out_ch)]
            self.conv.set_weights(layer_idx, weights, biases)
        
        result = self.conv.forward(input_data)
        
        # Should return pooled features from final layer
        self.assertEqual(len(result), 24)  # final out_channels
        
        # All values should be finite
        for val in result:
            self.assertTrue(math.isfinite(val))
    
    def test_forward_performance(self):
        """Test convolution performance"""
        # Performance test with realistic sequence
        seq_len = 256
        input_data = [[0.1 * i for _ in range(14)] for i in range(seq_len)]
        
        # Set minimal weights
        for layer_idx in range(2):
            config = self.layers_config[layer_idx]
            weights = [[[0.1] * config["in_channels"]] * config["kernel_size"]] * config["out_channels"]
            biases = [0.0] * config["out_channels"]
            self.conv.set_weights(layer_idx, weights, biases)
        
        start_time = time.time()
        result = self.conv.forward(input_data)
        elapsed = time.time() - start_time
        
        # Should be reasonably fast for raw Python
        self.assertLess(elapsed, 0.1)  # < 100ms
        self.assertEqual(len(result), 24)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        result = self.conv.forward([])
        self.assertEqual(len(result), 24)  # Should still return expected size
        
        # All values should be from biases (no conv contribution)
        for val in result:
            self.assertTrue(math.isfinite(val))


class TestMultiLayerByteCNN(unittest.TestCase):
    """Test complete multi-layer ByteCNN model"""
    
    def setUp(self):
        self.model_2layer = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
        self.model_3layer = MultiLayerByteCNN.create_3layer_32kb(max_len=128)
    
    def test_factory_methods(self):
        """Test factory method creation"""
        # 2-layer model
        self.assertEqual(len(self.model_2layer.layers_config), 2)
        self.assertEqual(self.model_2layer.layers_config[0]["in_channels"], 14)
        self.assertEqual(self.model_2layer.layers_config[0]["out_channels"], 28)
        self.assertEqual(self.model_2layer.layers_config[1]["out_channels"], 24)
        
        # 3-layer model  
        self.assertEqual(len(self.model_3layer.layers_config), 3)
        self.assertEqual(self.model_3layer.layers_config[2]["out_channels"], 24)
    
    def test_forward_indices_basic(self):
        """Test forward pass with byte indices"""
        # Simple test indices (ASCII characters)
        indices = [72, 101, 108, 108, 111]  # "Hello"
        
        result = self.model_2layer.forward_indices(indices)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
        
        # Should be deterministic with same input
        result2 = self.model_2layer.forward_indices(indices)
        self.assertEqual(result, result2)
    
    def test_forward_indices_variable_lengths(self):
        """Test forward pass with different sequence lengths"""
        test_lengths = [1, 10, 50, 127, 128]  # Including max_len
        
        for length in test_lengths:
            indices = list(range(length))
            result = self.model_2layer.forward_indices(indices)
            
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    def test_forward_indices_empty(self):
        """Test forward pass with empty input"""
        result = self.model_2layer.forward_indices([])
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_predict_basic(self):
        """Test text prediction functionality"""
        text = "Hello world!"
        
        result_truncate = self.model_2layer.predict(text, strategy="truncate")
        result_average = self.model_2layer.predict(text, strategy="average")
        
        # Both should return valid probabilities
        for result in [result_truncate, result_average]:
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    def test_predict_strategies(self):
        """Test different prediction strategies"""
        # Long text to test sliding window
        long_text = "This is a long message. " * 50  # ~1200 chars, triggers sliding window
        
        results = {}
        for strategy in ["truncate", "average", "attention"]:
            result = self.model_2layer.predict(long_text, strategy=strategy)
            results[strategy] = result
            
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
        
        # Results might differ between strategies
        # (can't assert equality due to different processing)
    
    def test_predict_edge_cases(self):
        """Test prediction with edge case inputs"""
        edge_cases = [
            "",                    # Empty string
            "A",                   # Single char
            "🚀🎯💻",              # Emojis only
            "Hello 世界!",          # Mixed unicode
            "\n\t  ",              # Whitespace only
            "A" * 1000,            # Very long
        ]
        
        for text in edge_cases:
            result = self.model_2layer.predict(text, strategy="truncate")
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    def test_load_weights_from_dict(self):
        """Test weight loading functionality"""
        # Create minimal weight dict
        weights_dict = {
            "embedding": [[0.1 * i * j for j in range(14)] for i in range(256)],
            "conv1_weight": [[[0.1] * 14] * 3] * 28,
            "conv1_bias": [0.01] * 28,
            "conv2_weight": [[[0.1] * 28] * 3] * 24,
            "conv2_bias": [0.01] * 24,
            "classifier_weight": [[0.1] * 24] * 48,
            "classifier_bias": [0.01] * 48,
            "output_weight": [[0.1] * 48],
            "output_bias": [0.01],
            # Batch norm parameters
            "bn1_weight": [1.0] * 28,
            "bn1_bias": [0.0] * 28,
            "bn1_running_mean": [0.0] * 28,
            "bn1_running_var": [1.0] * 28,
            "bn2_weight": [1.0] * 24,
            "bn2_bias": [0.0] * 24,
            "bn2_running_mean": [0.0] * 24,
            "bn2_running_var": [1.0] * 24,
        }
        
        # Should load without error
        self.model_2layer.load_weights_from_dict(weights_dict)
        
        # Test that loaded weights affect prediction
        result = self.model_2layer.forward_indices([65, 66, 67])  # ABC
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_model_performance(self):
        """Test model inference performance"""
        # Performance test
        test_text = "This is a test message for performance evaluation."
        
        # Warmup
        for _ in range(5):
            self.model_2layer.predict(test_text, strategy="truncate")
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            result = self.model_2layer.predict(test_text, strategy="truncate")
        elapsed = time.time() - start_time
        
        # Should achieve reasonable throughput
        rps = 100 / elapsed
        self.assertGreater(rps, 50)  # > 50 RPS for truncate strategy
        
        print(f"Model performance: {rps:.1f} RPS ({elapsed*10:.1f}ms per prediction)")
    
    def test_layer_depth_variations(self):
        """Test models with different layer depths"""
        # Test 1, 2, and 3 layer configurations
        configs = [
            # 1-layer config
            [{"in_channels": 14, "out_channels": 32, "kernel_size": 3}],
            # 2-layer config (default)
            [{"in_channels": 14, "out_channels": 28, "kernel_size": 3},
             {"in_channels": 28, "out_channels": 24, "kernel_size": 3}],
            # 3-layer config
            [{"in_channels": 14, "out_channels": 32, "kernel_size": 3},
             {"in_channels": 32, "out_channels": 28, "kernel_size": 3},
             {"in_channels": 28, "out_channels": 24, "kernel_size": 3}]
        ]
        
        for i, config in enumerate(configs):
            model = MultiLayerByteCNN(config, hidden_dim=48, max_len=128)
            
            # Test basic functionality
            result = model.forward_indices([72, 101, 108, 108, 111])
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
            
            print(f"{len(config)}-layer model: {result:.4f}")
    
    def test_memory_efficiency(self):
        """Test memory usage and pre-allocated buffers"""
        # Check that pre-allocated buffers are reused
        initial_buffers = [
            id(self.model_2layer.multi_conv.layer_outputs[0]),
            id(self.model_2layer.multi_conv.layer_outputs[1])
        ]
        
        # Multiple predictions shouldn't change buffer IDs
        for _ in range(10):
            self.model_2layer.forward_indices([65, 66, 67])
        
        final_buffers = [
            id(self.model_2layer.multi_conv.layer_outputs[0]),
            id(self.model_2layer.multi_conv.layer_outputs[1])
        ]
        
        self.assertEqual(initial_buffers, final_buffers)
        print("✅ Memory buffers properly reused")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Create model
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=512)
        
        # Load some weights (minimal for functionality test)
        weights_dict = {
            "embedding": [[0.01 * (i % 10) * (j % 10) for j in range(14)] for i in range(256)],
            "conv1_weight": [[[0.1 if i == j else 0.01 for _ in range(14)] for k in range(3)] for i in range(28) for j in range(1)],
            "conv1_bias": [0.001 * i for i in range(28)],
            "conv2_weight": [[[0.1 if i == j else 0.01 for _ in range(28)] for k in range(3)] for i in range(24) for j in range(1)],
            "conv2_bias": [0.001 * i for i in range(24)],
            "classifier_weight": [[0.01 if i == j else 0.001 for j in range(24)] for i in range(48)],
            "classifier_bias": [0.001] * 48,
            "output_weight": [[0.1] * 48],
            "output_bias": [0.5],  # Bias towards middle range
            "bn1_weight": [1.0] * 28,
            "bn1_bias": [0.0] * 28,
            "bn1_running_mean": [0.0] * 28,
            "bn1_running_var": [1.0] * 28,
            "bn2_weight": [1.0] * 24,
            "bn2_bias": [0.0] * 24,
            "bn2_running_mean": [0.0] * 24,
            "bn2_running_var": [1.0] * 24,
        }
        
        model.load_weights_from_dict(weights_dict)
        
        # Test various inputs
        test_cases = [
            "Hello world!",
            "This is a toxic message",
            "Normal conversation",
            "",
            "🚀" * 100,  # Long emoji sequence
            "A" * 1000,   # Very long text
        ]
        
        results = []
        for text in test_cases:
            for strategy in ["truncate", "average"]:
                result = model.predict(text, strategy=strategy)
                results.append(result)
                
                # Basic sanity checks
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0.0)
                self.assertLessEqual(result, 1.0)
        
        print(f"End-to-end test completed: {len(results)} predictions generated")
        print(f"Result range: {min(results):.4f} - {max(results):.4f}")
    
    def test_model_consistency(self):
        """Test model consistency across multiple runs"""
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=256)
        
        test_text = "Consistency test message"
        
        # Multiple predictions should be identical (deterministic)
        results = []
        for _ in range(10):
            result = model.predict(test_text, strategy="truncate")
            results.append(result)
        
        # All results should be exactly equal
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])
        
        print(f"✅ Model consistency verified: {results[0]:.6f}")


def run_performance_benchmarks():
    """Additional performance benchmarks"""
    print("\n🏃 Running Performance Benchmarks")
    print("=" * 50)
    
    # Create models
    model_2layer = MultiLayerByteCNN.create_2layer_32kb(max_len=512)
    model_3layer = MultiLayerByteCNN.create_3layer_32kb(max_len=512)
    
    test_cases = [
        ("Short text", "Hello!"),
        ("Medium text", "This is a medium length message for testing."),
        ("Long text", "This is a much longer message that should trigger different processing strategies. " * 10),
    ]
    
    for model_name, model in [("2-layer", model_2layer), ("3-layer", model_3layer)]:
        print(f"\n{model_name} Model:")
        
        for case_name, text in test_cases:
            # Warmup
            for _ in range(5):
                model.predict(text, strategy="truncate")
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                result = model.predict(text, strategy="truncate")
            elapsed = time.time() - start_time
            
            rps = 100 / elapsed
            print(f"  {case_name:12s}: {rps:6.1f} RPS ({elapsed*10:5.1f}ms/100)")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional benchmarks
    run_performance_benchmarks()