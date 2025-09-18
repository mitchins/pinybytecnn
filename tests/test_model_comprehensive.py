#!/usr/bin/env python3
"""
Comprehensive tests for ByteCNN model to reach 80% coverage
Focus on core functionality and edge cases
"""

import unittest
import json
import tempfile
import os
from tinybytecnn.model import ByteCNN


class TestByteCNNModel(unittest.TestCase):
    """Test core ByteCNN model functionality"""

    def setUp(self):
        self.model = ByteCNN(
            vocab_size=256,
            embed_dim=16,
            conv_filters=32,
            conv_kernel_size=3,
            hidden_dim=64,
            output_dim=1,
            max_len=128,
        )

    def test_initialization(self):
        """Test model initialization parameters"""
        self.assertEqual(self.model.vocab_size, 256)
        self.assertEqual(self.model.embed_dim, 16)
        self.assertEqual(self.model.conv_filters, 32)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.output_dim, 1)
        self.assertEqual(self.model.max_len, 128)
        self.assertEqual(self.model.padding, 1)  # conv_kernel_size // 2

    def test_forward_indices_basic(self):
        """Test basic forward pass with indices"""
        indices = [65, 66, 67, 68, 69]  # A-E
        result = self.model.forward_indices(indices)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_forward_indices_edge_cases(self):
        """Test forward pass with edge case inputs"""
        # Single character
        result = self.model.forward_indices([65])
        self.assertIsInstance(result, float)
        
        # Empty list - should still work due to zero padding in embedding
        result = self.model.forward_indices([0])  # Use zero index instead of empty
        self.assertIsInstance(result, float)
        
        # Maximum length
        max_indices = list(range(self.model.max_len))
        result = self.model.forward_indices(max_indices)
        self.assertIsInstance(result, float)

    def test_predict_truncate_strategy(self):
        """Test predict with truncate strategy"""
        text = "Hello world"
        result = self.model.predict(text, strategy="truncate")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_predict_average_strategy(self):
        """Test predict with average strategy"""
        text = "Short text"
        result = self.model.predict(text, strategy="average")
        self.assertIsInstance(result, float)

    def test_predict_attention_strategy(self):
        """Test predict with attention strategy"""
        text = "Short text"
        result = self.model.predict(text, strategy="attention")
        self.assertIsInstance(result, float)

    def test_predict_invalid_strategy(self):
        """Test predict with invalid strategy raises error"""
        with self.assertRaises(ValueError) as cm:
            self.model.predict("test", strategy="invalid")
        self.assertIn("Unknown strategy", str(cm.exception))

    def test_predict_sliding_window_short_text(self):
        """Test sliding window with text shorter than max_len"""
        short_text = "Hi"
        result_avg = self.model.predict_sliding_window(short_text, "average")
        result_att = self.model.predict_sliding_window(short_text, "attention")
        
        # Should fall back to truncate strategy
        result_truncate = self.model.predict(short_text, "truncate")
        self.assertAlmostEqual(result_avg, result_truncate, places=5)
        self.assertAlmostEqual(result_att, result_truncate, places=5)

    def test_predict_sliding_window_long_text(self):
        """Test sliding window with text longer than max_len"""
        # Create long text
        long_text = "This is a very long text. " * 20  # Should exceed max_len
        
        result_avg = self.model.predict_sliding_window(long_text, "average")
        result_att = self.model.predict_sliding_window(long_text, "attention")
        
        self.assertIsInstance(result_avg, float)
        self.assertIsInstance(result_att, float)
        self.assertGreaterEqual(result_avg, 0.0)
        self.assertLessEqual(result_avg, 1.0)
        self.assertGreaterEqual(result_att, 0.0)
        self.assertLessEqual(result_att, 1.0)

    def test_predict_sliding_window_invalid_strategy(self):
        """Test sliding window with invalid strategy"""
        with self.assertRaises(ValueError) as cm:
            self.model.predict_sliding_window("test", "invalid")
        self.assertIn("Unknown sliding window strategy", str(cm.exception))


class TestByteCNNWeightLoading(unittest.TestCase):
    """Test weight loading functionality"""

    def test_transpose_conv_weight(self):
        """Test conv weight transposition"""
        # Create test weights: [out_channels][in_channels][kernel]
        weights = [
            [[1.0, 2.0], [3.0, 4.0]],  # out_channel 0
            [[5.0, 6.0], [7.0, 8.0]]   # out_channel 1  
        ]
        
        result = ByteCNN._transpose_conv_weight(weights)
        
        # Should be: [out_channels][kernel][in_channels]
        self.assertEqual(len(result), 2)  # out_channels
        self.assertEqual(len(result[0]), 2)  # kernel_size
        self.assertEqual(len(result[0][0]), 2)  # in_channels
        
        # Check specific values
        self.assertEqual(result[0][0][0], 1.0)  # [0][0][0] = weights[0][0][0]
        self.assertEqual(result[0][1][0], 2.0)  # [0][1][0] = weights[0][0][1]

    def test_from_weight_dict_basic(self):
        """Test creating model from weight dictionary"""
        weights = {
            "embedding": [[0.1, 0.2] for _ in range(256)],  # vocab_size=256, embed_dim=2
            "conv1_weight": [[[0.1, 0.2], [0.3, 0.4]]],  # 1 filter, 2 in_ch, 2 kernel
            "conv1_bias": [0.1],
            "classifier_weight": [[0.5]],  # 1 out, 1 in (conv filters=1)
            "classifier_bias": [0.1],
            "output_weight": [[0.7]],  # 1 out, 1 in (hidden)
            "output_bias": [0.1]
        }
        
        model = ByteCNN.from_weight_dict(weights, max_len=64)
        
        self.assertEqual(model.vocab_size, 256)
        self.assertEqual(model.embed_dim, 2)
        self.assertEqual(model.conv_filters, 1)
        self.assertEqual(model.conv_kernel_size, 2)
        self.assertEqual(model.hidden_dim, 1)
        self.assertEqual(model.output_dim, 1)
        self.assertEqual(model.max_len, 64)

    def test_from_weight_dict_validation_errors(self):
        """Test weight dict validation errors"""
        base_weights = {
            "embedding": [[0.1, 0.2] for _ in range(256)],
            "conv1_weight": [[[0.1, 0.2], [0.3, 0.4]]],
            "conv1_bias": [0.1],
            "classifier_weight": [[0.5]],  # Correct: 1 conv filter -> 1 input
            "classifier_bias": [0.1],
            "output_weight": [[0.7]],
            "output_bias": [0.1]
        }
        
        # Test embed_dim mismatch
        bad_weights = base_weights.copy()
        bad_weights["conv1_weight"] = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]  # 3 in_ch vs 2 embed_dim
        with self.assertRaises(ValueError) as cm:
            ByteCNN.from_weight_dict(bad_weights)
        self.assertIn("conv in_channels", str(cm.exception))
        
        # Test classifier input mismatch
        bad_weights = base_weights.copy()
        bad_weights["classifier_weight"] = [[0.5, 0.6]]  # 2 in vs 1 conv_filters
        with self.assertRaises(ValueError) as cm:
            ByteCNN.from_weight_dict(bad_weights)
        self.assertIn("classifier in_dim", str(cm.exception))
        
        # Test output dim not 1
        bad_weights = base_weights.copy()
        bad_weights["output_bias"] = [0.1, 0.2]  # 2 outputs
        with self.assertRaises(ValueError) as cm:
            ByteCNN.from_weight_dict(bad_weights)
        self.assertIn("Only binary output supported", str(cm.exception))
        
        # Test output input mismatch
        bad_weights = base_weights.copy()
        bad_weights["output_weight"] = [[0.7, 0.8]]  # 2 in vs 1 hidden_dim
        with self.assertRaises(ValueError) as cm:
            ByteCNN.from_weight_dict(bad_weights)
        self.assertIn("output in_dim", str(cm.exception))

    def test_load_weights_json(self):
        """Test loading weights from JSON file"""
        test_weights = {
            "embedding": [[0.1, 0.2] for _ in range(10)],
            "conv1_weight": [[[0.1, 0.2]]],
            "conv1_bias": [0.1],
            "classifier_weight": [[0.5]],
            "classifier_bias": [0.1],
            "output_weight": [[0.7]],
            "output_bias": [0.1],
            "model_info": {"name": "test_model"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_weights, f)
            temp_path = f.name
        
        try:
            loaded_weights = ByteCNN.load_weights_json(temp_path)
            self.assertEqual(loaded_weights["model_info"]["name"], "test_model")
            self.assertEqual(len(loaded_weights["embedding"]), 10)
        finally:
            os.unlink(temp_path)

    def test_load_weights_npz_without_numpy(self):
        """Test NPZ loading fails gracefully without numpy"""
        # Create a temporary file that would be .npz
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            # This should raise RuntimeError about numpy
            with self.assertRaises(RuntimeError) as cm:
                ByteCNN.load_weights_npz(f.name)
            self.assertIn("NumPy is required", str(cm.exception))


if __name__ == '__main__':
    unittest.main()