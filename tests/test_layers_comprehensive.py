#!/usr/bin/env python3
"""
Comprehensive tests for tinybytecnn layers to reach 80% coverage
Focus on Conv1DMaxPool, Embedding, Dense, Sigmoid
"""

import unittest
import math
from tinybytecnn.layers import Conv1DMaxPool, Embedding, Dense, Sigmoid


class TestConv1DMaxPoolComprehensive(unittest.TestCase):
    """Comprehensive tests for Conv1DMaxPool layer"""

    def test_initialization_basic(self):
        """Test basic initialization"""
        conv = Conv1DMaxPool(in_dim=10, out_channels=20, kernel_size=3)
        self.assertEqual(conv.in_dim, 10)
        self.assertEqual(conv.out_channels, 20)
        self.assertEqual(conv.kernel_size, 3)
        self.assertEqual(conv.padding, 0)
        self.assertTrue(conv.use_bias)

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        conv = Conv1DMaxPool(in_dim=5, out_channels=10, kernel_size=5, 
                           padding=2, bias=False, init_scale=0.1)
        self.assertEqual(conv.in_dim, 5)
        self.assertEqual(conv.out_channels, 10)
        self.assertEqual(conv.kernel_size, 5)
        self.assertEqual(conv.padding, 2)
        self.assertFalse(conv.use_bias)
        self.assertIsNone(conv.bias)
        self.assertIsNone(conv.bias_grad)

    def test_initialization_validation(self):
        """Test initialization parameter validation"""
        with self.assertRaises(ValueError):
            Conv1DMaxPool(in_dim=0, out_channels=5, kernel_size=3)
        
        with self.assertRaises(ValueError):
            Conv1DMaxPool(in_dim=5, out_channels=0, kernel_size=3)
            
        with self.assertRaises(ValueError):
            Conv1DMaxPool(in_dim=5, out_channels=5, kernel_size=0)

    def test_forward_basic(self):
        """Test basic forward pass"""
        conv = Conv1DMaxPool(in_dim=3, out_channels=2, kernel_size=3, padding=1)
        
        # Input: [seq_len=5, in_dim=3]
        x = [[1.0, 2.0, 3.0] for _ in range(5)]
        
        result = conv.forward(x)
        self.assertEqual(len(result), 2)  # out_channels
        for val in result:
            self.assertIsInstance(val, float)

    def test_forward_input_validation(self):
        """Test forward pass input validation"""
        conv = Conv1DMaxPool(in_dim=3, out_channels=2, kernel_size=3)
        
        # Invalid input types
        with self.assertRaises(ValueError):
            conv.forward([])  # Empty
            
        with self.assertRaises(ValueError):
            conv.forward("invalid")  # Not list
            
        with self.assertRaises(ValueError):
            conv.forward([[1.0, 2.0]])  # Wrong dim

    def test_forward_sequence_too_short(self):
        """Test forward with sequence too short for kernel"""
        conv = Conv1DMaxPool(in_dim=3, out_channels=2, kernel_size=5)
        
        # seq_len=2, kernel=5, padding=0 -> effective_len=2 < 5
        x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        with self.assertRaises(ValueError) as cm:
            conv.forward(x)
        self.assertIn("too short", str(cm.exception))

    def test_forward_with_padding(self):
        """Test forward pass with different padding"""
        conv = Conv1DMaxPool(in_dim=2, out_channels=1, kernel_size=3, padding=1)
        
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = conv.forward(x)
        
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0], 0.0)  # ReLU output

    def test_zero_grad(self):
        """Test gradient zeroing"""
        conv = Conv1DMaxPool(in_dim=2, out_channels=2, kernel_size=3)
        
        # Set some gradients
        for f in range(conv.out_channels):
            for i in range(conv.kernel_size):
                for c in range(conv.in_dim):
                    conv.weight_grad[f][i][c] = 1.0
            if conv.bias_grad:
                conv.bias_grad[f] = 1.0
        
        conv.zero_grad()
        
        # Check all gradients are zero
        for f in range(conv.out_channels):
            for i in range(conv.kernel_size):
                for c in range(conv.in_dim):
                    self.assertEqual(conv.weight_grad[f][i][c], 0.0)
            if conv.bias_grad:
                self.assertEqual(conv.bias_grad[f], 0.0)

    def test_backward_without_forward(self):
        """Test backward fails without forward"""
        conv = Conv1DMaxPool(in_dim=2, out_channels=2, kernel_size=3)
        
        with self.assertRaises(RuntimeError):
            conv.backward([1.0, 1.0])

    def test_backward_basic(self):
        """Test basic backward pass"""
        conv = Conv1DMaxPool(in_dim=2, out_channels=2, kernel_size=3, padding=1)
        
        # Forward pass first
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        conv.forward(x)
        
        # Backward pass
        grad_out = [1.0, 0.5]
        grad_in = conv.backward(grad_out)
        
        self.assertEqual(len(grad_in), 3)  # seq_len
        self.assertEqual(len(grad_in[0]), 2)  # in_dim

    def test_backward_with_negative_activation(self):
        """Test backward pass when pre-activation is negative (ReLU off)"""
        conv = Conv1DMaxPool(in_dim=2, out_channels=1, kernel_size=3, padding=1)
        
        # Set weights to produce negative activations
        for f in range(conv.out_channels):
            for i in range(conv.kernel_size):
                for c in range(conv.in_dim):
                    conv.weight[f][i][c] = -1.0
        if conv.bias:
            conv.bias[0] = -10.0  # Large negative bias
        
        # Forward pass with positive inputs -> should get negative pre-activations
        x = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        result = conv.forward(x)
        
        # Should get 0.0 from ReLU
        self.assertEqual(result[0], 0.0)
        
        # Backward pass - should hit the continue statement for negative pre-activation
        grad_out = [1.0]
        grad_in = conv.backward(grad_out)
        
        self.assertEqual(len(grad_in), 3)  # seq_len
        self.assertEqual(len(grad_in[0]), 2)  # in_dim

    def test_parameters(self):
        """Test parameters access"""
        conv = Conv1DMaxPool(in_dim=2, out_channels=1, kernel_size=3)
        
        params = conv.parameters()
        self.assertIn('weight', params)
        self.assertIn('weight_grad', params)
        self.assertIn('bias', params)
        self.assertIn('bias_grad', params)


class TestEmbeddingComprehensive(unittest.TestCase):
    """Comprehensive tests for Embedding layer"""

    def test_initialization(self):
        """Test embedding initialization"""
        embed = Embedding(vocab_size=100, dim=50)
        self.assertEqual(embed.vocab_size, 100)
        self.assertEqual(embed.dim, 50)
        self.assertEqual(len(embed.weight), 100)
        self.assertEqual(len(embed.weight[0]), 50)

    def test_initialization_validation(self):
        """Test initialization parameter validation"""
        with self.assertRaises(ValueError):
            Embedding(vocab_size=0, dim=10)
            
        with self.assertRaises(ValueError):
            Embedding(vocab_size=10, dim=0)

    def test_forward_basic(self):
        """Test basic forward pass"""
        embed = Embedding(vocab_size=10, dim=5)
        
        indices = [0, 1, 2]
        result = embed.forward(indices)
        
        self.assertEqual(len(result), 3)  # sequence length
        self.assertEqual(len(result[0]), 5)  # embedding dim

    def test_forward_edge_cases(self):
        """Test forward with edge cases"""
        embed = Embedding(vocab_size=10, dim=5)
        
        # Empty indices
        result = embed.forward([])
        self.assertEqual(len(result), 0)
        
        # Single index
        result = embed.forward([5])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 5)

    def test_forward_index_validation(self):
        """Test forward index validation"""
        embed = Embedding(vocab_size=10, dim=5)
        
        # Index too high
        with self.assertRaises(ValueError) as cm:
            embed.forward([10])
        self.assertIn("out of range", str(cm.exception))
        
        # Negative index
        with self.assertRaises(ValueError) as cm:
            embed.forward([-1])
        self.assertIn("out of range", str(cm.exception))


class TestDenseComprehensive(unittest.TestCase):
    """Comprehensive tests for Dense layer"""

    def test_initialization(self):
        """Test dense layer initialization"""
        dense = Dense(in_dim=10, out_dim=5)
        self.assertEqual(dense.in_dim, 10)
        self.assertEqual(dense.out_dim, 5)
        self.assertEqual(len(dense.weight), 5)  # out_dim rows
        self.assertEqual(len(dense.weight[0]), 10)  # in_dim cols
        self.assertEqual(len(dense.bias), 5)

    def test_initialization_validation(self):
        """Test initialization parameter validation"""
        with self.assertRaises(ValueError):
            Dense(in_dim=0, out_dim=5)
            
        with self.assertRaises(ValueError):
            Dense(in_dim=5, out_dim=0)

    def test_forward_basic(self):
        """Test basic forward pass"""
        dense = Dense(in_dim=3, out_dim=2)
        
        x = [1.0, 2.0, 3.0]
        result = dense.forward(x)
        
        self.assertEqual(len(result), 2)  # out_dim
        for val in result:
            self.assertIsInstance(val, float)

    def test_forward_input_validation(self):
        """Test forward input validation"""
        dense = Dense(in_dim=3, out_dim=2)
        
        # Wrong input dimension
        with self.assertRaises(ValueError) as cm:
            dense.forward([1.0, 2.0])  # Only 2 inputs, expect 3
        self.assertIn("Expected input dim 3", str(cm.exception))

    def test_forward_computation(self):
        """Test forward computation is correct"""
        dense = Dense(in_dim=2, out_dim=1)
        
        # Set known weights and bias
        dense.weight[0] = [2.0, 3.0]
        dense.bias[0] = 1.0
        
        x = [4.0, 5.0]
        result = dense.forward(x)
        
        # Expected: 2*4 + 3*5 + 1 = 8 + 15 + 1 = 24
        self.assertAlmostEqual(result[0], 24.0, places=5)


class TestSigmoidComprehensive(unittest.TestCase):
    """Comprehensive tests for Sigmoid activation"""

    def test_forward_basic(self):
        """Test basic sigmoid forward pass"""
        sigmoid = Sigmoid()
        
        x = [0.0, 1.0, -1.0]
        result = sigmoid.forward(x)
        
        self.assertEqual(len(result), 3)
        
        # Check known values
        self.assertAlmostEqual(result[0], 0.5, places=5)  # sigmoid(0) = 0.5
        self.assertAlmostEqual(result[1], 1.0/(1.0 + math.exp(-1.0)), places=5)
        self.assertAlmostEqual(result[2], 1.0/(1.0 + math.exp(1.0)), places=5)

    def test_forward_edge_cases(self):
        """Test sigmoid with edge cases"""
        sigmoid = Sigmoid()
        
        # Empty input
        result = sigmoid.forward([])
        self.assertEqual(len(result), 0)
        
        # Large positive value
        result = sigmoid.forward([100.0])
        self.assertAlmostEqual(result[0], 1.0, places=5)
        
        # Large negative value  
        result = sigmoid.forward([-100.0])
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_forward_range(self):
        """Test sigmoid output range"""
        sigmoid = Sigmoid()
        
        test_values = [-10.0, -1.0, 0.0, 1.0, 10.0]
        result = sigmoid.forward(test_values)
        
        for val in result:
            self.assertGreater(val, 0.0)
            self.assertLess(val, 1.0)


if __name__ == '__main__':
    unittest.main()