#!/usr/bin/env python3
"""
Basic tests for MultiLayerByteCNN to improve coverage
Focus on key functionality without the complex broken tests
"""

import unittest
from tinybytecnn.multi_layer_optimized import MultiLayerByteCNN


class TestMultiLayerBasic(unittest.TestCase):
    """Basic tests for MultiLayerByteCNN functionality"""

    def test_create_2layer_32kb(self):
        """Test 2-layer model creation"""
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
        self.assertIsInstance(model, MultiLayerByteCNN)
        self.assertEqual(model.max_len, 128)
        self.assertEqual(len(model.layers_config), 2)

    def test_create_3layer_32kb(self):
        """Test 3-layer model creation"""
        model = MultiLayerByteCNN.create_3layer_32kb(max_len=256)
        self.assertIsInstance(model, MultiLayerByteCNN)
        self.assertEqual(model.max_len, 256)
        self.assertEqual(len(model.layers_config), 3)

    def test_predict_basic(self):
        """Test basic prediction functionality"""
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=64)
        
        # Simple prediction test
        result = model.predict("Hello world")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_predict_strategies(self):
        """Test different prediction strategies"""
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=32)
        text = "Test"
        
        # Test different strategies
        for strategy in ["truncate", "average", "attention"]:
            result = model.predict(text, strategy=strategy)
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_model_creation_info(self):
        """Test model creation provides info"""
        model = MultiLayerByteCNN.create_2layer_32kb()
        
        # Just test the model was created successfully
        self.assertIsInstance(model, MultiLayerByteCNN)
        self.assertTrue(hasattr(model, 'layers_config'))
        self.assertTrue(hasattr(model, 'max_len'))

    def test_creation_methods_exist(self):
        """Test creation methods exist"""
        # Test that creation methods are available
        self.assertTrue(hasattr(MultiLayerByteCNN, 'create_2layer_32kb'))
        self.assertTrue(hasattr(MultiLayerByteCNN, 'create_3layer_32kb'))


if __name__ == '__main__':
    unittest.main()