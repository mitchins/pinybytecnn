#!/usr/bin/env python3
"""
Unit tests for PinyByteCNN accuracy validation module
Testing tinygrad-style safety/accuracy checks
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tinybytecnn.accuracy_validation import (
    AccuracyValidator,
    validate_optimized_model
)
from tinybytecnn.multi_layer_optimized import MultiLayerByteCNN


class TestAccuracyValidator(unittest.TestCase):
    """Test accuracy validation functionality"""
    
    def setUp(self):
        self.validator = AccuracyValidator(tolerance_pct=2.0)
        self.model = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
    
    def test_initialization(self):
        """Test validator initialization"""
        self.assertEqual(self.validator.tolerance, 0.02)  # 2% -> 0.02
        self.assertIsInstance(self.validator.test_cases, list)
        self.assertGreater(len(self.validator.test_cases), 0)
        
        # Check test case format
        for text, expected_prob in self.validator.test_cases:
            self.assertIsInstance(text, str)
            self.assertIsInstance(expected_prob, float)
            self.assertGreaterEqual(expected_prob, 0.0)
            self.assertLessEqual(expected_prob, 1.0)
    
    def test_tolerance_settings(self):
        """Test different tolerance settings"""
        tolerances = [1.0, 2.0, 5.0, 10.0]
        
        for tol in tolerances:
            validator = AccuracyValidator(tolerance_pct=tol)
            self.assertEqual(validator.tolerance, tol / 100.0)
    
    def test_validate_model_accuracy_basic(self):
        """Test basic model accuracy validation"""
        # Load minimal weights to make model functional
        weights_dict = self._create_minimal_weights()
        self.model.load_weights_from_dict(weights_dict)
        
        results = self.validator.validate_model_accuracy(self.model)
        
        # Check result structure
        required_keys = [
            'total_tests', 'passed', 'failed', 'accuracy_deviations',
            'performance_metrics', 'validation_passed', 'detailed_results'
        ]
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check values
        self.assertIsInstance(results['total_tests'], int)
        self.assertIsInstance(results['passed'], int)
        self.assertIsInstance(results['failed'], int)
        self.assertIsInstance(results['validation_passed'], bool)
        self.assertEqual(results['total_tests'], results['passed'] + results['failed'])
        
        # Detailed results should match test count
        self.assertEqual(len(results['detailed_results']), results['total_tests'])
    
    def test_validate_model_accuracy_performance_metrics(self):
        """Test performance metrics collection"""
        weights_dict = self._create_minimal_weights()
        self.model.load_weights_from_dict(weights_dict)
        
        results = self.validator.validate_model_accuracy(self.model)
        
        if results['accuracy_deviations']:
            metrics = results['performance_metrics']
            
            # Check metric keys
            expected_metrics = [
                'mean_deviation_pct', 'max_deviation_pct', 
                'std_deviation_pct', 'mean_inference_ms'
            ]
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], float)
                self.assertGreaterEqual(metrics[metric], 0.0)
    
    def test_validate_model_accuracy_with_errors(self):
        """Test validation with model errors"""
        # Create a model that will throw errors
        broken_model = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
        # Don't load weights - should cause issues
        
        with patch.object(broken_model, 'predict', side_effect=Exception("Test error")):
            results = self.validator.validate_model_accuracy(broken_model)
            
            # Should handle errors gracefully
            self.assertIsInstance(results, dict)
            self.assertEqual(results['passed'], 0)  # All should fail
            self.assertEqual(results['failed'], results['total_tests'])
    
    def test_validate_conversion_accuracy(self):
        """Test conversion accuracy validation"""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Mock PyTorch loading since we don't have real model
            with patch('torch.load', return_value={'test': 'data'}):
                results = self.validator.validate_conversion_accuracy(
                    tmp_path, self.model
                )
                
                # Check result structure
                required_keys = [
                    'total_comparisons', 'within_tolerance', 'max_deviation',
                    'mean_deviation', 'comparisons', 'validation_passed'
                ]
                for key in required_keys:
                    self.assertIn(key, results)
                
                self.assertIsInstance(results['validation_passed'], bool)
        
        finally:
            os.unlink(tmp_path)
    
    def test_validate_conversion_accuracy_file_not_found(self):
        """Test conversion validation with missing file"""
        results = self.validator.validate_conversion_accuracy(
            "nonexistent_file.pth", self.model
        )
        
        self.assertFalse(results['validation_passed'])
        self.assertIn('error', results)
    
    def test_get_expected_range(self):
        """Test expected range calculation"""
        test_cases = [
            ("You're an idiot!", (0.6, 1.0)),  # High toxicity
            ("Thanks for help", (0.0, 0.2)),   # Low toxicity  
            ("Normal message", (0.0, 0.5)),    # Moderate
            ("I hate stupid things", (0.6, 1.0)),  # High toxicity
            ("Great job everyone!", (0.0, 0.2)),   # Low toxicity
        ]
        
        for text, expected_range in test_cases:
            result_range = self.validator._get_expected_range(text)
            self.assertEqual(result_range, expected_range)
    
    def test_print_validation_summary(self):
        """Test validation summary printing"""
        # Create mock results
        results = {
            'total_tests': 10,
            'passed': 7,
            'failed': 3,
            'validation_passed': True,
            'performance_metrics': {
                'mean_deviation_pct': 15.5,
                'max_deviation_pct': 45.2,
                'std_deviation_pct': 12.1,
                'mean_inference_ms': 25.3
            }
        }
        
        # Should not raise exception
        with patch('builtins.print'):
            self.validator._print_validation_summary(results)
    
    def _create_minimal_weights(self):
        """Helper to create minimal weight dict for testing"""
        return {
            "embedding": [[0.01 * i * j for j in range(14)] for i in range(256)],
            "conv1_weight": [[[0.1] * 14] * 3] * 28,
            "conv1_bias": [0.01] * 28,
            "conv2_weight": [[[0.1] * 28] * 3] * 24,
            "conv2_bias": [0.01] * 24,
            "classifier_weight": [[0.1] * 24] * 48,
            "classifier_bias": [0.01] * 48,
            "output_weight": [[0.1] * 48],
            "output_bias": [0.5],
            "bn1_weight": [1.0] * 28,
            "bn1_bias": [0.0] * 28,
            "bn1_running_mean": [0.0] * 28,
            "bn1_running_var": [1.0] * 28,
            "bn2_weight": [1.0] * 24,
            "bn2_bias": [0.0] * 24,
            "bn2_running_mean": [0.0] * 24,
            "bn2_running_var": [1.0] * 24,
        }


class TestValidateOptimizedModel(unittest.TestCase):
    """Test convenience validation function"""
    
    def test_validate_optimized_model_success(self):
        """Test successful model validation"""
        # Create temporary weights file
        weights_data = {
            "embedding": [[0.01] * 14] * 256,
            "conv1_weight": [[[0.1] * 14] * 3] * 28,
            "conv1_bias": [0.01] * 28,
            "conv2_weight": [[[0.1] * 28] * 3] * 24,
            "conv2_bias": [0.01] * 24,
            "classifier_weight": [[0.1] * 24] * 48,
            "classifier_bias": [0.01] * 48,
            "output_weight": [[0.1] * 48],
            "output_bias": [0.5],
            "bn1_weight": [1.0] * 28,
            "bn1_bias": [0.0] * 28,
            "bn1_running_mean": [0.0] * 28,
            "bn1_running_var": [1.0] * 28,
            "bn2_weight": [1.0] * 24,
            "bn2_bias": [0.0] * 24,
            "bn2_running_mean": [0.0] * 24,
            "bn2_running_var": [1.0] * 24,
            "model_info": {
                "architecture_type": "2layer_32kb",
                "layers": 2
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(weights_data, tmp)
            tmp_path = tmp.name
        
        try:
            # Mock validation to return True for testing
            with patch('tinybytecnn.accuracy_validation.AccuracyValidator') as mock_validator:
                mock_instance = MagicMock()
                mock_instance.validate_model_accuracy.return_value = {'validation_passed': True}
                mock_validator.return_value = mock_instance
                
                result = validate_optimized_model("", tmp_path)
                self.assertTrue(result)
        
        finally:
            os.unlink(tmp_path)
    
    def test_validate_optimized_model_file_not_found(self):
        """Test validation with missing weights file"""
        result = validate_optimized_model("", "nonexistent_weights.json")
        self.assertFalse(result)
    
    def test_validate_optimized_model_invalid_json(self):
        """Test validation with invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write("invalid json content")
            tmp_path = tmp.name
        
        try:
            result = validate_optimized_model("", tmp_path)
            self.assertFalse(result)
        
        finally:
            os.unlink(tmp_path)
    
    def test_validate_optimized_model_unknown_architecture(self):
        """Test validation with unknown architecture type"""
        weights_data = {
            "model_info": {
                "architecture_type": "unknown_type"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(weights_data, tmp)
            tmp_path = tmp.name
        
        try:
            result = validate_optimized_model("", tmp_path)
            self.assertFalse(result)
        
        finally:
            os.unlink(tmp_path)
    
    def test_validate_optimized_model_3layer(self):
        """Test validation with 3-layer architecture"""
        weights_data = {
            "model_info": {
                "architecture_type": "3layer_32kb",
                "layers": 3
            }
            # ... would need 3-layer weights in real test
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(weights_data, tmp)
            tmp_path = tmp.name
        
        try:
            # Should handle 3-layer architecture
            result = validate_optimized_model("", tmp_path)
            # Will fail due to missing weights, but shouldn't crash
            self.assertFalse(result)
        
        finally:
            os.unlink(tmp_path)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_validator_with_zero_tolerance(self):
        """Test validator with zero tolerance"""
        validator = AccuracyValidator(tolerance_pct=0.0)
        self.assertEqual(validator.tolerance, 0.0)
        
        # Should still work but be very strict
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
        results = validator.validate_model_accuracy(model)
        self.assertIsInstance(results, dict)
    
    def test_validator_with_high_tolerance(self):
        """Test validator with very high tolerance"""
        validator = AccuracyValidator(tolerance_pct=100.0)
        self.assertEqual(validator.tolerance, 1.0)
        
        # Should be very permissive
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
        results = validator.validate_model_accuracy(model)
        self.assertIsInstance(results, dict)
    
    def test_empty_test_cases(self):
        """Test validator with empty test cases"""
        validator = AccuracyValidator()
        validator.test_cases = []  # Empty test cases
        
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=128)
        results = validator.validate_model_accuracy(model)
        
        self.assertEqual(results['total_tests'], 0)
        self.assertEqual(results['passed'], 0)
        self.assertEqual(results['failed'], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)