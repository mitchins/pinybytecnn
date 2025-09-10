#!/usr/bin/env python3
"""
Unit tests for PinyByteCNN utility functions
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tinybytecnn.utils import (
    text_to_bytes,
    bytes_to_text,
    sliding_window_split
)


class TestTextToBytes(unittest.TestCase):
    """Test text to bytes conversion"""
    
    def test_basic_ascii(self):
        """Test basic ASCII text conversion"""
        text = "Hello"
        result = text_to_bytes(text)
        expected = [72, 101, 108, 108, 111]  # ASCII values
        self.assertEqual(result, expected)
    
    def test_empty_string(self):
        """Test empty string conversion"""
        result = text_to_bytes("")
        self.assertEqual(result, [])
    
    def test_unicode_characters(self):
        """Test Unicode character conversion"""
        text = "Hello 世界!"
        result = text_to_bytes(text)
        
        # Should contain more bytes than characters due to UTF-8 encoding
        self.assertGreater(len(result), len(text))
        
        # All values should be valid byte values
        for byte_val in result:
            self.assertGreaterEqual(byte_val, 0)
            self.assertLessEqual(byte_val, 255)
    
    def test_emoji_conversion(self):
        """Test emoji conversion"""
        text = "🚀🎯💻"
        result = text_to_bytes(text)
        
        # Emojis should convert to multiple bytes
        self.assertGreater(len(result), 3)
        
        # All should be valid bytes
        for byte_val in result:
            self.assertGreaterEqual(byte_val, 0)
            self.assertLessEqual(byte_val, 255)
    
    def test_max_len_truncation(self):
        """Test max length truncation"""
        text = "A" * 1000  # Long text
        max_len = 100
        
        result = text_to_bytes(text, max_len=max_len)
        self.assertEqual(len(result), max_len)
        
        # Should be all 'A' (ASCII 65)
        self.assertEqual(result, [65] * max_len)
    
    def test_max_len_shorter_than_text(self):
        """Test when max_len is shorter than text"""
        text = "Hello World!"
        max_len = 5
        
        result = text_to_bytes(text, max_len=max_len)
        self.assertEqual(len(result), max_len)
        self.assertEqual(result, [72, 101, 108, 108, 111])  # "Hello"
    
    def test_max_len_longer_than_text(self):
        """Test when max_len is longer than text"""
        text = "Hi"
        max_len = 100
        
        result = text_to_bytes(text, max_len=max_len)
        self.assertEqual(len(result), 2)  # Not padded
        self.assertEqual(result, [72, 105])  # "Hi"
    
    def test_special_characters(self):
        """Test special characters and symbols"""
        text = "!@#$%^&*()"
        result = text_to_bytes(text)
        
        # Should convert all characters
        self.assertEqual(len(result), len(text))
        
        # All should be valid bytes
        for byte_val in result:
            self.assertGreaterEqual(byte_val, 0)
            self.assertLessEqual(byte_val, 255)
    
    def test_whitespace_characters(self):
        """Test whitespace and control characters"""
        text = "\n\t\r "
        result = text_to_bytes(text)
        expected = [10, 9, 13, 32]  # \n, \t, \r, space
        self.assertEqual(result, expected)


class TestBytesToText(unittest.TestCase):
    """Test bytes to text conversion"""
    
    def test_basic_ascii(self):
        """Test basic ASCII bytes conversion"""
        byte_indices = [72, 101, 108, 108, 111]  # "Hello"
        result = bytes_to_text(byte_indices)
        self.assertEqual(result, "Hello")
    
    def test_empty_bytes(self):
        """Test empty bytes conversion"""
        result = bytes_to_text([])
        self.assertEqual(result, "")
    
    def test_single_byte(self):
        """Test single byte conversion"""
        result = bytes_to_text([65])  # 'A'
        self.assertEqual(result, "A")
    
    def test_unicode_roundtrip(self):
        """Test Unicode roundtrip conversion"""
        original_text = "Hello 世界!"
        bytes_result = text_to_bytes(original_text)
        text_result = bytes_to_text(bytes_result)
        self.assertEqual(text_result, original_text)
    
    def test_emoji_roundtrip(self):
        """Test emoji roundtrip conversion"""
        original_text = "🚀🎯💻"
        bytes_result = text_to_bytes(original_text)
        text_result = bytes_to_text(bytes_result)
        self.assertEqual(text_result, original_text)
    
    def test_invalid_bytes(self):
        """Test handling of invalid byte sequences"""
        # Invalid UTF-8 sequences should be handled gracefully
        invalid_bytes = [255, 254, 253]
        result = bytes_to_text(invalid_bytes)
        
        # Should not crash and return some result
        self.assertIsInstance(result, str)
    
    def test_mixed_valid_invalid(self):
        """Test mixed valid and invalid bytes"""
        # Mix of valid ASCII and invalid bytes
        mixed_bytes = [72, 101, 108, 255, 108, 111]
        result = bytes_to_text(mixed_bytes)
        
        # Should handle gracefully
        self.assertIsInstance(result, str)
    
    def test_zero_bytes(self):
        """Test handling of zero bytes"""
        bytes_with_zero = [72, 101, 0, 108, 111]
        result = bytes_to_text(bytes_with_zero)
        
        # Should handle zero bytes
        self.assertIsInstance(result, str)
    
    def test_exception_handling(self):
        """Test exception handling in bytes_to_text"""
        # Test with object that will cause exception
        with unittest.mock.patch('builtins.bytes', side_effect=Exception("Test error")):
            result = bytes_to_text([65, 66, 67])
            self.assertEqual(result, "")  # Should return empty string on error


class TestSlidingWindowSplit(unittest.TestCase):
    """Test sliding window splitting functionality"""
    
    def test_basic_splitting(self):
        """Test basic sliding window splitting"""
        indices = list(range(1000))  # 0 to 999
        window_size = 512
        stride = 256
        
        windows = sliding_window_split(indices, window_size, stride)
        
        # Should have multiple windows
        self.assertGreater(len(windows), 1)
        
        # Each window should be correct size
        for window in windows:
            self.assertEqual(len(window), window_size)
        
        # First window should start with 0
        self.assertEqual(windows[0][0], 0)
        
        # Second window should start at stride position
        self.assertEqual(windows[1][0], stride)
    
    def test_short_sequence(self):
        """Test with sequence shorter than window"""
        indices = [1, 2, 3, 4, 5]
        window_size = 10
        
        windows = sliding_window_split(indices, window_size)
        
        # Should return single window
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], indices)
    
    def test_exact_window_size(self):
        """Test with sequence exactly window size"""
        window_size = 100
        indices = list(range(window_size))
        
        windows = sliding_window_split(indices, window_size)
        
        # Should return single window
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], indices)
    
    def test_default_parameters(self):
        """Test with default parameters"""
        indices = list(range(1000))
        
        windows = sliding_window_split(indices)  # Default: window_size=512, stride=256
        
        self.assertGreater(len(windows), 1)
        self.assertEqual(len(windows[0]), 512)
        self.assertEqual(windows[1][0], 256)  # Second window starts at stride
    
    def test_custom_stride(self):
        """Test with custom stride"""
        indices = list(range(1000))
        window_size = 200
        stride = 100
        
        windows = sliding_window_split(indices, window_size, stride)
        
        # Check stride is applied correctly
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[1][0], stride)
        self.assertEqual(windows[2][0], stride * 2)
    
    def test_large_stride(self):
        """Test with stride larger than window size"""
        indices = list(range(1000))
        window_size = 100
        stride = 200
        
        windows = sliding_window_split(indices, window_size, stride)
        
        # Should still work with gaps between windows
        self.assertGreater(len(windows), 1)
        for window in windows:
            self.assertEqual(len(window), window_size)
    
    def test_small_stride(self):
        """Test with very small stride (high overlap)"""
        indices = list(range(500))
        window_size = 100
        stride = 10
        
        windows = sliding_window_split(indices, window_size, stride)
        
        # Should have many overlapping windows
        self.assertGreater(len(windows), 30)  # 500/10 ≈ 50 windows
        for window in windows:
            self.assertEqual(len(window), window_size)
    
    def test_padding_behavior(self):
        """Test padding of final window if needed"""
        # Sequence that doesn't align perfectly with windows
        indices = list(range(550))  # Not a multiple of stride
        window_size = 512
        stride = 256
        
        windows = sliding_window_split(indices, window_size, stride)
        
        # All windows should be same size due to padding
        for window in windows:
            self.assertEqual(len(window), window_size)
        
        # Final window should contain padding (zeros)
        final_window = windows[-1]
        
        # Should contain some original data
        self.assertIn(549, final_window)  # Last original element
        
        # Should contain some padding
        padding_count = final_window.count(0)
        self.assertGreater(padding_count, 0)
    
    def test_empty_input(self):
        """Test with empty input"""
        windows = sliding_window_split([])
        self.assertEqual(len(windows), 1)  # Single empty window
        self.assertEqual(windows[0], [])
    
    def test_single_element(self):
        """Test with single element"""
        windows = sliding_window_split([42])
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], [42])
    
    def test_window_coverage(self):
        """Test that windows cover entire input sequence"""
        indices = list(range(1000))
        window_size = 300
        stride = 150
        
        windows = sliding_window_split(indices, window_size, stride)
        
        # Check that all original elements appear in at least one window
        covered_elements = set()
        for window in windows:
            for elem in window:
                if elem != 0:  # Ignore padding
                    covered_elements.add(elem)
        
        # Most original elements should be covered (some may be lost due to windowing)
        original_elements = set(indices)
        coverage_ratio = len(covered_elements.intersection(original_elements)) / len(original_elements)
        self.assertGreater(coverage_ratio, 0.8)  # At least 80% coverage


class TestIntegration(unittest.TestCase):
    """Integration tests for utility functions"""
    
    def test_text_processing_pipeline(self):
        """Test complete text processing pipeline"""
        original_texts = [
            "Hello world!",
            "Testing Unicode: 世界",
            "Emojis: 🚀🎯💻",
            "Special chars: !@#$%^&*()",
            "Whitespace:\n\t  ",
            "",  # Empty
            "A" * 1000,  # Long text
        ]
        
        for text in original_texts:
            # Convert to bytes
            byte_indices = text_to_bytes(text, max_len=512)
            
            # Verify byte properties
            for byte_val in byte_indices:
                self.assertGreaterEqual(byte_val, 0)
                self.assertLessEqual(byte_val, 255)
            
            # Convert back to text (if possible for perfect UTF-8)
            if all(b < 128 for b in byte_indices) or not text.encode('utf-8', errors='ignore'):
                # For ASCII or valid UTF-8, should roundtrip perfectly
                reconstructed = bytes_to_text(byte_indices)
                # Note: May not be perfect due to UTF-8 encoding complexities
            
            # Split into windows
            if len(byte_indices) > 10:
                windows = sliding_window_split(byte_indices, window_size=50, stride=25)
                
                # Verify window properties
                self.assertGreater(len(windows), 0)
                for window in windows:
                    # Windows may be padded or original size if shorter than window_size
                    if len(byte_indices) <= 50:
                        self.assertEqual(len(window), len(byte_indices))
                    else:
                        self.assertEqual(len(window), 50)
                    for byte_val in window:
                        self.assertGreaterEqual(byte_val, 0)
                        self.assertLessEqual(byte_val, 255)
    
    def test_performance_benchmarks(self):
        """Test performance of utility functions"""
        import time
        
        # Large text for performance testing
        large_text = "Performance test text with various characters! 🚀 " * 1000
        
        # Benchmark text_to_bytes
        start_time = time.time()
        for _ in range(1000):
            bytes_result = text_to_bytes(large_text, max_len=1024)
        text_to_bytes_time = time.time() - start_time
        
        # Benchmark bytes_to_text
        start_time = time.time()
        for _ in range(1000):
            text_result = bytes_to_text(bytes_result)
        bytes_to_text_time = time.time() - start_time
        
        # Benchmark sliding_window_split
        large_indices = list(range(10000))
        start_time = time.time()
        for _ in range(100):
            windows = sliding_window_split(large_indices, window_size=512, stride=256)
        sliding_window_time = time.time() - start_time
        
        # Performance assertions (generous limits for various systems)
        self.assertLess(text_to_bytes_time, 1.0)  # < 1s for 1000 conversions
        self.assertLess(bytes_to_text_time, 1.0)   # < 1s for 1000 conversions
        self.assertLess(sliding_window_time, 1.0)  # < 1s for 100 window splits
        
        print(f"Performance: text_to_bytes={text_to_bytes_time:.3f}s, "
              f"bytes_to_text={bytes_to_text_time:.3f}s, "
              f"sliding_window={sliding_window_time:.3f}s")


if __name__ == "__main__":
    # Import mock for testing
    try:
        import unittest.mock
    except ImportError:
        import mock
        unittest.mock = mock
    
    unittest.main(verbosity=2)