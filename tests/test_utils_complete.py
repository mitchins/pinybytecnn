#!/usr/bin/env python3
"""
Complete utils tests to push coverage over 80%
"""

import unittest
from tinybytecnn.utils import relu_vec


class TestUtilsComplete(unittest.TestCase):
    """Complete utils testing"""

    def test_relu_vec_complete(self):
        """Test relu_vec with comprehensive inputs"""
        # Test the missing relu_vec line
        result = relu_vec([-1.0, 0.0, 1.0, -5.0, 10.0])
        expected = [0.0, 0.0, 1.0, 0.0, 10.0]
        self.assertEqual(result, expected)

    def test_relu_vec_edge_cases(self):
        """Test relu_vec edge cases"""
        # Empty list
        self.assertEqual(relu_vec([]), [])
        
        # Single values
        self.assertEqual(relu_vec([5.0]), [5.0])
        self.assertEqual(relu_vec([-3.0]), [0.0])


if __name__ == '__main__':
    unittest.main()