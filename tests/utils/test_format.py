import unittest

from meme_nexus.utils.format import format_number


class TestFormatNumber(unittest.TestCase):
    """Test cases for format_number function"""

    def test_basic_formatting(self):
        """Test basic number formatting functionality"""
        # Test numbers of various magnitudes
        self.assertEqual(format_number(123), "123")
        self.assertEqual(format_number(1234), "1.23K")
        self.assertEqual(format_number(1234567), "1.23M")
        self.assertEqual(format_number(1234567890), "1.23B")
        self.assertEqual(format_number(1234567890000), "1.23T")

        # Test negative numbers
        self.assertEqual(format_number(-1234), "-1.23K")
        self.assertEqual(format_number(-1234567), "-1.23M")

    def test_precision_parameter(self):
        """Test the effect of the precision parameter"""
        self.assertEqual(format_number(1234, precision=0), "1K")
        self.assertEqual(format_number(1234, precision=1), "1.2K")
        self.assertEqual(format_number(1234, precision=3), "1.234K")
        self.assertEqual(format_number(1000, precision=2), "1K")  # Test exact value
        self.assertEqual(format_number(1000.5, precision=2), "1K")  # Test rounding

    def test_is_format_k_parameter(self):
        """Test the effect of the is_format_k parameter"""
        # Test cases with is_format_k=False
        self.assertEqual(format_number(123, is_format_k=False), "123")
        self.assertEqual(format_number(1234, is_format_k=False), "1234")
        self.assertEqual(format_number(12345, is_format_k=False), "12345")
        self.assertEqual(
            format_number(12345.1, precision=4, is_format_k=False), "12345.1"
        )

        # Compare is_format_k=True and False cases
        self.assertEqual(format_number(1234, is_format_k=True), "1.23K")
        self.assertEqual(format_number(1234, is_format_k=False), "1234")

    def test_edge_cases(self):
        """Test edge cases"""
        # Test zero value
        self.assertEqual(format_number(0), "0")
        self.assertEqual(format_number(0, precision=2), "0")

        # Test very small values
        self.assertEqual(format_number(0.123, precision=3), "0.123")
        self.assertEqual(format_number(0.123, precision=4), "0.123")
        self.assertEqual(format_number(0.123, precision=5), "0.123")

        # Test values near unit boundaries
        self.assertEqual(format_number(999), "999")
        self.assertEqual(format_number(999.9, precision=1), "999.9")
        self.assertEqual(format_number(1000), "1K")

        # Test rounding with precision=0
        self.assertEqual(format_number(1499, precision=0), "1K")
        self.assertEqual(format_number(1500, precision=0), "2K")

    def test_error_handling(self):
        """Test error handling"""
        # Test negative precision
        with self.assertRaises(ValueError):
            format_number(1234, precision=-1)


if __name__ == "__main__":
    unittest.main()
