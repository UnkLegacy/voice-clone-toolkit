"""
Test runner for Voice Clone Toolkit

Discovers and runs all unit tests in the tests/ directory.
"""

import sys
import unittest

if __name__ == '__main__':
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
