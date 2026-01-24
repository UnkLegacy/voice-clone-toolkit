#!/usr/bin/env python3
"""
Run quality and integrity tests for Voice Clone Toolkit.

This script runs additional tests to catch code quality issues like:
- Import integrity problems (missing imports, duplicates)
- Error formatting issues (using print_progress instead of print_error)
- Code duplication and structural problems

These tests complement the main unit tests and help catch the types
of issues that can be introduced during refactoring.
"""

import sys
import unittest
from pathlib import Path


def main():
    """Run all quality tests."""
    print("=" * 60)
    print("VOICE CLONE TOOLKIT - QUALITY TESTS")
    print("=" * 60)
    print("Running tests to catch import, formatting, and structure issues...")
    print()
    
    # Create test suite with quality tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add our quality test modules
    quality_test_modules = [
        'tests.test_import_integrity',
        'tests.test_error_formatting',
        'tests.test_code_quality'
    ]
    
    for module_name in quality_test_modules:
        try:
            module_suite = loader.loadTestsFromName(module_name)
            suite.addTest(module_suite)
            print(f"✓ Loaded {module_name}")
        except Exception as e:
            print(f"✗ Failed to load {module_name}: {e}")
            return 1
    
    print()
    print("Running quality tests...")
    print("-" * 60)
    
    # Run the tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,  # Capture stdout/stderr during tests
        failfast=False  # Run all tests even if some fail
    )
    
    result = runner.run(suite)
    
    print("-" * 60)
    print("\nQUALITY TEST SUMMARY:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, error in result.failures:
            print(f"  {test}: {error.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"  {test}: {error.split('\\n')[-2]}")
    
    if result.wasSuccessful():
        print("\n🎉 All quality tests passed!")
        return 0
    else:
        print(f"\n⚠️  Quality issues found: {len(result.failures + result.errors)} tests failed")
        print("\nThese tests help catch issues like:")
        print("  - Missing or duplicate imports")
        print("  - Inappropriate error message formatting")
        print("  - Code duplication that should be refactored")
        print("  - Functions redefined instead of using utilities")
        print("\nConsider fixing these issues to improve code quality.")
        return 1


if __name__ == '__main__':
    sys.exit(main())