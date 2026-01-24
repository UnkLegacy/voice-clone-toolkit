#!/usr/bin/env python3
"""
Voice Clone Toolkit Test Runner

This script runs all unit tests for the Voice Clone Toolkit project.
It provides colored output, detailed error reporting, and test statistics.

Usage:
    python run_tests.py          # Run main unit tests
    python run_tests.py --all    # Run all tests including quality tests
    python run_tests.py --quality # Run only quality tests
"""

import sys
import unittest
import argparse


def run_main_tests():
    """Run the main unit tests."""
    print("=" * 60)
    print("VOICE CLONE TOOLKIT - MAIN UNIT TESTS")
    print("=" * 60)
    
    # Discover and run tests (excluding quality tests)
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Remove quality tests from the suite
    quality_test_patterns = ['test_import_integrity', 'test_error_formatting', 'test_code_quality']
    filtered_suite = unittest.TestSuite()
    
    for test_group in suite:
        for test_class in test_group:
            test_name = str(test_class._tests[0].__class__.__module__ if test_class._tests else '')
            if not any(pattern in test_name for pattern in quality_test_patterns):
                filtered_suite.addTest(test_class)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(filtered_suite)
    
    return result.wasSuccessful()


def run_quality_tests():
    """Run the quality tests."""
    print("=" * 60)
    print("VOICE CLONE TOOLKIT - QUALITY TESTS")  
    print("=" * 60)
    print("Running tests to catch import, formatting, and structure issues...")
    print()
    
    # Load only quality tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    quality_test_modules = [
        'tests.test_import_integrity',
        'tests.test_error_formatting', 
        'tests.test_code_quality'
    ]
    
    for module_name in quality_test_modules:
        try:
            module_suite = loader.loadTestsFromName(module_name)
            suite.addTest(module_suite)
        except Exception as e:
            print(f"Warning: Could not load {module_name}: {e}")
    
    if suite.countTestCases() == 0:
        print("No quality tests found.")
        return True
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Run tests based on command line arguments."""
    parser = argparse.ArgumentParser(
        description='Voice Clone Toolkit Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py          # Run main unit tests only
  python run_tests.py --all    # Run all tests (main + quality)  
  python run_tests.py --quality # Run only quality tests
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all tests including quality tests'
    )
    
    parser.add_argument(
        '--quality',
        action='store_true', 
        help='Run only quality tests (import integrity, error formatting, code structure)'
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.quality:
        # Run only quality tests
        success = run_quality_tests()
    elif args.all:
        # Run both main and quality tests
        print("Running all tests (main + quality)...\n")
        
        main_success = run_main_tests()
        print("\n" + "="*60 + "\n")
        quality_success = run_quality_tests()
        
        success = main_success and quality_success
        
        print("\n" + "="*60)
        print("OVERALL RESULTS:")
        print(f"Main tests: {'PASSED' if main_success else 'FAILED'}")
        print(f"Quality tests: {'PASSED' if quality_success else 'FAILED'}")
        print("="*60)
        
    else:
        # Run only main tests (default)
        success = run_main_tests()
        print("\nTip: Run 'python run_tests.py --all' to also check code quality.")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
