# Testing Guide

This document provides information about testing the Qwen3-TTS Scripts.

## Quick Start

```bash
# Run all tests
python run_tests.py

# Or use unittest directly
python -m unittest discover tests -v
```

## Test Coverage

The project includes comprehensive unit tests for:

### Clone_Voice.py
- ✅ Voice profile loading from JSON
- ✅ Text loading from files or inline
- ✅ Directory creation and management
- ✅ WAV file saving with different formats
- ✅ Command-line argument parsing
- ✅ Voice profile listing

### Clone_Voice_Conversation.py
- ✅ JSON configuration loading
- ✅ Script format parsing (`[Actor] dialogue`)
- ✅ Script list parsing
- ✅ Audio file operations
- ✅ Command-line argument parsing
- ✅ Script listing functionality

## Running Specific Tests

```bash
# Run tests for a specific module
python -m unittest tests.test_clone_voice

# Run a specific test class
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles

# Run a specific test method
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles.test_load_valid_profiles
```

## Test Organization

```
tests/
├── __init__.py                           # Package initialization
├── README.md                             # Testing documentation
├── test_clone_voice.py                   # Tests for Clone_Voice.py
└── test_clone_voice_conversation.py      # Tests for Clone_Voice_Conversation.py
```

## Writing New Tests

### 1. Create Test File

```python
# tests/test_new_module.py
import unittest
from new_module import function_to_test

class TestNewModule(unittest.TestCase):
    def test_something(self):
        result = function_to_test("input")
        self.assertEqual(result, "expected")
```

### 2. Use Test Fixtures

```python
class TestWithFixtures(unittest.TestCase):
    def setUp(self):
        """Runs before each test method"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Runs after each test method"""
        shutil.rmtree(self.temp_dir)
    
    def test_with_temp_dir(self):
        # Use self.temp_dir in test
        pass
```

### 3. Mock External Dependencies

```python
from unittest.mock import Mock, patch

class TestWithMocks(unittest.TestCase):
    @patch('module.external_function')
    def test_mocked_function(self, mock_func):
        mock_func.return_value = "mocked"
        result = my_function()
        self.assertEqual(result, "mocked")
```

## Test Assertions

Common assertions used in tests:

```python
# Equality
self.assertEqual(a, b)
self.assertNotEqual(a, b)

# Boolean
self.assertTrue(x)
self.assertFalse(x)

# Existence
self.assertIn(item, collection)
self.assertNotIn(item, collection)

# Exceptions
with self.assertRaises(Exception):
    function_that_raises()

# Files/Paths
self.assertTrue(os.path.exists(path))
```

## Coverage Reports

Install coverage tool:

```bash
pip install coverage
```

Generate coverage report:

```bash
# Run tests with coverage
coverage run -m unittest discover tests

# View terminal report
coverage report

# View detailed line-by-line HTML report
coverage html
# Open htmlcov/index.html
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Run tests
        run: python run_tests.py
```

## Test Philosophy

- **Unit Tests**: Test individual functions in isolation
- **No External Dependencies**: Tests don't require models or GPU
- **Fast**: All tests should run in seconds
- **Deterministic**: Tests should always produce same results
- **Isolated**: Each test is independent

## Debugging Failed Tests

```bash
# Run with maximum verbosity
python -m unittest tests.test_module.TestClass.test_method -v

# Use Python debugger
python -m pdb run_tests.py
```

## Best Practices

1. **One assertion per test** (when possible)
2. **Descriptive test names**: `test_load_nonexistent_file_returns_empty_dict`
3. **Test edge cases**: Empty inputs, invalid data, boundary conditions
4. **Clean up**: Use `tearDown()` to remove temporary files
5. **Don't test external libraries**: Trust that dependencies work
6. **Mock expensive operations**: File I/O, network calls, model loading

## Common Issues

### Import Errors

Make sure to run tests from project root:

```bash
# Good
python -m unittest discover tests

# May cause import issues
cd tests
python test_clone_voice.py
```

### Temp File Cleanup

Always clean up in `tearDown()`:

```python
def tearDown(self):
    if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
        shutil.rmtree(self.temp_dir)
```

### Path Issues

Use `os.path.join()` for cross-platform paths:

```python
# Good
path = os.path.join(self.temp_dir, "file.txt")

# Bad (Windows/Linux incompatible)
path = self.temp_dir + "/file.txt"
```

## Future Test Additions

Tests to consider adding:

- Integration tests with actual models (optional, separate suite)
- Performance benchmarks
- Audio quality validation tests
- Configuration validation tests
- Script syntax validation tests

## Support

For questions about testing:
1. Check this guide
2. Read tests/README.md
3. Review existing test files as examples
4. Check Python unittest documentation

---

**Remember**: Good tests make development faster and safer! 🧪✨
