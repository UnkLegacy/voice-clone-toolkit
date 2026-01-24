# Testing Guide

This document provides information about testing the Voice Clone Toolkit.

## Quick Start

```bash
# Run main unit tests only (default)
python run_tests.py

# Run all tests (main + quality)
python run_tests.py --all

# Run only quality tests
python run_tests.py --quality

# Or use unittest directly
python -m unittest discover tests -v
```

## Test Coverage

The project includes comprehensive unit tests for:

### clone_voice.py
- ✅ Voice profile loading from JSON
- ✅ Text loading from files or inline
- ✅ Directory creation and management
- ✅ WAV file saving with different formats
- ✅ Command-line argument parsing
- ✅ Voice profile listing

### clone_voice_conversation.py
- ✅ JSON configuration loading
- ✅ Script format parsing (`[VoiceName] dialogue`)
- ✅ Script list parsing
- ✅ Audio file operations
- ✅ Command-line argument parsing
- ✅ Script listing functionality

### custom_voice.py
- ✅ Custom voice profile loading from JSON
- ✅ Speaker profile listing
- ✅ Directory creation and management
- ✅ WAV file saving with different formats
- ✅ Command-line argument parsing

### voice_design.py
- ✅ Voice design profile loading from JSON
- ✅ Design profile listing
- ✅ Directory creation and management
- ✅ WAV file saving with different formats
- ✅ Command-line argument parsing

### voice_design_clone.py
- ✅ Voice design + clone profile loading from JSON
- ✅ Design clone profile listing
- ✅ Directory creation and management
- ✅ WAV file saving with different formats
- ✅ Command-line argument parsing

### convert_audio_format.py
- ✅ Audio format conversion (WAV to MP3)
- ✅ Single file and directory conversion
- ✅ Command-line argument parsing
- ✅ Error handling for missing files

### utils/ modules
- ✅ Progress reporting and error handling (`progress.py`)
- ✅ Audio processing and format conversion (`audio_utils.py`)
- ✅ Command-line argument parsing (`cli_args.py`)
- ✅ JSON configuration management (`config_loader.py`)
- ✅ Model loading and device management (`model_utils.py`)
- ✅ File operations and text handling (`file_utils.py`)

## Quality Tests

In addition to main unit tests, the project includes **quality tests** that catch common development issues:

### Import Integrity Tests (`test_import_integrity.py`)
- ✅ Missing imports (torch, wave, pydub, etc.)
- ✅ Duplicate imports across modules
- ✅ Undefined name usage
- ✅ Direct script execution failures

### Error Formatting Tests (`test_error_formatting.py`)
- ✅ Inappropriate use of print_progress for errors
- ✅ Missing print_error imports
- ✅ Consistent error message patterns

### Code Quality Tests (`test_code_quality.py`)
- ✅ Duplicate function definitions
- ✅ Large code duplication between files
- ✅ Function redefinition issues
- ✅ Proper utility function usage

## Running Specific Tests

### Main Unit Tests
```bash
# Run tests for a specific module
python -m unittest tests.test_clone_voice -v
python -m unittest tests.test_custom_voice -v
python -m unittest tests.test_voice_design -v

# Run utility tests
python -m unittest tests.test_utils_progress -v
python -m unittest tests.test_utils_audio_utils -v

# Run a specific test class
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles -v

# Run a specific test method
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles.test_load_valid_profiles -v
```

### Quality Tests
```bash
# Run all quality tests
python run_tests.py --quality

# Run specific quality test modules
python -m unittest tests.test_import_integrity -v
python -m unittest tests.test_error_formatting -v
python -m unittest tests.test_code_quality -v
```

## Test Organization

```
tests/
├── __init__.py                           # Package initialization
├── README.md                             # Testing documentation
│
├── Main Unit Tests:
├── test_clone_voice.py                   # Tests for src/clone_voice.py
├── test_clone_voice_conversation.py      # Tests for src/clone_voice_conversation.py
├── test_custom_voice.py                  # Tests for src/custom_voice.py
├── test_voice_design.py                  # Tests for src/voice_design.py
├── test_voice_design_clone.py            # Tests for src/voice_design_clone.py
├── test_convert_audio_format.py          # Tests for src/convert_audio_format.py
├── test_utils_progress.py                # Tests for src/utils/progress.py
├── test_utils_audio_utils.py             # Tests for src/utils/audio_utils.py
├── test_utils_cli_args.py                # Tests for src/utils/cli_args.py
├── test_utils_config_loader.py           # Tests for src/utils/config_loader.py
├── test_utils_model_utils.py             # Tests for src/utils/model_utils.py
├── test_utils_file_utils.py              # Tests for src/utils/file_utils.py
│
└── Quality Tests:
    ├── test_import_integrity.py          # Import-related quality checks
    ├── test_error_formatting.py          # Error message formatting checks
    └── test_code_quality.py              # Code structure and duplication checks
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

This project includes automated testing via GitHub Actions!

### GitHub Actions Workflow

The repository is configured with a GitHub Actions workflow (`.github/workflows/tests.yml`) that automatically runs all unit tests on every push.

**Features:**
- ✅ Runs on every push to any branch
- ✅ Tests against multiple Python versions (3.10, 3.11, 3.12)
- ✅ Automatically installs all dependencies
- ✅ Provides test results in the Actions tab

**View test results:**
- Check the **Actions** tab in the GitHub repository
- Look for the [![Tests](https://github.com/UnkLegacy/voice-clone-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/UnkLegacy/voice-clone-toolkit/actions/workflows/tests.yml) badge in the README

The workflow configuration is available at `.github/workflows/tests.yml`:

```yaml
name: Run Unit Tests

on:
  push:
    branches: [ "**" ]
  pull_request:
    branches: [ "main", "master" ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run unit tests
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
