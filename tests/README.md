# Tests Directory

Unit tests for all Qwen3-TTS scripts.

## Running Tests

### Run All Tests

```bash
# From project root
python -m unittest discover tests

# Or with verbose output
python -m unittest discover tests -v
```

### Run Specific Test File

```bash
# Test Clone_Voice.py
python -m unittest tests.test_clone_voice

# Test Clone_Voice_Conversation.py
python -m unittest tests.test_clone_voice_conversation
```

### Run Specific Test Class

```bash
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles
```

### Run Specific Test Method

```bash
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles.test_load_valid_profiles
```

## Test Coverage

To generate test coverage reports:

```bash
# Install coverage tool
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests

# View coverage report
coverage report

# Generate HTML coverage report
coverage html
# Open htmlcov/index.html in browser
```

## Test Structure

- `test_clone_voice.py` - Tests for Clone_Voice.py
  - Voice profile loading
  - Text file loading
  - Directory creation
  - WAV file saving
  - Command-line argument parsing

- `test_clone_voice_conversation.py` - Tests for Clone_Voice_Conversation.py
  - JSON configuration loading
  - Script format parsing
  - Script list parsing
  - Audio file operations
  - Command-line argument parsing

## Adding New Tests

When adding new functionality:

1. Create test file: `tests/test_your_module.py`
2. Import the module to test
3. Create test classes inheriting from `unittest.TestCase`
4. Write test methods (must start with `test_`)
5. Use assertions to verify behavior

### Example Test

```python
import unittest
from your_module import your_function

class TestYourFunction(unittest.TestCase):
    def test_basic_functionality(self):
        result = your_function("input")
        self.assertEqual(result, "expected_output")
```

## Test Fixtures

Tests use `setUp()` and `tearDown()` methods to:
- Create temporary directories/files
- Set up test data
- Clean up after tests

## Mocking

Tests use `unittest.mock` to:
- Mock file system operations
- Mock external dependencies
- Test without requiring actual models

## Continuous Integration

These tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    python -m unittest discover tests -v
```

## Requirements

Tests require only Python standard library modules:
- `unittest` - Test framework
- `tempfile` - Temporary file/directory creation
- `shutil` - File operations
- `unittest.mock` - Mocking support

No additional dependencies needed!
