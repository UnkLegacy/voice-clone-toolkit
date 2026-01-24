# Voice Clone Toolkit - Test Suite

This directory contains comprehensive unit tests for the Voice Clone Toolkit, including both **main unit tests** and **quality tests** designed to catch common development issues.

## 📁 Test Structure

```
tests/
├── README.md                     # This file
├── test_*.py                     # Main unit tests for individual modules
├── test_import_integrity.py      # Quality tests for import issues
├── test_error_formatting.py      # Quality tests for error message formatting  
├── test_code_quality.py          # Quality tests for code structure
└── __init__.py                   # Test package initialization
```

## 🧪 Running Tests

### Basic Usage
```bash
# Run main unit tests only (default)
python run_tests.py

# Run all tests (main + quality)
python run_tests.py --all

# Run only quality tests  
python run_tests.py --quality
```

### Individual Test Categories
```bash
# Run specific test modules
python -m unittest tests.test_clone_voice -v
python -m unittest tests.test_import_integrity -v
python -m unittest tests.test_error_formatting -v
```

## 📊 Test Categories

### 🎯 Main Unit Tests

Standard unit tests that verify the functionality of individual modules:

| Test Module | Purpose |
|-------------|---------|
| `test_clone_voice.py` | Voice cloning script functionality |
| `test_clone_voice_conversation.py` | Conversation generation |
| `test_custom_voice.py` | Custom voice generation |
| `test_voice_design.py` | Voice design functionality |
| `test_voice_design_clone.py` | Combined voice design + cloning |
| `test_convert_audio_format.py` | Audio format conversion |
| `test_utils_*.py` | Individual utility module tests |

### 🔍 Quality Tests

Specialized tests designed to catch common development issues:

#### **Import Integrity Tests** (`test_import_integrity.py`)

**Purpose**: Catch import-related issues that can break scripts

**Issues Detected**:
- ✅ **Missing imports** (like `torch`, `wave`, `playsound`)
- ✅ **Duplicate imports** (same module imported multiple times)
- ✅ **Undefined name usage** (using modules without importing)
- ✅ **Direct execution failures** (scripts that can't run with `--help`)

**Example Issues Caught**:
```python
# Missing import
torch.device("cuda")  # Error: torch not imported

# Duplicate import  
import sys
# ...later...
import sys  # Caught as duplicate

# Undefined usage
AudioSegment.from_wav()  # Error: pydub not imported
```

#### **Error Formatting Tests** (`test_error_formatting.py`)

**Purpose**: Ensure consistent error message formatting

**Issues Detected**:
- ✅ **Inappropriate error formatting** (using `print_progress("Error: ...")` instead of `print_error()`)
- ✅ **Missing error function imports** (using error messages without importing `print_error`)
- ✅ **Inconsistent error handling patterns**

**Example Issues Caught**:
```python
# Wrong formatting
print_progress("Error: File not found")  # Should be print_error()

# Missing import
print_error("Something failed")  # Error: print_error not imported
```

#### **Code Quality Tests** (`test_code_quality.py`)

**Purpose**: Detect structural and duplication issues

**Issues Detected**:
- ✅ **Duplicate function definitions** across scripts
- ✅ **Large code duplication** (5+ identical lines between files)
- ✅ **Function redefinition** (defining utility functions locally)
- ✅ **Improper utility usage** (not using shared utility functions)

**Example Issues Caught**:
```python
# Duplicate function across files
def save_audio(...):  # Should use utils.audio_utils.save_audio()

# Code duplication
for run_num in range(1, batch_runs + 1):  # Same loop in multiple files
```

## 🎯 How These Tests Would Have Caught Your Issues

### Issue: Missing Imports (`torch`, `wave`, `playsound`)

**Test That Would Catch**: `test_import_integrity.py`

**Specific Test Methods**:
- `test_scripts_help_command()` - Tests direct execution with `--help`
- `test_undefined_names_in_main_scripts()` - Detects usage without imports
- `test_main_scripts_importable()` - Tests module importability

```python
def test_undefined_names_in_main_scripts(self):
    """Test for common undefined name patterns that indicate missing imports."""
    undefined_patterns = {
        'torch': ['torch.device', 'torch.cuda', 'torch.bfloat16'],
        'wave': ['wave.open'],
        'playsound': ['playsound('],
    }
    # ... checks for usage without imports
```

### Issue: Inappropriate Error Formatting

**Test That Would Catch**: `test_error_formatting.py`

**Specific Test Method**:
- `test_no_error_with_print_progress()` - Detects `print_progress("Error: ...")`

```python
def test_no_error_with_print_progress(self):
    """Test that error messages don't use print_progress inappropriately."""
    error_patterns = [
        r'print_progress\([^)]*["\'].*[Ee]rror:.*["\']',  # print_progress("Error: ...")
    ]
    # ... scans code for inappropriate patterns
```

### Issue: Duplicate Imports

**Test That Would Catch**: `test_import_integrity.py`

**Specific Test Method**:
- `test_no_duplicate_imports()` - Uses AST parsing to detect duplicate imports

```python
def test_no_duplicate_imports(self):
    """Test that modules don't have duplicate imports."""
    # Parse AST and count import statements
    duplicates = {imp: count for imp, count in import_counts.items() if count > 1}
```

### Issue: Redundant Functions

**Test That Would Catch**: `test_code_quality.py`

**Specific Test Methods**:
- `test_no_duplicate_function_definitions()` - Finds functions defined in multiple scripts
- `test_utility_functions_not_redefined()` - Detects redefinition of utility functions

```python
def test_utility_functions_not_redefined(self):
    """Test that main scripts don't redefine utility functions."""
    # Checks if scripts define functions that exist in utils/
```

## 🚀 Integration with Development Workflow

### Pre-commit Checks
```bash
# Add to your development workflow
python run_tests.py --all  # Run before committing
```

### CI/CD Integration
```yaml
# Add to GitHub Actions or similar
- name: Run Tests  
  run: python run_tests.py --all
```

### Development Best Practices

1. **Run quality tests after refactoring** - They catch issues introduced during code restructuring
2. **Use `--all` flag during development** - Catches issues early before they become problems
3. **Fix quality test failures** - They indicate real code quality issues
4. **Add new quality tests** - When you encounter new categories of issues

## 📈 Test Coverage

**Main Tests**: Cover functionality and behavior of all modules
**Quality Tests**: Cover structural integrity and development practices

Combined, they provide:
- ✅ **Functional coverage** - Does the code work?
- ✅ **Structural coverage** - Is the code well-organized?  
- ✅ **Integration coverage** - Do the parts work together?
- ✅ **Quality coverage** - Does the code follow good practices?

## 🎯 When to Run Each Type

| Scenario | Command | Purpose |
|----------|---------|---------|
| **Daily development** | `python run_tests.py` | Quick functionality check |
| **Before committing** | `python run_tests.py --all` | Complete validation |
| **After refactoring** | `python run_tests.py --quality` | Catch structural issues |
| **CI/CD pipeline** | `python run_tests.py --all` | Full validation |

## 📝 Adding New Tests

### For New Functionality
Add tests to existing `test_*.py` files or create new ones following the naming pattern.

### For New Quality Checks
Add test methods to the quality test files:
- `test_import_integrity.py` - Import-related checks
- `test_error_formatting.py` - Error message checks
- `test_code_quality.py` - Structural checks

### Example Quality Test
```python
def test_new_quality_check(self):
    """Test description."""
    for file_path in self.all_files:
        with self.subTest(file=file_path):
            # Your quality check logic
            if issue_found:
                self.fail(f"Quality issue in {file_path}: {issue_description}")
```

The quality tests complement the main unit tests by catching the types of issues that are easy to introduce during development but hard to notice until they cause runtime errors.