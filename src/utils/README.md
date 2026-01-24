# Voice Clone Toolkit - Utility Modules

This directory contains shared utility modules that provide common functionality used across all Voice Clone Toolkit scripts. The modular architecture promotes code reuse, reduces duplication, and improves maintainability.

## 📁 Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| [`progress.py`](#progresspy) | Progress reporting & error handling | `print_progress()`, `print_error()`, `handle_*_error()` |
| [`audio_utils.py`](#audio_utilspy) | Audio processing & format conversion | `save_audio()`, `save_wav()`, `play_audio()` |
| [`cli_args.py`](#cli_argspy) | Command-line argument parsing | `create_standard_parser()`, `add_common_args()` |
| [`config_loader.py`](#config_loaderpy) | JSON configuration management | `load_voice_clone_profiles()`, `load_json_config()` |
| [`model_utils.py`](#model_utilspy) | ML model loading & device management | `load_voice_clone_model()`, `get_device()` |
| [`file_utils.py`](#file_utilspy) | File operations & text handling | `load_text_from_file_or_string()`, `ensure_directory_exists()` |

## 📚 Module Documentation

### `progress.py`

**Purpose**: Standardized progress reporting, error handling, and user feedback across all scripts.

**Key Functions**:
- `print_progress(message)` - Display formatted progress messages with timestamps
- `print_error(message, show_traceback=False)` - Display formatted error messages to stderr
- `print_warning(message)` - Display formatted warning messages
- `handle_fatal_error(message)` - Handle fatal errors and exit with code 1
- `handle_processing_error(message)` - Handle processing errors and return False
- `handle_mp3_conversion_error(wav_path, error)` - Handle MP3 conversion failures gracefully
- `handle_audio_playback_error(error)` - Handle audio playback errors
- `with_error_handling(exit_on_error=True)` - Decorator for automatic error handling

**Usage Example**:
```python
from .utils.progress import print_progress, handle_fatal_error

print_progress("Loading voice profiles...")
try:
    # Some operation
    pass
except Exception as e:
    handle_fatal_error(f"Failed to load profiles: {e}")
```

---

### `audio_utils.py`

**Purpose**: Audio file operations, format conversion (WAV/MP3), and playback functionality with optional dependency handling.

**Key Functions**:
- `ensure_output_dir(output_path)` - Create output directories if they don't exist
- `save_wav(audio_data, sample_rate, output_path)` - Save audio as WAV file
- `save_audio(audio_data, sample_rate, output_path, format="wav", bitrate="192k")` - Save audio in specified format
- `play_audio(file_path, no_play=False)` - Play audio file with fallback options
- `get_audio_info(file_path)` - Get audio file metadata (duration, size, etc.)

**Supported Formats**:
- **WAV**: Lossless format using built-in `wave` module
- **MP3**: Compressed format using `pydub` + `ffmpeg` (optional dependencies)

**Usage Example**:
```python
from .utils.audio_utils import save_audio, play_audio

# Save as WAV
save_audio(audio_data, sample_rate, "output/voice.wav")

# Save as MP3 with custom bitrate
save_audio(audio_data, sample_rate, "output/voice.mp3", format="mp3", bitrate="320k")

# Play the audio
play_audio("output/voice.wav")
```

---

### `cli_args.py`

**Purpose**: Reusable command-line argument parsers that provide consistent CLI interfaces across all scripts.

**Key Functions**:
- `create_base_parser(description, profiles=None, examples=None)` - Create parser with description and help
- `add_audio_format_args(parser)` - Add `--output-format` and `--bitrate` arguments
- `add_generation_control_args(parser)` - Add batch control arguments (`--batch-runs`, `--no-single`, etc.)
- `add_playback_args(parser)` - Add `--no-play` argument
- `add_voice_selection_args(parser, choices)` - Add voice profile selection
- `add_common_args(parser, **kwargs)` - Add all common arguments at once
- `validate_generation_args(args)` - Validate argument combinations
- `get_generation_modes(args)` - Determine which generation modes to run
- `create_standard_parser(description, **kwargs)` - Create complete parser with all common args

**Usage Example**:
```python
from .utils.cli_args import create_standard_parser, validate_generation_args

parser = create_standard_parser(
    description="Voice cloning script",
    include_voice_selection=True,
    voice_choices=["voice1", "voice2"]
)
args = parser.parse_args()
validate_generation_args(args)
```

---

### `config_loader.py`

**Purpose**: JSON configuration management including profile loading, validation, and default configuration creation.

**Key Functions**:
- `load_json_config(config_path, create_default_func=None)` - Load JSON with error handling
- `process_text_fields(profiles)` - Process text fields that may be file paths
- `load_voice_clone_profiles()` - Load voice cloning profiles
- `load_custom_voice_profiles()` - Load custom voice profiles  
- `load_voice_design_profiles()` - Load voice design profiles
- `load_conversation_scripts()` - Load conversation script definitions
- `validate_profile_structure(profile, required_fields)` - Validate profile format
- `get_profile_choices(profiles)` - Extract profile names for CLI choices
- `get_default_profile(profiles, preferred_name=None)` - Get default profile

**Default Creation Functions**:
- `create_default_voice_clone_profiles()` - Create default voice clone config
- `create_default_custom_voice_profiles()` - Create default custom voice config
- `create_default_voice_design_profiles()` - Create default voice design config

**Usage Example**:
```python
from .utils.config_loader import load_voice_clone_profiles, get_profile_choices

profiles = load_voice_clone_profiles()
choices = get_profile_choices(profiles)
default_profile = get_default_profile(profiles, "MyVoice")
```

---

### `model_utils.py`

**Purpose**: Qwen3-TTS model loading with device detection (CUDA/CPU), memory optimization, and progress tracking.

**Key Functions**:
- `get_device()` - Detect optimal device (CUDA/CPU) for model loading
- `load_model_with_device(model_path, device=None)` - Load model with device selection and progress
- `load_voice_clone_model(model_path=None)` - Load voice cloning model
- `load_custom_voice_model(model_path=None)` - Load custom voice model
- `load_voice_design_model(model_path=None)` - Load voice design model
- `get_model_memory_usage(model)` - Get model memory footprint
- `clear_gpu_cache()` - Clear GPU memory cache
- `get_device_info()` - Get device capabilities and memory info
- `validate_model_path(model_path)` - Validate model directory structure

**Usage Example**:
```python
from .utils.model_utils import load_voice_clone_model, get_device

device = get_device()  # "cuda" or "cpu"
model = load_voice_clone_model("Qwen_Models/Qwen3-TTS-12Hz-1.7B-Base")
```

---

### `file_utils.py`

**Purpose**: Safe file operations, text loading, directory management, and path utilities used across all scripts.

**Key Functions**:
- `load_text_from_file_or_string(input_value)` - Load text from file path or use as literal string
- `ensure_directory_exists(directory_path)` - Create directory and parents if needed
- `get_safe_filename(filename, max_length=255)` - Create filesystem-safe filenames
- `get_unique_filepath(base_path, max_attempts=1000)` - Generate unique file paths
- `read_text_file(file_path, strip_whitespace=True)` - Read text file with error handling
- `write_text_file(file_path, content, create_dirs=True)` - Write text file safely
- `get_file_info(file_path)` - Get file metadata (size, modification time, etc.)
- `find_files(directory, pattern="*", recursive=True)` - Find files matching pattern
- `copy_file(source, destination, create_dirs=True)` - Copy files safely
- `validate_file_exists(file_path)` - Validate file exists and is readable
- `get_file_extension(file_path, include_dot=True)` - Extract file extension

**Usage Example**:
```python
from .utils.file_utils import load_text_from_file_or_string, ensure_directory_exists

# Load text (from file or use as literal)
text = load_text_from_file_or_string("texts/script.txt")

# Ensure output directory exists
ensure_directory_exists("output/MyVoice")

# Create safe filename
safe_name = get_safe_filename("My Voice: Sample.wav")  # "My_Voice__Sample.wav"
```

## 🔄 Import Patterns

### Direct Imports from Utility Modules
```python
# Import specific functions you need
from .utils.progress import print_progress, print_error
from .utils.audio_utils import save_audio, play_audio
from .utils.config_loader import load_voice_clone_profiles
```

### Package-Level Imports (via `__init__.py`)
```python
# Import commonly used functions from the package
from .utils import print_progress, save_audio, load_voice_clone_profiles
```

## 🧪 Testing

Each utility module has comprehensive unit tests located in the `tests/` directory:

- `tests/test_utils_progress.py` - Progress and error handling tests
- `tests/test_utils_audio_utils.py` - Audio processing tests
- `tests/test_utils_cli_args.py` - CLI argument parsing tests
- `tests/test_utils_config_loader.py` - Configuration loading tests
- `tests/test_utils_model_utils.py` - Model loading tests
- `tests/test_utils_file_utils.py` - File operation tests

Run utility tests:
```bash
# Test all utilities
python -m unittest tests.test_utils_* -v

# Test specific utility
python -m unittest tests.test_utils_progress -v
```

## 🔧 Dependencies

### Required Dependencies
- **Built-in modules**: `os`, `sys`, `pathlib`, `json`, `argparse`, `wave`, `time`, `re`
- **Third-party**: `numpy`, `torch`

### Optional Dependencies
- **`tqdm`**: Progress bars (fallback available)
- **`pydub`**: MP3 conversion (WAV-only fallback)
- **`ffmpeg`**: Audio format conversion (required for `pydub` MP3 support)
- **`pygame`**: Audio playback (fallback to `winsound` on Windows, or no playback)

The utility modules gracefully handle missing optional dependencies and provide appropriate fallbacks or warnings.

## 🏗️ Architecture Benefits

The modular utility architecture provides several key benefits:

1. **Code Reuse**: Common functionality is centralized and reused across all scripts
2. **Consistency**: Standardized interfaces for CLI args, error handling, and file operations
3. **Maintainability**: Changes to common functionality only need to be made in one place
4. **Testability**: Each utility module can be tested independently
5. **Flexibility**: Optional dependencies are handled gracefully with fallbacks
6. **Extensibility**: New utility functions can be easily added without modifying existing scripts

This design makes the Voice Clone Toolkit more robust, maintainable, and easier to extend with new features.