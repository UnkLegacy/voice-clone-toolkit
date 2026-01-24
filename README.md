# Qwen3-TTS Voice Clone Scripts

A collection of Python scripts for voice cloning and text-to-speech generation using the Qwen3-TTS models.

## 📜 Available Scripts

- **Clone_Voice.py** - Clone voices and generate speech with single or multiple voice profiles
- **Clone_Voice_Conversation.py** - Generate conversations between multiple cloned voices with script support
- **Voice_Design.py** - Design custom voices using natural language descriptions
- **Voice_Design_Clone.py** - Combine voice design with cloning for consistent character voices
- **Custom_Voice.py** - Generate speech with custom voice models

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Voice Profile Structure](#voice-profile-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Voice Cloning**: Clone any voice using reference audio and transcript
- **Batch Generation**: Efficiently generate multiple audio samples with the same cloned voice
- **JSON Configuration**: Easy-to-edit voice profiles stored in JSON format
- **Flexible Control**: Toggle features via config file or command-line arguments
- **Progress Tracking**: Visual progress bars with tqdm integration
- **Audio Playback**: Automatic playback of generated audio (optional)
- **Multiple Profiles**: Store and switch between multiple voice profiles easily

## 🔧 Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/Qwen3-TTS_Scripts.git
cd Qwen3-TTS_Scripts

# Create virtual environment (recommended)
python -m venv qwen-env
source qwen-env/bin/activate  # Linux/Mac
# or: qwen-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

For detailed installation instructions, troubleshooting, and platform-specific notes, see **[INSTALLATION.md](INSTALLATION.md)**.

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch with CUDA support (if using GPU)

### Install from requirements.txt

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install development dependencies (includes testing tools)
pip install -r requirements-dev.txt
```

### Manual Installation

```bash
# Core dependencies
pip install torch numpy

# Qwen3-TTS package
pip install qwen-tts

# Optional but recommended
pip install tqdm      # For progress bars
pip install pygame    # For audio playback (Windows/Linux/Mac)
```

### PyTorch Installation Notes

For GPU support, install PyTorch with CUDA:

```bash
# CUDA 11.8 (RTX 30xx, RTX 40xx)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (RTX 40xx)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8 (RTX 50xx series - RTX 5080, etc.)
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# CPU only (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Note**: RTX 50xx series cards require nightly PyTorch builds with CUDA 12.8 support.

📊 **GPU Compatibility**: For detailed GPU-specific installation instructions, see **[GPU_COMPATIBILITY.md](GPU_COMPATIBILITY.md)**.

### Windows-specific Audio Playback

On Windows, if pygame installation fails, the script will fall back to the built-in `winsound` module.

## 🧪 Testing

The project includes comprehensive unit tests for all scripts.

### Run All Tests

```bash
# Easy way
python run_tests.py

# Or with unittest
python -m unittest discover tests -v
```

### Run Specific Tests

```bash
# Test a specific module
python -m unittest tests.test_clone_voice

# Test a specific class
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles

# Test a specific method
python -m unittest tests.test_clone_voice.TestLoadVoiceProfiles.test_load_valid_profiles
```

### Test Coverage

```bash
pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html  # Generate HTML report
```

For detailed testing information, see [TESTING.md](TESTING.md).

## ⚙️ Configuration

### Quick Toggle Settings

Edit the top of `Clone_Voice.py` to quickly change default behavior:

```python
# Default settings (can be overridden by command-line arguments)
DEFAULT_VOICE = "Grandma"  # Change this to switch between voices (or use list: ["DougDoug", "Grandma"])
RUN_SINGLE = True          # Set to False to skip single generation
RUN_BATCH = True           # Set to False to skip batch generation
PLAY_AUDIO = True          # Set to False to skip audio playback
COMPARE_MODE = False       # Set to True to use sample_transcript as single_text (for quality comparison)
```

**Process multiple voices:**
```python
DEFAULT_VOICE = ["DougDoug", "Grandma", "Example_Grandma"]  # Process multiple voices by default
```

### Voice Profiles Configuration

Voice profiles are stored in `config/voice_clone_profiles.json`. This file contains all the information needed for each voice:

```json
{
  "ProfileName": {
    "voice_sample_file": "./input/audio_file.wav",
    "sample_transcript": "Transcript of the reference audio...",
    "single_text": "Text to use for single generation...",
    "batch_texts": [
      "First batch text...",
      "Second batch text...",
      "Third batch text..."
    ]
  }
}
```

#### Using Text Files Instead of Inline Strings

**All text fields support file paths!** You can use file paths instead of inline text for `sample_transcript`, `single_text`, and `batch_texts`. This is especially useful for longer content:

```json
{
  "MyVoice": {
    "voice_sample_file": "./input/MyVoice.wav",
    "sample_transcript": "./texts/MyVoice_transcript.txt",
    "single_text": "./texts/MyVoice_single.txt",
    "batch_texts": [
      "./texts/batch_1.txt",
      "./texts/batch_2.txt",
      "./texts/batch_3.txt"
    ]
  }
}
```

**Example with file loading** (see `Example_Grandma` profile):
```json
{
  "Example_Grandma": {
    "voice_sample_file": "./input/Grandma.wav",
    "sample_transcript": "./texts/example_transcript.txt",
    "single_text": "./texts/example_single.txt",
    "batch_texts": [
      "You can mix inline text...",
      "./texts/example_batch_1.txt"
    ]
  }
}
```

**Benefits of using text files:**
- Easier to manage longer texts (especially transcripts!)
- Keep your JSON config clean and readable
- Edit texts without touching the config file
- Reuse the same text across multiple profiles
- No need to escape quotes or special characters

#### Adding a New Voice Profile

1. Open `config/voice_clone_profiles.json`
2. Add a new entry following the structure above:

```json
{
  "MyVoice": {
    "voice_sample_file": "./input/MyVoice.wav",
    "sample_transcript": "This is the reference transcript matching the audio.",
    "single_text": "This is what will be generated for single mode.",
    "batch_texts": [
      "First sentence for batch generation.",
      "Second sentence for batch generation.",
      "Third sentence for batch generation."
    ]
  }
}
```

3. Place your reference audio file in the `input/` directory
4. Update `DEFAULT_VOICE` in `Clone_Voice.py` or use `--voice MyVoice` flag

#### Using Text Files for Longer Content

For longer texts (especially transcripts!), you can reference text files instead of inline strings:

1. Create text files in a `texts/` directory (or any location you prefer):

```
texts/
├── narrator_transcript.txt    (for sample_transcript)
├── narrator_single.txt         (for single_text)
├── narrator_batch_1.txt        (for batch_texts)
└── narrator_batch_2.txt        (for batch_texts)
```

**Example text file content** (`texts/narrator_transcript.txt`):
```
This is the transcript of my reference audio.
It can be multiple lines and paragraphs.
No need to escape quotes or worry about JSON formatting!
```

2. Reference them in your profile:

```json
{
  "Narrator": {
    "voice_sample_file": "./input/Narrator.wav",
    "sample_transcript": "./texts/narrator_transcript.txt",
    "single_text": "./texts/narrator_single.txt",
    "batch_texts": [
      "./texts/narrator_batch_1.txt",
      "./texts/narrator_batch_2.txt",
      "You can also mix inline text with file paths!"
    ]
  }
}
```

**How it works:**
- The script automatically detects if a value is a file path and loads the content
- If the file doesn't exist, it treats the value as literal text (fallback)
- Works for all text fields: `sample_transcript`, `single_text`, and `batch_texts`
- You'll see `[INFO] Loaded text from file: ...` messages when files are loaded

## 🚀 Usage

### Basic Usage

Run with default settings (as configured in the script):

```bash
python Clone_Voice.py
```

### Using a Specific Voice Profile

```bash
python Clone_Voice.py --voice DougDoug
```

### Using Multiple Voice Profiles

```bash
# Process multiple voices in one run
python Clone_Voice.py --voices DougDoug Grandma

# Process all three example profiles
python Clone_Voice.py --voices DougDoug Grandma Example_Grandma
```

### List Available Voice Profiles

```bash
python Clone_Voice.py --list-voices
```

This will display all configured voice profiles with their details.

## 📝 Command-Line Arguments

### Voice Selection

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--voice` | `-v` | Select a single voice profile to use | `--voice Grandma` |
| `--voices` | | Select multiple voice profiles to process | `--voices DougDoug Grandma` |
| `--list-voices` | | List all available voice profiles and exit | `--list-voices` |

### Generation Control

| Argument | Description | Use Case |
|----------|-------------|----------|
| `--only-single` | Only run single voice generation | Quick testing of one sample |
| `--only-batch` | Only run batch voice generation | Generate multiple samples efficiently |
| `--no-single` | Skip single voice generation | When you only want batch outputs |
| `--no-batch` | Skip batch voice generation | When you only want one sample |

### Playback Control

| Argument | Description | Use Case |
|----------|-------------|----------|
| `--no-play` | Skip audio playback | Running in headless/batch mode |

### Quality Comparison

| Argument | Description | Use Case |
|----------|-------------|----------|
| `--compare` | Use sample_transcript as single_text | Compare generated audio quality against the original reference audio |

## 📁 Voice Profile Structure

Each voice profile in `config/voice_clone_profiles.json` contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `voice_sample_file` | string | Yes | Path to reference audio file (WAV format recommended) |
| `sample_transcript` | string | Yes | Accurate transcript of the reference audio (or path to .txt file) |
| `single_text` | string | Yes | Text to synthesize in single generation mode (or path to .txt file, ignored in compare mode) |
| `batch_texts` | array | Yes | List of texts to synthesize in batch mode (can be inline strings or paths to .txt files) |

### Tips for Best Results

1. **Voice Sample File** (`voice_sample_file`): 
   - Use high-quality, clear audio (no background noise)
   - 5-30 seconds is ideal
   - WAV format recommended (24kHz or 48kHz sample rate)

2. **Sample Transcript** (`sample_transcript`):
   - Must be an accurate transcript of the reference audio
   - Include punctuation for better prosody
   - Match the actual spoken words exactly
   - Use `--compare` mode to test quality by generating this exact text

3. **Generation Text** (`single_text` and `batch_texts`):
   - Can be any text you want to synthesize (inline or file path)
   - Keep sentences natural and grammatically correct
   - Use punctuation to control pacing
   - For longer texts, use file paths (e.g., `"./texts/my_long_text.txt"`) to keep config clean

## 💡 Examples

### Example 1: Quick Test with Default Voice

```bash
# Uses the DEFAULT_VOICE setting from the script
python Clone_Voice.py --only-single --no-play
```

### Example 1b: Process Multiple Default Voices

```python
# In Clone_Voice.py, set:
DEFAULT_VOICE = ["DougDoug", "Grandma"]
```

```bash
# Then run without arguments to process all default voices
python Clone_Voice.py
```

### Example 2: Generate Multiple Samples

```bash
# Generate only batch outputs with DougDoug voice
python Clone_Voice.py --voice DougDoug --only-batch
```

### Example 3: Silent Batch Processing

```bash
# Generate all outputs without playback (good for batch processing)
python Clone_Voice.py --voice Grandma --no-play
```

### Example 4: Full Pipeline

```bash
# Run both single and batch with playback
python Clone_Voice.py --voice DougDoug
```

### Example 5: Explore Available Voices

```bash
# List all configured voice profiles
python Clone_Voice.py --list-voices
```

Output:
```
============================================================
AVAILABLE VOICE PROFILES
============================================================

DougDoug:
  Voice sample file: ./input/DougDoug.wav
  Sample transcript: It was developed by Atrioc.  So one time, like probably...
  Single text: What's up Twitch chat?!  I'm DougDoug!  And today we're g...
  Batch texts: 4 samples

Grandma:
  Voice sample file: ./input/Grandma.wav
  Sample transcript: Can sit and talk to me and I'll ignore the other two. ...
  Single text: I don't think that PC will ever be back in stock, I imagi...
  Batch texts: 4 samples
============================================================
```

### Example 6: Compare Mode (Quality Testing)

```bash
# Generate the same text as the reference audio to compare quality
python Clone_Voice.py --voice DougDoug --compare --only-single
```

This is useful for:
- Testing voice cloning quality by comparing generated audio with the original
- Hearing how well the model reproduces the exact same text
- Quality assurance before generating custom text

In compare mode, the script will:
1. Use the `sample_transcript` as the text to generate
2. Create a cloned version of the reference audio
3. Allow you to compare the original vs. cloned audio side-by-side

### Example 7: Process Multiple Voices

```bash
# Generate outputs for multiple voice profiles in one run
python Clone_Voice.py --voices DougDoug Grandma

# Process multiple voices with specific settings
python Clone_Voice.py --voices DougDoug Grandma Example_Grandma --only-single --no-play

# Or set multiple voices as default in the config
# In Clone_Voice.py: DEFAULT_VOICE = ["DougDoug", "Grandma"]
python Clone_Voice.py
```

**Benefits:**
- Efficient: Model is loaded once and reused for all voices
- Convenient: Process your entire voice library in one command
- Consistent: All voices use the same generation settings
- Time-saving: Perfect for batch processing and comparisons

**Example output:**
```
============================================================
PROCESSING 2 VOICE PROFILE(S)
============================================================
[INFO] Voices to process: DougDoug, Grandma
[INFO] Loading model...

================================================================================
PROCESSING VOICE 1/2: DougDoug
================================================================================
[INFO] VOICE CLONING: DougDoug
...

================================================================================
PROCESSING VOICE 2/2: Grandma
================================================================================
[INFO] VOICE CLONING: Grandma
...

================================================================================
SUMMARY
================================================================================
[INFO] Successfully processed: 2/2 voices
[INFO] Total execution time: 45.32 seconds
================================================================================
```

## 📂 Output Structure

Generated audio files are organized by voice profile:

```
output/
└── Clone_Voice/
    ├── DougDoug/
    │   ├── DougDoug_clone.wav      # Single generation output
    │   ├── DougDoug_clone_1.wav    # Batch generation output 1
    │   ├── DougDoug_clone_2.wav    # Batch generation output 2
    │   ├── DougDoug_clone_3.wav    # Batch generation output 3
    │   └── DougDoug_clone_4.wav    # Batch generation output 4
    └── Grandma/
        ├── Grandma_clone.wav
        ├── Grandma_clone_1.wav
        ├── Grandma_clone_2.wav
        ├── Grandma_clone_3.wav
        └── Grandma_clone_4.wav
```

When processing multiple voices with `--voices`, each voice creates its own subdirectory. This keeps outputs organized and prevents conflicts between different voice profiles.

## 🔍 Troubleshooting

### Common Issues

#### "Voice profile not found"

**Problem**: The specified voice profile doesn't exist in the config file.

**Solution**: 
```bash
# Check available profiles
python Clone_Voice.py --list-voices

# Use an existing profile
python Clone_Voice.py --voice DougDoug
```

#### "Reference audio not found"

**Problem**: The audio file path in the profile is incorrect or the file doesn't exist.

**Solution**:
1. Check that the audio file exists at the specified path
2. Verify the path in `config/voice_clone_profiles.json`
3. Ensure the path is relative to the script location

#### "No audio playback available"

**Problem**: Neither pygame nor winsound is available for playback.

**Solution**:
```bash
# Install pygame for cross-platform audio playback
pip install pygame

# Or run without playback
python Clone_Voice.py --no-play
```

#### "CUDA out of memory"

**Problem**: GPU doesn't have enough memory.

**Solution**:
- Close other applications using GPU
- Script will automatically fall back to CPU if CUDA fails
- Consider using a smaller batch size by editing the `batch_texts` array

#### "JSON decode error"

**Problem**: The config file has invalid JSON syntax.

**Solution**:
1. Validate your JSON at [jsonlint.com](https://jsonlint.com/)
2. Check for:
   - Missing commas between entries
   - Unclosed quotes or brackets
   - Trailing commas (not allowed in JSON)

#### "Loaded text from file" messages

**Problem**: Not really a problem! Just informational messages.

**Explanation**: When you use file paths for `single_text`, `batch_texts`, or `sample_transcript`, the script will display:
```
[INFO] Loaded text from file: ./texts/my_text.txt
```

This confirms that the file was successfully loaded. If you see "Using value as literal text instead," it means the file wasn't found and the value is being treated as literal text.

### Performance Tips

1. **GPU vs CPU**: 
   - GPU is 10-50x faster than CPU
   - Script automatically detects and uses GPU if available

2. **Batch Generation**:
   - More efficient than generating multiple singles
   - Reuses the voice prompt for faster processing

3. **Model Location**:
   - Keep models on fast storage (SSD)
   - Default location: `Qwen_Models/Qwen3-TTS-12Hz-1.7B-Base/`

## 📄 License

This project uses the Qwen3-TTS model. Please refer to the [Qwen3-TTS license](https://github.com/QwenLM/Qwen3-TTS) for usage terms.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For issues related to:
- **Script functionality**: Open an issue in this repository
- **Qwen3-TTS model**: Visit the [official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)

## 🎭 Conversation Scripts (Clone_Voice_Conversation.py)

### Overview

Generate realistic conversations between multiple cloned voices using script files. Perfect for:
- Podcasts and dialogues
- Character interactions
- Educational content
- Storytelling with multiple narrators

### Quick Start

```bash
# Run with default conversation
python Clone_Voice_Conversation.py

# Use a specific script
python Clone_Voice_Conversation.py --script tech_discussion

# List available scripts
python Clone_Voice_Conversation.py --list-scripts

# Generate without playback
python Clone_Voice_Conversation.py --no-play

# Keep lines separate (no concatenation)
python Clone_Voice_Conversation.py --no-concatenate
```

### Configuration

Conversation scripts are defined in `config/conversation_scripts.json`:

```json
{
  "my_conversation": {
    "actors": ["DougDoug", "Example_Grandma"],
    "script": [
      "[DougDoug] Hi Grandma, how are you?",
      "[Example_Grandma] I'm doing great, dear!",
      "[DougDoug] That's wonderful to hear!"
    ]
  }
}
```

### Script Format

**Inline script (in JSON):**
```json
"script": [
  "[Actor1] First line of dialogue",
  "[Actor2] Second line of dialogue",
  "[Actor1] Third line back to Actor1"
]
```

**File-based script:**
```json
"script": "./scripts/my_conversation.txt"
```

**Script file format (`.txt`):**
```
[DougDoug] Hey there, how's it going?
[Example_Grandma] Oh wonderful, dear! How are you?
[DougDoug] I'm doing great, thanks for asking!
```

### Output Structure

```
output/
└── Conversations/
    └── my_conversation/
        ├── my_conversation_line_001_DougDoug.wav       # Individual lines
        ├── my_conversation_line_002_Example_Grandma.wav
        ├── my_conversation_line_003_DougDoug.wav
        └── my_conversation_full.wav                    # Concatenated conversation
```

### Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--script` | Select conversation script | `--script tech_discussion` |
| `--list-scripts` | List all available scripts | `--list-scripts` |
| `--no-play` | Skip audio playback | `--no-play` |
| `--no-concatenate` | Don't create combined audio file | `--no-concatenate` |

### Example Scripts

The project includes several example conversations:

1. **example_conversation** - Simple greeting between DougDoug and Grandma
2. **tech_discussion** - Technology discussion showcasing Grandma's tech knowledge
3. **script_file_example** - Demonstrates loading scripts from external files

### Creating Custom Scripts

1. **Add actors to voice profiles** (if not already present):
   ```json
   // In config/voice_clone_profiles.json
   {
     "MyActor": {
       "voice_sample_file": "./input/MyActor.wav",
       "sample_transcript": "Reference transcript..."
     }
   }
   ```

2. **Create script file** (optional):
   ```
   // In scripts/my_script.txt
   [MyActor] Hello, this is my first line!
   [OtherActor] And this is my response!
   ```

3. **Add to conversation config**:
   ```json
   // In config/conversation_scripts.json
   {
     "my_custom_script": {
       "actors": ["MyActor", "OtherActor"],
       "script": "./scripts/my_script.txt"
     }
   }
   ```

4. **Generate**:
   ```bash
   python Clone_Voice_Conversation.py --script my_custom_script
   ```

### Tips for Best Results

- **Natural Dialogue**: Write as people actually speak (contractions, interruptions, etc.)
- **Punctuation**: Use commas, periods, and exclamation points to control pacing
- **Line Length**: Keep individual lines conversational (1-3 sentences)
- **Actor Consistency**: Ensure actor names match voice profile names exactly
- **Test First**: Start with short scripts to verify voice quality

---

**Note**: This is an unofficial collection of scripts for working with Qwen3-TTS. For the official implementation and documentation, please visit the [Qwen3-TTS GitHub repository](https://github.com/QwenLM/Qwen3-TTS).
