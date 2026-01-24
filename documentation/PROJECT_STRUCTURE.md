# Project Structure

Complete directory structure and file organization for the Qwen3-TTS Scripts project.

## Root Directory

```
Qwen3-TTS_Scripts/
├── .gitignore                    # Git ignore rules
├── README.md                     # Main project documentation
├── INSTALLATION.md               # Installation guide
├── run_tests.py                  # Test runner script
│
├── documentation/                # Documentation files
│   ├── CONVERSATION_GUIDE.md          # Conversation script guide
│   ├── GPU_COMPATIBILITY.md           # GPU compatibility guide
│   ├── PROJECT_STRUCTURE.md           # This file
│   └── TESTING.md                     # Testing guide
│
├── src/                          # Source code directory
│   ├── __init__.py                    # Package initialization
│   ├── clone_voice.py                 # Main voice cloning script
│   ├── clone_voice_conversation.py    # Conversation generation script
│   ├── custom_voice.py                # Custom voice generation
│   ├── voice_design.py                # Voice design script
│   └── voice_design_clone.py          # Combined voice design + clone
│
├── config/                       # Configuration files
│   ├── voice_clone_profiles.json      # Voice profile definitions
│   └── conversation_scripts.json      # Conversation script definitions
│
├── input/                        # Reference audio files
│   ├── .gitkeep                       # Keeps directory in git
│   └── README.md                      # Input directory guide
│
├── texts/                        # Text files for voice profiles
│   ├── README.md                      # Text files guide
│   ├── example_transcript.txt         # Example transcript
│   ├── example_single.txt             # Example single text
│   ├── example_batch_1.txt            # Example batch text
│   └── dougdoug_transcript.txt        # DougDoug transcript
│
├── conversation_scripts/         # Conversation script files
│   ├── README.md                      # Scripts directory guide
│   └── example_script.txt             # Example conversation
│
├── tests/                        # Unit tests
│   ├── __init__.py                    # Test package init
│   ├── README.md                      # Testing documentation
│   ├── test_clone_voice.py            # Clone_Voice tests
│   └── test_clone_voice_conversation.py  # Conversation tests
│
├── output/                       # Generated audio files (git ignored)
│   ├── Clone_Voice/                   # Clone voice outputs
│   │   ├── DougDoug/
│   │   │   ├── DougDoug_clone.wav
│   │   │   ├── DougDoug_clone_1.wav
│   │   │   └── ...
│   │   └── Grandma/
│   │       └── ...
│   ├── Conversations/                 # Conversation outputs
│   │   └── example_conversation/
│   │       ├── example_conversation_line_001_DougDoug.wav
│   │       └── example_conversation_full.wav
│   └── ...
│
├── Qwen_Models/                  # Model files (git ignored, downloaded separately)
│   ├── Qwen3-TTS-12Hz-1.7B-Base/
│   ├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   └── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
│
└── qwen-env/                     # Virtual environment (git ignored)
    └── ...
```

## Directory Descriptions

### Source Code (`src/`)

Main Python scripts located in `src/` directory:

| File | Purpose |
|------|---------|
| `src/clone_voice.py` | Main voice cloning with single/batch generation |
| `src/clone_voice_conversation.py` | Multi-actor conversation generation |
| `src/custom_voice.py` | Custom voice model generation |
| `src/voice_design.py` | Design voices with natural language |
| `src/voice_design_clone.py` | Combine voice design with cloning |

### Root Scripts

| File | Purpose |
|------|---------|
| `run_tests.py` | Test runner for all unit tests |

### Configuration (`config/`)

Contains JSON configuration files:

- **voice_clone_profiles.json**: Voice profiles with reference audio, transcripts, and generation texts
- **conversation_scripts.json**: Conversation scripts with actors and dialogues

### Input (`input/`)

Reference audio files for voice cloning:

- Place your `.wav` files here
- Files are git-ignored (not committed)
- Directory structure is preserved

### Texts (`texts/`)

Text content files that can be referenced in configs:

- `sample_transcript`: Reference audio transcripts
- `single_text`: Single generation text
- `batch_texts`: Batch generation texts

### Conversation Scripts (`conversation_scripts/`)

Conversation script files:

- Text files with `[Actor] dialogue` format
- Referenced in `conversation_scripts.json`
- Can contain multi-actor conversations

### Tests (`tests/`)

Unit tests for all scripts:

- `test_clone_voice.py`: Tests for Clone_Voice.py
- `test_clone_voice_conversation.py`: Tests for Clone_Voice_Conversation.py
- Run with `python run_tests.py`

### Output (`output/`)

Generated audio files (git-ignored):

- Organized by script and voice profile
- Individual and concatenated audio files
- Automatically created by scripts

### Models (`Qwen_Models/`)

Downloaded model files (git-ignored):

- Too large for git
- Download separately from Hugging Face
- Three model variants supported

## Git Ignore Rules

The following are **not committed** to git:

- `output/` - Generated audio files
- `Qwen_Models/` - Model files
- `*.wav` - All WAV audio files
- `input/*` - Input audio files (except README.md)
- `__pycache__/` - Python cache
- `.idea/` - PyCharm files
- `.cursor/` - CursorAI files
- `qwen-env/` - Virtual environment

## File Naming Conventions

### Output Files

**Clone Voice:**
```
{voice_name}_clone.wav              # Single generation
{voice_name}_clone_1.wav            # Batch generation #1
{voice_name}_clone_2.wav            # Batch generation #2
```

**Conversations:**
```
{script_name}_line_001_{actor}.wav  # Individual line
{script_name}_line_002_{actor}.wav  # Next line
{script_name}_full.wav              # Concatenated audio
```

### Configuration Files

- **JSON format**: All configs use JSON
- **snake_case**: Field names use snake_case
- **Descriptive names**: Clear, self-documenting names

### Script Files

- **Markdown**: Documentation uses `.md`
- **Text content**: Use `.txt` for content files
- **Python**: Use `.py` for scripts

## Adding New Components

### New Voice Profile

1. Add audio to `input/`
2. Add entry to `config/voice_clone_profiles.json`
3. Optionally create text files in `texts/`

### New Conversation

1. Create script in `scripts/` (optional)
2. Add entry to `config/conversation_scripts.json`
3. Run with `--script` flag

### New Test

1. Create `tests/test_new_module.py`
2. Import module and create test cases
3. Run with `python run_tests.py`

## Best Practices

1. **Keep configs in `config/`**: All JSON configuration files
2. **Keep content in `texts/` or `scripts/`**: Separates code from content
3. **Don't commit audio**: Large files, user-specific
4. **Don't commit models**: Download separately
5. **Write tests**: For any new functionality
6. **Update documentation**: Keep READMEs current

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation (root) |
| `INSTALLATION.md` | Installation guide (root) |
| `documentation/CONVERSATION_GUIDE.md` | Detailed conversation script guide |
| `documentation/GPU_COMPATIBILITY.md` | GPU compatibility and PyTorch installation |
| `documentation/PROJECT_STRUCTURE.md` | This file - project organization |
| `documentation/TESTING.md` | Testing guide and best practices |
| `input/README.md` | Input directory guide |
| `texts/README.md` | Text files guide |
| `conversation_scripts/README.md` | Conversation scripts guide |
| `tests/README.md` | Testing documentation |

## Future Expansion

When adding new scripts or features:

1. Add script to root directory
2. Create corresponding test file in `tests/`
3. Add configuration (if needed) to `config/`
4. Update documentation
5. Add examples
6. Update this structure guide

---

**Remember**: A well-organized project is easier to maintain and contribute to! 📁✨
