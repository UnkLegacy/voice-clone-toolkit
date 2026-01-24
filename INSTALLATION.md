# Installation Guide

Quick start guide for installing Qwen3-TTS Voice Clone Scripts.

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **GPU** (optional but recommended): NVIDIA GPU with CUDA support

## Quick Install

### 1. Clone or Download Repository

```bash
git clone https://github.com/yourusername/Qwen3-TTS_Scripts.git
cd Qwen3-TTS_Scripts
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv qwen-env

# Activate virtual environment
# Windows:
qwen-env\Scripts\activate

# Linux/Mac:
source qwen-env/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

That's it! You're ready to use the scripts.

## Advanced Installation

### Check Your GPU

Before installing, check which GPU you have:

```bash
# Windows
nvidia-smi

# Linux
nvidia-smi

# Or check in Python
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

### GPU Support (NVIDIA CUDA)

For faster generation with GPU, choose the appropriate CUDA version for your GPU:

#### RTX 30xx / RTX 40xx Series (CUDA 11.8)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### RTX 40xx Series (CUDA 12.1)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### RTX 50xx Series (CUDA 12.8 - Requires Nightly Build)
```bash
# For RTX 5080 and other RTX 50xx cards
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note**: RTX 50xx series cards require the nightly (pre-release) builds of PyTorch with CUDA 12.8 support.

#### Quick Reference Table

| GPU Series | Recommended CUDA | PyTorch Install Command |
|------------|------------------|------------------------|
| RTX 50xx (5080, 5090, etc.) | CUDA 12.8 | `pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128` |
| RTX 40xx (4090, 4080, etc.) | CUDA 12.1 or 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| RTX 30xx (3090, 3080, etc.) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| RTX 20xx (2080, 2070, etc.) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| GTX 16xx | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| No GPU / CPU only | N/A | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

📊 **For detailed GPU compatibility info, troubleshooting, and performance comparisons, see [GPU_COMPATIBILITY.md](GPU_COMPATIBILITY.md)**.

### CPU Only (Smaller Download)

If you don't have a GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Development Installation

If you want to contribute or run tests:

```bash
pip install -r requirements-dev.txt
```

This includes:
- Testing tools (pytest, coverage)
- Code quality tools (flake8, black, mypy)
- Documentation tools (sphinx)

## Download Models

Download the Qwen3-TTS models from Hugging Face:

```bash
# Create models directory
mkdir -p Qwen_Models

# Download models (example using git lfs)
cd Qwen_Models

# Base model (for voice cloning)
git lfs clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base

# CustomVoice model
git lfs clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# VoiceDesign model
git lfs clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

cd ..
```

**Note**: Models are large (~3-5GB each). Make sure you have enough disk space.

## Add Reference Audio

Place your reference audio files in the `input/` directory:

```bash
# Copy your audio files
cp /path/to/your/audio.wav input/
```

Supported formats:
- WAV (recommended)
- MP3
- Other formats supported by the library

## Verify Installation

Run a simple test:

```bash
# Run unit tests
python run_tests.py

# Or try listing voice profiles
python Clone_Voice.py --list-voices
```

## Troubleshooting

### "No module named 'qwen_tts'"

Install the Qwen3-TTS package:

```bash
pip install qwen-tts

# Or from GitHub
pip install git+https://github.com/QwenLM/Qwen-Audio.git
```

### "CUDA out of memory"

Try:
1. Close other applications
2. Reduce batch size in scripts
3. Use CPU mode: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### RTX 50xx Series / CUDA 12.8 Issues

If you have an RTX 5080 or other RTX 50xx card and get CUDA errors:

```bash
# Use nightly builds with CUDA 12.8 support
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Symptoms:**
- "CUDA error: no kernel image is available for execution on the device"
- "CUDA capability sm_90 is not compatible"

**Solution:** RTX 50xx cards require PyTorch with CUDA 12.8, which is currently only available in nightly builds.

### "No audio playback"

Install pygame:

```bash
pip install pygame
```

On Windows, the script will fall back to `winsound` if pygame is not available.

### Import Errors

Make sure you're running from the project root directory:

```bash
cd /path/to/Qwen3-TTS_Scripts
python Clone_Voice.py --help
```

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt
- Virtual environment activation: `qwen-env\Scripts\activate`
- Audio playback works with both pygame and winsound

### Linux

- May need to install additional audio libraries:
  ```bash
  sudo apt-get install python3-pygame
  sudo apt-get install portaudio19-dev
  ```

### macOS

- May need to install audio libraries via Homebrew:
  ```bash
  brew install portaudio
  pip install pygame
  ```

## Next Steps

After installation:

1. **Read the README**: `README.md` for usage instructions
2. **Try examples**: Run with default settings first
3. **Add your voices**: Edit `config/voice_clone_profiles.json`
4. **Create conversations**: Edit `config/conversation_scripts.json`

## Updating

To update dependencies:

```bash
# Pull latest code
git pull

# Update packages
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove everything:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf qwen-env

# Remove models (if you want)
rm -rf Qwen_Models

# Remove generated outputs
rm -rf output
```

## Getting Help

- **Documentation**: Check `README.md`, `TESTING.md`, `CONVERSATION_GUIDE.md`
- **Issues**: Check if models are downloaded correctly
- **Tests**: Run `python run_tests.py` to verify setup

---

Happy voice cloning! 🎙️✨
