# Qwen Models Directory

This directory is for storing Qwen3-TTS model files used by the Voice Clone Toolkit.

## 📥 Model Downloads

Download the official Qwen3-TTS models from the [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS?tab=readme-ov-file#released-models-description-and-download).

### Recommended Model

**Qwen3-TTS-12Hz-1.7B-Base** (Recommended for most users)
- Balanced performance and quality
- Supports voice cloning and custom voices
- Good for both CPU and GPU

### Directory Structure

After downloading, your structure should look like:

```
Qwen_Models/
└── Qwen3-TTS-12Hz-1.7B-Base/
    ├── config.json
    ├── model.safetensors
    └── (other model files)
```

## ⚠️ Important Notes

- **Model files are NOT included in this repository** due to their large size
- You must download models separately before using the toolkit
- The scripts will automatically look for models in this directory
- Keep models on fast storage (SSD recommended) for best performance

## 📝 Additional Information

For detailed installation instructions, see [INSTALLATION.md](../INSTALLATION.md) in the project root.

For GPU-specific setup, see [GPU_COMPATIBILITY.md](../documentation/GPU_COMPATIBILITY.md).
