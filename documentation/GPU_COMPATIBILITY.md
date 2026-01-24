# GPU Compatibility Guide

Guide for installing PyTorch with the correct CUDA version for your NVIDIA GPU.

## Quick Reference

| GPU Series | CUDA Version | Installation Command |
|------------|--------------|---------------------|
| **RTX 50xx** (5090, 5080, 5070, etc.) | **CUDA 12.8** (Nightly) | See [RTX 50xx Instructions](#rtx-50xx-series-cuda-128) |
| **RTX 40xx** (4090, 4080, 4070, etc.) | CUDA 12.1 or 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| **RTX 30xx** (3090, 3080, 3070, etc.) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| **RTX 20xx** (2080 Ti, 2080, 2070, etc.) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| **GTX 16xx** (1660 Ti, 1660, 1650, etc.) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| **GTX 10xx** (1080 Ti, 1080, 1070, etc.) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| **No GPU / CPU Only** | N/A | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

## Detailed Instructions

### RTX 50xx Series (CUDA 12.8)

**Cards**: RTX 5090, RTX 5080, RTX 5070, etc.

**Requirements**: 
- PyTorch nightly builds (pre-release)
- CUDA 12.8 support

**Installation**:
```bash
# Install PyTorch and torchvision
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install torchaudio
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Why Nightly Builds?**
RTX 50xx cards use the Ada Lovelace Next architecture with compute capability SM_90+, which requires CUDA 12.8. This support is currently only available in PyTorch nightly builds.

**Verification**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
PyTorch version: 2.6.0.dev20250xxx (or similar)
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5080 (or your card)
```

### RTX 40xx Series (CUDA 12.1)

**Cards**: RTX 4090, RTX 4080, RTX 4070 Ti, RTX 4070, RTX 4060, etc.

**Installation**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Alternative** (CUDA 11.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### RTX 30xx Series (CUDA 11.8)

**Cards**: RTX 3090, RTX 3080 Ti, RTX 3080, RTX 3070 Ti, RTX 3070, RTX 3060, etc.

**Installation**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### RTX 20xx Series (CUDA 11.8)

**Cards**: RTX 2080 Ti, RTX 2080, RTX 2070, RTX 2060, etc.

**Installation**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### GTX 16xx Series (CUDA 11.8)

**Cards**: GTX 1660 Ti, GTX 1660, GTX 1650, etc.

**Installation**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only (No GPU)

If you don't have an NVIDIA GPU or want to test on CPU:

**Installation**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Note**: CPU generation will be significantly slower (10-50x) than GPU.

## Checking Your GPU

### Windows

```bash
nvidia-smi
```

### Linux

```bash
nvidia-smi
lspci | grep -i nvidia
```

### macOS

macOS doesn't support NVIDIA GPUs for CUDA. Use CPU installation.

### Python

```python
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No CUDA GPU available")
```

## Troubleshooting

### "CUDA error: no kernel image is available"

**Problem**: Your GPU is too new for the installed PyTorch version.

**Solution**: 
- RTX 50xx: Use nightly builds with CUDA 12.8
- RTX 40xx: Use CUDA 12.1 or 11.8
- Older cards: Use CUDA 11.8

### "RuntimeError: CUDA out of memory"

**Solutions**:
1. Close other GPU-intensive applications
2. Reduce batch size in scripts
3. Monitor GPU memory: `nvidia-smi`
4. Restart Python kernel/script

### "torch.cuda.is_available() returns False"

**Possible causes**:
1. NVIDIA drivers not installed
2. Wrong PyTorch version (CPU-only)
3. CUDA toolkit not installed

**Solutions**:
1. Install/update NVIDIA drivers
2. Reinstall PyTorch with correct CUDA version
3. Verify installation: `nvidia-smi`

### Nightly Build Instability

**Issue**: Nightly builds may have bugs or changes.

**Solution**:
- Check PyTorch GitHub for known issues
- Consider stable release when available for your GPU
- Pin specific nightly version if needed:
  ```bash
  pip install torch==2.6.0.dev20250115+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
  ```

## Performance Comparison

Approximate generation times for a 10-second audio clip:

| GPU | Time | Speed vs CPU |
|-----|------|--------------|
| RTX 5080 | ~2-3 sec | 30-40x faster |
| RTX 4090 | ~2-3 sec | 30-40x faster |
| RTX 3090 | ~3-4 sec | 20-30x faster |
| RTX 3070 | ~4-6 sec | 15-20x faster |
| CPU (Ryzen 9) | ~60-90 sec | 1x (baseline) |

*Note: Times are approximate and depend on model, settings, and system configuration.*

## Updating PyTorch

When a stable PyTorch release supports your GPU:

```bash
# Uninstall current version
pip uninstall torch torchvision torchaudio

# Install stable version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
- [PyTorch Nightly Builds](https://download.pytorch.org/whl/nightly/)

## Questions?

- Check NVIDIA driver version: `nvidia-smi`
- Check CUDA version: `nvcc --version` (if CUDA toolkit installed)
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"`

---

**Last Updated**: January 2026 (includes RTX 50xx series support)
