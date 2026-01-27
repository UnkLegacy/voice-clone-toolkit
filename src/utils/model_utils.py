"""
Model loading and management utilities.

This module provides unified model loading functionality with device management
and progress reporting for all voice generation scripts.
"""

import torch
from typing import Optional, Dict, Any

from .progress import print_progress, print_error

# Optional dependency handling
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    tqdm = None

# Import the Qwen3TTSModel
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: qwen_tts not installed. Install with 'pip install qwen-tts'")
    Qwen3TTSModel = None


def get_device() -> torch.device:
    """
    Get the best available device (CUDA or CPU).
    
    Returns:
        torch.device: The device to use for model loading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print_progress(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print_progress("Using CPU (CUDA not available)")
    
    return device


def fix_pad_token_id(model: 'Qwen3TTSModel') -> None:
    """
    Fix pad_token_id warning by setting it explicitly in generation_config.
    
    This prevents the warning: "Setting `pad_token_id` to `eos_token_id` for open-end generation."
    
    Args:
        model: The loaded Qwen3TTSModel instance
    """
    try:
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            # Try to get pad_token_id from tokenizer first
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                if hasattr(model.tokenizer, 'pad_token_id') and model.tokenizer.pad_token_id is not None:
                    model.generation_config.pad_token_id = model.tokenizer.pad_token_id
                    return
            
            # Fallback to eos_token_id if tokenizer doesn't have pad_token_id
            if hasattr(model.generation_config, 'eos_token_id') and model.generation_config.eos_token_id is not None:
                model.generation_config.pad_token_id = model.generation_config.eos_token_id
    except Exception:
        # If setting pad_token_id fails, continue anyway (non-critical)
        pass


def load_model_with_device(model_path: str, 
                          model_name: Optional[str] = None,
                          dtype: torch.dtype = torch.bfloat16,
                          device_map: Optional[Dict[str, Any]] = None,
                          show_progress: bool = True) -> 'Qwen3TTSModel':
    """
    Load a Qwen3-TTS model with automatic device detection and progress indication.
    
    Args:
        model_path: Path to the model directory
        model_name: Optional name for progress messages (e.g., "VoiceDesign", "CustomVoice")
        dtype: Data type for model loading (default: torch.bfloat16)
        device_map: Custom device mapping (default: auto-detected)
        show_progress: Whether to show progress messages
        
    Returns:
        Loaded Qwen3TTSModel instance
        
    Raises:
        ImportError: If qwen_tts is not installed
        Exception: If model loading fails
    """
    if Qwen3TTSModel is None:
        raise ImportError("qwen_tts not installed. Install with 'pip install qwen-tts'")
    
    if show_progress:
        print_progress("Checking CUDA availability...")
    
    device = get_device()
    
    # Use default device mapping if none provided
    if device_map is None:
        device_map = {"": device}
    
    # Create descriptive progress message
    model_desc = f"{model_name} model" if model_name else "model"
    
    if show_progress:
        print_progress(f"Loading {model_desc} from {model_path}...")
    
    try:
        if tqdm and show_progress:
            # Create a simple progress indicator for model loading
            desc = f"Loading {model_name} model" if model_name else "Loading model"
            with tqdm(total=1, desc=desc, unit="step") as pbar:
                model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=device_map,
                    dtype=dtype,
                )
                pbar.update(1)
        else:
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=dtype,
            )
        
        # Fix pad_token_id warning by setting it explicitly
        fix_pad_token_id(model)
        
        if show_progress:
            print_progress(f"{model_desc.capitalize()} loaded successfully!")
        
        return model
    
    except Exception as e:
        print_error(f"Error loading {model_desc}: {e}")
        raise


def load_voice_clone_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-Base") -> 'Qwen3TTSModel':
    """
    Load the voice cloning model with standard settings.
    
    Args:
        model_path: Path to the voice clone model directory
        
    Returns:
        Loaded Qwen3TTSModel instance for voice cloning
    """
    return load_model_with_device(model_path, model_name="Voice Clone")


def load_custom_voice_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-CustomVoice") -> 'Qwen3TTSModel':
    """
    Load the custom voice model with standard settings.
    
    Args:
        model_path: Path to the custom voice model directory
        
    Returns:
        Loaded Qwen3TTSModel instance for custom voice generation
    """
    return load_model_with_device(model_path, model_name="CustomVoice")


def load_voice_design_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-VoiceDesign") -> 'Qwen3TTSModel':
    """
    Load the voice design model with standard settings.
    
    Args:
        model_path: Path to the voice design model directory
        
    Returns:
        Loaded Qwen3TTSModel instance for voice design generation
    """
    return load_model_with_device(model_path, model_name="VoiceDesign")


def get_model_memory_usage() -> Dict[str, Any]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {"device": "cpu", "memory_info": "N/A (using CPU)"}
    
    device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3    # Convert to GB
    memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    return {
        "device": f"cuda:{device}",
        "device_name": torch.cuda.get_device_name(device),
        "memory_allocated_gb": round(memory_allocated, 2),
        "memory_reserved_gb": round(memory_reserved, 2),
        "memory_total_gb": round(memory_total, 2),
        "memory_free_gb": round(memory_total - memory_reserved, 2)
    }


def clear_gpu_cache():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_progress("GPU memory cache cleared")


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
        })
        info.update(get_model_memory_usage())
    else:
        info["device"] = "cpu"
    
    return info


def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path exists and contains expected files.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        True if path appears to be a valid model directory
    """
    import os
    from pathlib import Path
    
    path = Path(model_path)
    
    if not path.exists():
        print_progress(f"Model path does not exist: {model_path}")
        return False
    
    if not path.is_dir():
        print_progress(f"Model path is not a directory: {model_path}")
        return False
    
    # Check for common model files (adjust based on Qwen3-TTS structure)
    expected_files = ["config.json", "pytorch_model.bin"]
    missing_files = []
    
    for file in expected_files:
        if not (path / file).exists():
            # Check for alternative file patterns
            if file == "pytorch_model.bin":
                # Check for alternative PyTorch model file patterns
                model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
                if not model_files:
                    missing_files.append(file)
            else:
                missing_files.append(file)
    
    if missing_files:
        print_progress(f"Warning: Model directory missing expected files: {missing_files}")
        print_progress("This may still be a valid model directory.")
    
    return True  # Return True even with warnings, let the model loading handle validation