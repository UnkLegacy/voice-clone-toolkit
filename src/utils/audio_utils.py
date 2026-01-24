"""
Audio processing and playback utilities.

This module provides audio-related utilities for saving, converting, and playing
audio files across all voice generation scripts.
"""

import os
import wave
import time
from pathlib import Path
from typing import Optional, Union, Callable
import numpy as np

from .progress import print_progress, handle_mp3_conversion_error, handle_audio_playback_error

# Optional dependency handling
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Audio playback handling with fallbacks
playsound: Optional[Callable[[str], None]] = None

try:
    from pygame import mixer  # type: ignore
    mixer.init()
    def playsound_pygame(filepath: str):
        """Play audio using pygame."""
        mixer.music.load(filepath)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    playsound = playsound_pygame
except (ImportError, Exception):
    try:
        import winsound  # type: ignore
        def playsound_winsound(filepath: str):
            """Play audio using Windows built-in winsound."""
            winsound.PlaySound(filepath, winsound.SND_FILENAME)
        playsound = playsound_winsound
    except ImportError:
        print("Warning: No audio playback library available.")
        print("Install with 'pip install pygame' for audio playback.")
        playsound = None


def ensure_output_dir(output_dir: str = "output") -> None:
    """
    Ensure the output directory exists.
    
    Args:
        output_dir: Directory path to create
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def normalize_audio(audio_data: np.ndarray, target_level: float = 0.95, 
                   method: str = "peak") -> np.ndarray:
    """
    Normalize audio data to a target level.
    
    This is useful for balancing volume levels across different voices or audio clips.
    Each voice/clip is normalized independently to the same target level (e.g., 95% peak),
    so all voices end up balanced against each other. For example:
    - If DougDoug and Grandma are quiet, they'll be boosted to 95% peak
    - If Sohee is loud, she'll be reduced to 95% peak
    - All voices end up at the same level, balanced against each other
    
    Args:
        audio_data: Audio data as numpy array (int16 or float)
        target_level: Target peak level (0.0 to 1.0 for float, or 0 to 32767 for int16)
                     Default 0.95 leaves some headroom to prevent clipping
        method: Normalization method - "peak" (default) or "rms"
                - "peak": Normalizes based on peak amplitude (prevents clipping)
                - "rms": Normalizes based on RMS (root mean square) for perceived loudness
    
    Returns:
        np.ndarray: Normalized audio data (same dtype as input)
    
    Examples:
        # Normalize quiet audio to match other voices
        normalized = normalize_audio(quiet_audio, target_level=0.95)
        
        # Use RMS normalization for better perceived loudness matching
        normalized = normalize_audio(quiet_audio, target_level=0.7, method="rms")
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Convert to float for processing
    is_int16 = audio_data.dtype == np.int16
    if is_int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Calculate current level
    if method == "peak":
        current_level = np.max(np.abs(audio_float))
    elif method == "rms":
        current_level = np.sqrt(np.mean(audio_float ** 2))
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'peak' or 'rms'")
    
    # Avoid division by zero (silent audio)
    if current_level < 1e-6:
        return audio_data
    
    # Calculate gain factor
    gain = target_level / current_level
    
    # Apply gain
    normalized_float = audio_float * gain
    
    # Prevent clipping
    normalized_float = np.clip(normalized_float, -1.0, 1.0)
    
    # Convert back to original dtype
    if is_int16:
        normalized = (normalized_float * 32767).astype(np.int16)
    else:
        normalized = normalized_float
    
    return normalized


def adjust_volume(audio_data: np.ndarray, volume_factor: float) -> np.ndarray:
    """
    Adjust audio volume by a multiplication factor.
    
    Args:
        audio_data: Audio data as numpy array (int16 or float)
        volume_factor: Volume adjustment factor
                      - 1.0 = no change
                      - 2.0 = double volume (+6 dB)
                      - 0.5 = half volume (-6 dB)
                      - 0.0 = silence
    
    Returns:
        np.ndarray: Volume-adjusted audio data (same dtype as input)
    
    Examples:
        # Boost DougDoug voice by 50%
        boosted = adjust_volume(dougdoug_audio, volume_factor=1.5)
        
        # Reduce Grandma voice by 20%
        reduced = adjust_volume(grandma_audio, volume_factor=0.8)
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Convert to float for processing
    is_int16 = audio_data.dtype == np.int16
    if is_int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Apply volume adjustment
    adjusted_float = audio_float * volume_factor
    
    # Prevent clipping
    adjusted_float = np.clip(adjusted_float, -1.0, 1.0)
    
    # Convert back to original dtype
    if is_int16:
        adjusted = (adjusted_float * 32767).astype(np.int16)
    else:
        adjusted = adjusted_float
    
    return adjusted


def save_wav(filepath: str, audio_data: np.ndarray, sample_rate: int) -> None:
    """
    Save audio data to WAV file using wave module (no soundfile dependency).
    
    Args:
        filepath: Output file path
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure audio_data is in the correct format
    if audio_data.dtype != np.int16:
        # Convert float to int16
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)
    
    # Write WAV file
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def save_audio(filepath: str, audio_data: np.ndarray, sample_rate: int, 
               output_format: str = "wav", bitrate: str = "192k",
               normalize: bool = False, target_level: float = 0.95,
               volume_adjust: Optional[float] = None) -> str:
    """
    Save audio data to file in specified format (WAV or MP3).
    
    Args:
        filepath: Output file path (extension will be changed to match format)
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_format: Output format ("wav" or "mp3")
        bitrate: Bitrate for MP3 encoding (e.g., "192k", "320k")
        normalize: If True, normalize audio to target_level before saving
                   Useful for balancing volume levels across different voices
        target_level: Target peak level for normalization (0.0 to 1.0)
                     Default 0.95 leaves headroom to prevent clipping
        volume_adjust: Optional volume adjustment factor (e.g., 1.5 for +50%, 0.8 for -20%)
                      Applied before normalization if normalize is True
    
    Returns:
        str: Final output filepath (with correct extension)
    
    Examples:
        # Save with volume normalization (balances quiet voices)
        save_audio("output.wav", audio, sr, normalize=True)
        
        # Save with manual volume boost for quiet voices
        save_audio("output.wav", audio, sr, volume_adjust=1.5)
        
        # Combine both: boost then normalize
        save_audio("output.wav", audio, sr, normalize=True, volume_adjust=1.3)
    """
    # Apply volume adjustment if specified
    if volume_adjust is not None:
        audio_data = adjust_volume(audio_data, volume_adjust)
    
    # Apply normalization if requested
    if normalize:
        audio_data = normalize_audio(audio_data, target_level=target_level)
    
    # Always save as WAV first
    wav_path = str(Path(filepath).with_suffix('.wav'))
    save_wav(wav_path, audio_data, sample_rate)
    
    # If MP3 requested, convert
    if output_format.lower() == "mp3":
        if not PYDUB_AVAILABLE:
            print_progress("Warning: pydub not available. Saving as WAV instead.")
            print_progress("Install pydub with: pip install pydub")
            print_progress("Also requires ffmpeg to be installed on your system.")
            return wav_path
        
        mp3_path = str(Path(filepath).with_suffix('.mp3'))
        try:
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3", bitrate=bitrate)
            # Delete WAV file after successful MP3 conversion
            os.remove(wav_path)
            return mp3_path
        except Exception as e:
            return handle_mp3_conversion_error(e, wav_path)
    
    return wav_path


def play_audio(filepath: str) -> None:
    """
    Play an audio file using available audio playback library.
    
    Args:
        filepath: Path to the audio file
    """
    if not os.path.exists(filepath):
        print_progress(f"Warning: Audio file not found: {filepath}")
        return
    
    if playsound is None:
        print_progress("Warning: playsound not available. Cannot play audio.")
        print_progress(f"Audio file saved at: {os.path.abspath(filepath)}")
        return
    
    print_progress(f"Playing audio: {filepath}")
    try:
        playsound(filepath)
        print_progress("Playback completed")
    except Exception as e:
        handle_audio_playback_error(e, os.path.abspath(filepath))


def get_audio_info(filepath: str) -> dict:
    """
    Get basic information about an audio file.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        dict: Audio information including format, duration, etc.
    """
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    
    try:
        # Try to get info using wave module for WAV files
        if filepath.lower().endswith('.wav'):
            with wave.open(filepath, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                return {
                    "format": "WAV",
                    "sample_rate": sample_rate,
                    "frames": frames,
                    "duration": duration,
                    "channels": wav_file.getnchannels(),
                    "bit_depth": wav_file.getsampwidth() * 8
                }
        
        # Try using pydub for other formats
        elif PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(filepath)
            return {
                "format": filepath.split('.')[-1].upper(),
                "sample_rate": audio.frame_rate,
                "duration": len(audio) / 1000.0,  # Convert ms to seconds
                "channels": audio.channels,
                "bit_depth": audio.sample_width * 8
            }
        else:
            return {"error": "Cannot read audio format (pydub not available)"}
            
    except Exception as e:
        return {"error": f"Failed to read audio file: {e}"}