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
               output_format: str = "wav", bitrate: str = "192k") -> str:
    """
    Save audio data to file in specified format (WAV or MP3).
    
    Args:
        filepath: Output file path (extension will be changed to match format)
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_format: Output format ("wav" or "mp3")
        bitrate: Bitrate for MP3 encoding (e.g., "192k", "320k")
    
    Returns:
        str: Final output filepath (with correct extension)
    """
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