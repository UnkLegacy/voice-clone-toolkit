"""
Progress reporting and error handling utilities.

This module provides standardized progress reporting and error handling
functions used across all voice generation scripts.
"""

import sys
import traceback
from functools import wraps
from typing import Any, Callable, Optional


def print_progress(message: str) -> None:
    """
    Print a progress message with formatting.
    
    Args:
        message: The progress message to display
    """
    print(f"[INFO] {message}")


def print_error(message: str, show_traceback: bool = False) -> None:
    """
    Print an error message with standard formatting.
    
    Args:
        message: The error message to display
        show_traceback: Whether to show the full traceback
    """
    print(f"\n[ERROR] {message}", file=sys.stderr)
    if show_traceback:
        traceback.print_exc()


def print_warning(message: str) -> None:
    """
    Print a warning message with standard formatting.
    
    Args:
        message: The warning message to display
    """
    print_progress(f"Warning: {message}")


def handle_mp3_conversion_error(e: Exception, wav_path: str) -> str:
    """
    Handle MP3 conversion errors with standard message and fallback.
    
    Args:
        e: The exception that occurred during MP3 conversion
        wav_path: Path to the WAV file to fall back to
        
    Returns:
        The WAV file path as fallback
    """
    print_warning(f"MP3 conversion failed ({e}). Keeping WAV file.")
    return wav_path


def handle_audio_playback_error(e: Exception, filepath: str) -> None:
    """
    Handle audio playback errors with standard message.
    
    Args:
        e: The exception that occurred during audio playback
        filepath: Path to the audio file that failed to play
    """
    print_error(f"Error playing audio: {e}")
    print_progress(f"Audio file saved at: {filepath}")


def handle_fatal_error(e: Exception, message: str = "An error occurred") -> None:
    """
    Handle fatal errors with standard formatting and exit.
    
    Args:
        e: The exception that occurred
        message: Custom error message prefix
    """
    print(f"\n[ERROR] {message}: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


def handle_processing_error(e: Exception, item_name: str) -> bool:
    """
    Handle processing errors for individual items (like voices).
    
    Args:
        e: The exception that occurred
        item_name: Name of the item being processed
        
    Returns:
        False to indicate processing failure
    """
    print(f"\n[ERROR] Error processing '{item_name}': {e}", file=sys.stderr)
    traceback.print_exc()
    return False


def with_error_handling(exit_on_error: bool = True):
    """
    Decorator to add standard error handling to functions.
    
    Args:
        exit_on_error: Whether to exit the program on error (default: True)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if exit_on_error:
                    handle_fatal_error(e, f"Error in {func.__name__}")
                else:
                    print_error(f"Error in {func.__name__}: {e}", show_traceback=True)
                    return None
        return wrapper
    return decorator