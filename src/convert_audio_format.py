"""
Audio Format Conversion Utility

This utility converts audio files between different formats (WAV, MP3, etc.).
Can be used as a standalone script or imported as a module.

Usage as standalone script:
    # Convert single file
    python src/convert_audio_format.py input.wav output.mp3
    
    # Convert all WAVs in a directory to MP3
    python src/convert_audio_format.py output/Conversations/ --format mp3 --recursive
    
    # Convert and delete originals
    python src/convert_audio_format.py output/ --format mp3 --recursive --delete-original

Usage as module:
    from convert_audio_format import convert_audio_file, convert_directory
    
    convert_audio_file("input.wav", "output.mp3")
    convert_directory("output/Conversations", "mp3", recursive=True)
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List
import sys

# Import utilities from our new modular structure
try:
    from .import_helper import get_utils
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from import_helper import get_utils

# Single import - no duplicates!
utils = get_utils()

# Access utility functions
print_progress = utils.print_progress
print_error = utils.print_error

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Install with 'pip install pydub'")
    print("Note: pydub requires ffmpeg to be installed on your system.")
    print("  Windows: Download from https://ffmpeg.org/download.html")
    print("  Linux: sudo apt install ffmpeg")
    print("  Mac: brew install ffmpeg")


def print_info(message: str):
    """Print an info message."""
    print_progress(message)


def convert_audio_file(
    input_path: str,
    output_path: Optional[str] = None,
    output_format: str = "mp3",
    bitrate: str = "192k",
    delete_original: bool = False
) -> bool:
    """
    Convert an audio file to a different format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output file (if None, replaces extension of input)
        output_format: Output format (mp3, wav, ogg, flac, etc.)
        bitrate: Bitrate for compressed formats (e.g., "128k", "192k", "320k")
        delete_original: If True, delete original file after successful conversion
        
    Returns:
        True if conversion successful, False otherwise
    """
    if not PYDUB_AVAILABLE:
        print_error("pydub is not available. Cannot convert audio files.")
        return False
    
    if not os.path.exists(input_path):
        print_error(f"Input file not found: {input_path}")
        return False
    
    # Determine output path
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(f".{output_format}"))
    
    try:
        # Load audio file
        print_info(f"Converting: {input_path}")
        audio = AudioSegment.from_file(input_path)
        
        # Export to new format
        export_params = {"format": output_format}
        
        # Add bitrate for compressed formats
        if output_format in ["mp3", "ogg"]:
            export_params["bitrate"] = bitrate
        
        audio.export(output_path, **export_params)
        
        # Get file sizes for comparison
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
        
        print_info(f"  [OK] Converted: {output_path}")
        print_info(f"  Size: {input_size:.2f}MB -> {output_size:.2f}MB ({compression_ratio:.1f}% smaller)")
        
        # Delete original if requested
        if delete_original and input_path != output_path:
            os.remove(input_path)
            print_info(f"  Deleted original: {input_path}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to convert {input_path}: {e}")
        return False


def convert_directory(
    directory: str,
    output_format: str = "mp3",
    input_format: str = "wav",
    bitrate: str = "192k",
    recursive: bool = False,
    delete_original: bool = False
) -> tuple[int, int]:
    """
    Convert all audio files in a directory to a different format.
    
    Args:
        directory: Path to directory containing audio files
        output_format: Output format (mp3, wav, ogg, flac, etc.)
        input_format: Input format to search for (wav, mp3, etc.)
        bitrate: Bitrate for compressed formats
        recursive: If True, search subdirectories recursively
        delete_original: If True, delete original files after conversion
        
    Returns:
        Tuple of (successful_conversions, failed_conversions)
    """
    if not PYDUB_AVAILABLE:
        print_error("pydub is not available. Cannot convert audio files.")
        return (0, 0)
    
    if not os.path.exists(directory):
        print_error(f"Directory not found: {directory}")
        return (0, 0)
    
    # Find all audio files
    pattern = f"*.{input_format}"
    if recursive:
        files = list(Path(directory).rglob(pattern))
    else:
        files = list(Path(directory).glob(pattern))
    
    if not files:
        print_info(f"No {input_format} files found in {directory}")
        return (0, 0)
    
    print_info(f"Found {len(files)} {input_format} file(s) to convert")
    
    successful = 0
    failed = 0
    
    for file_path in files:
        if convert_audio_file(
            str(file_path),
            output_format=output_format,
            bitrate=bitrate,
            delete_original=delete_original
        ):
            successful += 1
        else:
            failed += 1
    
    print_info(f"Conversion complete: {successful} successful, {failed} failed")
    return (successful, failed)


def main():
    """Main function for standalone script usage."""
    parser = argparse.ArgumentParser(
        description="Convert audio files between different formats (WAV, MP3, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_audio_format.py input.wav output.mp3
  
  # Convert all WAVs in directory to MP3
  python convert_audio_format.py output/Conversations/ --format mp3
  
  # Convert recursively with custom bitrate
  python convert_audio_format.py output/ --format mp3 --recursive --bitrate 320k
  
  # Convert and delete originals
  python convert_audio_format.py output/ --format mp3 --recursive --delete-original
        """
    )
    
    parser.add_argument(
        "input",
        help="Input file or directory"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output file (for single file conversion)"
    )
    parser.add_argument(
        "--format",
        default="mp3",
        choices=["mp3", "wav", "ogg", "flac", "m4a"],
        help="Output format (default: mp3)"
    )
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="Bitrate for compressed formats (default: 192k)"
    )
    parser.add_argument(
        "--input-format",
        default="wav",
        help="Input format to search for in directory mode (default: wav)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively"
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete original files after successful conversion"
    )
    
    args = parser.parse_args()
    
    if not PYDUB_AVAILABLE:
        print_error("Cannot proceed without pydub. Please install:")
        print("  pip install pydub")
        sys.exit(1)
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        # Single file conversion
        if args.output is None:
            # Generate output path from input
            output_path = str(Path(args.input).with_suffix(f".{args.format}"))
        else:
            output_path = args.output
        
        success = convert_audio_file(
            args.input,
            output_path,
            args.format,
            args.bitrate,
            args.delete_original
        )
        sys.exit(0 if success else 1)
        
    elif os.path.isdir(args.input):
        # Directory conversion
        successful, failed = convert_directory(
            args.input,
            args.format,
            args.input_format,
            args.bitrate,
            args.recursive,
            args.delete_original
        )
        sys.exit(0 if failed == 0 else 1)
        
    else:
        print_error(f"Input not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
