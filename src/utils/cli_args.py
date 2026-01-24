"""
Command-line argument parsing utilities.

This module provides shared argument parsing functionality used across all
voice generation scripts, reducing code duplication and ensuring consistency.
"""

import argparse
from typing import Dict, Any, List, Optional


def create_base_parser(description: str, script_name: str = None, 
                      available_profiles: Optional[Dict[str, Any]] = None,
                      examples: Optional[List[str]] = None) -> argparse.ArgumentParser:
    """
    Create a base argument parser with standard formatting.
    
    Args:
        description: Description for the script
        script_name: Name of the script (for examples)
        available_profiles: Dictionary of available profiles for epilog
        examples: List of example command lines
        
    Returns:
        argparse.ArgumentParser: Configured parser
    """
    # Build epilog with available profiles and examples
    epilog_parts = []
    
    if available_profiles:
        profile_names = ', '.join(available_profiles.keys())
        epilog_parts.append(f"Available profiles: {profile_names}")
        epilog_parts.append("")  # Empty line
    
    if examples:
        epilog_parts.append("Examples:")
        epilog_parts.extend(examples)
    
    epilog = "\n".join(epilog_parts) if epilog_parts else None
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    return parser


def add_audio_format_args(parser: argparse.ArgumentParser) -> None:
    """
    Add audio format and bitrate arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["wav", "mp3"],
        default="wav",
        help="Output audio format (default: wav). MP3 requires pydub and ffmpeg."
    )
    
    parser.add_argument(
        "--bitrate",
        type=str,
        default="192k",
        help="Bitrate for MP3 encoding (default: 192k). Examples: 128k, 192k, 320k"
    )


def add_generation_control_args(parser: argparse.ArgumentParser, default_batch_runs: int = 1) -> None:
    """
    Add generation control arguments (single/batch, runs, etc.).
    
    Args:
        parser: ArgumentParser to add arguments to
        default_batch_runs: Default number of batch runs
    """
    parser.add_argument(
        "--no-single",
        action="store_true",
        help="Skip single voice generation"
    )
    
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Skip batch voice generation"
    )
    
    parser.add_argument(
        "--only-single",
        action="store_true",
        help="Only run single generation (skip batch)"
    )
    
    parser.add_argument(
        "--only-batch",
        action="store_true",
        help="Only run batch generation (skip single)"
    )
    
    parser.add_argument(
        "--batch-runs",
        type=int,
        default=None,
        help=f"Number of complete runs to generate for comparison (default: {default_batch_runs}). Creates run_1/, run_2/, etc. subdirectories"
    )


def add_playback_args(parser: argparse.ArgumentParser) -> None:
    """
    Add audio playback control arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip audio playback"
    )


def add_profile_listing_args(parser: argparse.ArgumentParser, 
                           profile_type: str = "voice profiles") -> None:
    """
    Add profile listing argument.
    
    Args:
        parser: ArgumentParser to add arguments to
        profile_type: Type of profiles (e.g., "voice profiles", "speakers")
    """
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help=f"List available {profile_type} and exit"
    )


def add_voice_selection_args(parser: argparse.ArgumentParser, 
                           profiles: Dict[str, Any],
                           default_voice: str,
                           arg_name: str = "voice",
                           arg_short: str = "v",
                           help_text: Optional[str] = None) -> None:
    """
    Add voice/speaker/profile selection arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
        profiles: Dictionary of available profiles
        default_voice: Default profile name
        arg_name: Name of the argument (e.g., "voice", "speaker", "profile")
        arg_short: Short form of the argument
        help_text: Custom help text, or None for default
    """
    if help_text is None:
        help_text = f"{arg_name.title()} profile to use (default: {default_voice})"
    
    parser.add_argument(
        f"--{arg_name}", f"-{arg_short}",
        type=str,
        default=None,
        choices=list(profiles.keys()),
        help=help_text
    )


def add_multi_voice_selection_args(parser: argparse.ArgumentParser,
                                 profiles: Dict[str, Any],
                                 arg_name: str = "voices",
                                 help_text: Optional[str] = None) -> None:
    """
    Add multiple voice selection arguments (like --voices).
    
    Args:
        parser: ArgumentParser to add arguments to
        profiles: Dictionary of available profiles
        arg_name: Name of the argument (default: "voices")
        help_text: Custom help text, or None for default
    """
    if help_text is None:
        help_text = f"Multiple {arg_name} to process"
        
    parser.add_argument(
        f"--{arg_name}",
        type=str,
        nargs="+",
        choices=list(profiles.keys()),
        help=help_text
    )


def add_common_args(parser: argparse.ArgumentParser, 
                   default_batch_runs: int = 1,
                   profile_type: str = "voice profiles") -> None:
    """
    Add all common arguments used across most scripts.
    
    Args:
        parser: ArgumentParser to add arguments to
        default_batch_runs: Default number of batch runs
        profile_type: Type of profiles for help text
    """
    add_audio_format_args(parser)
    add_generation_control_args(parser, default_batch_runs)
    add_playback_args(parser)
    add_profile_listing_args(parser, profile_type)


def validate_generation_args(args: argparse.Namespace) -> None:
    """
    Validate generation control arguments for conflicts.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        argparse.ArgumentError: If conflicting arguments are provided
    """
    conflicts = [
        (args.no_single and args.only_single, "--no-single and --only-single"),
        (args.no_batch and args.only_batch, "--no-batch and --only-batch"),
        (args.only_single and args.only_batch, "--only-single and --only-batch")
    ]
    
    for conflict, names in conflicts:
        if conflict:
            raise argparse.ArgumentError(None, f"Cannot use {names} together")


def get_generation_modes(args: argparse.Namespace) -> tuple[bool, bool]:
    """
    Determine which generation modes to run based on arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Returns:
        tuple[bool, bool]: (run_single, run_batch)
    """
    # Default: run both
    run_single = True
    run_batch = True
    
    # Apply overrides
    if args.only_single:
        run_batch = False
    elif args.only_batch:
        run_single = False
    elif args.no_single:
        run_single = False
    elif args.no_batch:
        run_batch = False
    
    return run_single, run_batch


def create_standard_parser(description: str, 
                         script_name: str,
                         profiles: Dict[str, Any],
                         default_profile: str,
                         default_batch_runs: int = 1,
                         profile_arg_name: str = "voice",
                         profile_arg_short: str = "v",
                         profile_type: str = "voice profiles",
                         additional_examples: Optional[List[str]] = None) -> argparse.ArgumentParser:
    """
    Create a fully configured parser with all standard arguments.
    
    Args:
        description: Script description
        script_name: Script filename for examples
        profiles: Available profiles dictionary
        default_profile: Default profile name
        default_batch_runs: Default batch runs
        profile_arg_name: Name for profile argument
        profile_arg_short: Short form for profile argument
        profile_type: Type of profiles for help text
        additional_examples: Additional example commands
        
    Returns:
        argparse.ArgumentParser: Fully configured parser
    """
    # Standard examples
    examples = [
        f"  python {script_name}                                    # Use default settings from config",
        f"  python {script_name} --{profile_arg_name} {default_profile}                        # Use specific profile",
        f"  python {script_name} --batch-runs 5                         # Generate 5 different versions",
        f"  python {script_name} --no-batch                             # Skip batch generation",
        f"  python {script_name} --only-single                          # Only run single generation",
        f"  python {script_name} --list-voices                          # List available profiles",
        f"  python {script_name} --output-format mp3                    # Save as MP3 instead of WAV",
        f"  python {script_name} --output-format mp3 --bitrate 320k     # MP3 with high bitrate",
    ]
    
    if additional_examples:
        examples.extend(additional_examples)
    
    # Create parser with examples
    parser = create_base_parser(description, script_name, profiles, examples)
    
    # Add profile selection
    add_voice_selection_args(
        parser, profiles, default_profile, 
        profile_arg_name, profile_arg_short
    )
    
    # Add all common arguments
    add_common_args(parser, default_batch_runs, profile_type)
    
    return parser