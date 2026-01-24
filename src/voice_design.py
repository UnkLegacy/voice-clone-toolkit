"""
Qwen3-TTS Voice Design Generation Script

This script demonstrates how to use the Qwen3-TTS VoiceDesign model to generate
speech with custom voice characteristics described in natural language.

Voice Design allows you to describe the desired voice characteristics (tone,
emotion, style) using natural language instructions, giving you more control
over the generated speech compared to predefined speakers.
"""

import time
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

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
handle_fatal_error = utils.handle_fatal_error
handle_processing_error = utils.handle_processing_error
save_audio = utils.save_audio
play_audio = utils.play_audio
ensure_output_dir = utils.ensure_output_dir
load_voice_design_profiles = utils.load_voice_design_profiles
load_voice_design_model = utils.load_voice_design_model
create_base_parser = utils.create_base_parser
add_common_args = utils.add_common_args
add_voice_selection_args = utils.add_voice_selection_args
get_generation_modes = utils.get_generation_modes
validate_file_exists = utils.validate_file_exists

from qwen_tts import Qwen3TTSModel

# Optional dependency for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# CONFIGURATION SECTION - Edit these to easily switch between different voice designs
# =============================================================================

# Path to voice design profiles configuration file
VOICE_DESIGN_PROFILES_CONFIG = "config/voice_design_profiles.json"

# Default settings (can be overridden by command-line arguments)
DEFAULT_PROFILE = "Professional_Narrator"  # Change this to switch between profiles
RUN_SINGLE = True                           # Set to False to skip single generation
RUN_BATCH = True                            # Set to False to skip batch generation
PLAY_AUDIO = True                           # Set to False to skip audio playback
BATCH_RUNS = 1                              # Number of complete runs to generate (for comparing different AI generations)

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================








def generate_single_voice(
    model: Qwen3TTSModel,
    text: str,
    language: str = "Chinese",
    instruct: str = "",
    output_file: str = "output/Voice_Design/output_voice_design.wav",
    output_format: str = "wav",
    bitrate: str = "192k"
) -> tuple:
    """
    Generate a single voice sample with custom voice design.
    
    Args:
        model: The loaded Qwen3TTSModel
        text: Text to synthesize
        language: Target language
        instruct: Natural language description of desired voice characteristics
        output_file: Output filename
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating single voice sample...")
    print_progress(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print_progress(f"  Language: {language}")
    print_progress(f"  Instruction: {instruct[:60]}{'...' if len(instruct) > 60 else instruct}")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=1, desc="Generating audio", unit="sample") as pbar:
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            pbar.update(1)
    else:
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
    
    duration = time.time() - start_time
    print_progress(f"Generation completed in {duration:.2f} seconds")
    
    print_progress(f"Saving to {output_file}...")
    final_path = save_audio(output_file, wavs[0], sr, output_format, bitrate)
    print_progress(f"Audio saved successfully: {final_path}")
    
    return wavs, sr


def generate_batch_voices(
    model: Qwen3TTSModel,
    texts: list,
    languages: list,
    instructs: list,
    output_prefix: str = "output/Voice_Design/output_voice_design",
    output_format: str = "wav",
    bitrate: str = "192k"
) -> tuple:
    """
    Generate multiple voice samples in batch with custom voice design.
    
    Args:
        model: The loaded Qwen3TTSModel
        texts: List of texts to synthesize
        languages: List of target languages
        instructs: List of natural language descriptions for voice characteristics
        output_prefix: Prefix for output filenames
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating batch of {len(texts)} voice samples...")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=len(texts), desc="Generating batch", unit="sample") as pbar:
            wavs, sr = model.generate_voice_design(
                text=texts,
                language=languages,
                instruct=instructs
            )
            pbar.update(len(texts))
    else:
        wavs, sr = model.generate_voice_design(
            text=texts,
            language=languages,
            instruct=instructs
        )
    
    duration = time.time() - start_time
    print_progress(f"Batch generation completed in {duration:.2f} seconds")
    
    # Save all outputs
    print_progress("Saving audio files...")
    for i, wav in enumerate(wavs, start=1):
        output_file = f"{output_prefix}_{i}"
        final_path = save_audio(output_file, wav, sr, output_format, bitrate)
        print_progress(f"  Saved: {final_path}")
    
    return wavs, sr




def parse_args(voice_profiles: Dict[str, Any]):
    """Parse command-line arguments using shared utilities."""
    # Create parser with standard structure
    parser = create_base_parser(
        description="Qwen3-TTS Voice Design Generation Script",
        script_name="src/voice_design.py",
        available_profiles=voice_profiles
    )
    
    # Add profile selection arguments  
    add_voice_selection_args(
        parser, voice_profiles, DEFAULT_PROFILE,
        arg_name="profile", arg_short="p",
        help_text=f"Voice design profile to use (default: {DEFAULT_PROFILE})"
    )
    
    # Add all common arguments
    add_common_args(parser, default_batch_runs=BATCH_RUNS, profile_type="voice design profiles")
    
    return parser.parse_args()


def list_voice_profiles(voice_profiles: Dict[str, Any]):
    """List all available voice profiles."""
    print("\n" + "="*60)
    print("AVAILABLE VOICE PROFILES")
    print("="*60)
    for name, profile in voice_profiles.items():
        print(f"\n{name}:")
        print(f"  Language: {profile['language']}")
        print(f"  Description: {profile.get('description', 'N/A')}")
        
        # Safely display text fields
        single_text = profile['single_text']
        single_instruct = profile['single_instruct']
        display_text = single_text[:60] if len(single_text) <= 60 else single_text[:60] + "..."
        display_instruct = single_instruct[:60] if len(single_instruct) <= 60 else single_instruct[:60] + "..."
        
        try:
            print(f"  Single text: {display_text}")
            print(f"  Single instruct: {display_instruct}")
        except UnicodeEncodeError:
            print(f"  Single text: [contains non-ASCII characters]")
            print(f"  Single instruct: [contains non-ASCII characters]")
        
        print(f"  Batch texts: {len(profile.get('batch_texts', []))} samples")
    print("="*60 + "\n")


def main():
    """Main function to run the Voice Design TTS generation pipeline."""
    try:
        # Load voice design profiles from JSON config
        voice_profiles = load_voice_design_profiles(VOICE_DESIGN_PROFILES_CONFIG)
        
        # Parse command-line arguments
        args = parse_args(voice_profiles)
        
        # Handle --list-voices
        if args.list_voices:
            list_voice_profiles(voice_profiles)
            return
        
        # Determine what to run using shared utilities
        run_single, run_batch = get_generation_modes(args)
        
        # Apply default overrides
        if not RUN_SINGLE:
            run_single = False
        if not RUN_BATCH:
            run_batch = False
        
        play_audio_enabled = PLAY_AUDIO and not args.no_play
        batch_runs = args.batch_runs if args.batch_runs is not None else BATCH_RUNS
        
        # Determine which profile to use
        profile_name = args.profile if args.profile else DEFAULT_PROFILE
        
        if profile_name not in voice_profiles:
            print_error(f"Voice design profile '{profile_name}' not found!")
            print_progress(f"Available profiles: {', '.join(voice_profiles.keys())}")
            sys.exit(1)
        
        profile = voice_profiles[profile_name]
        start_time = time.time()
        print("\n" + "="*60)
        print(f"VOICE DESIGN GENERATION: {profile_name}")
        if batch_runs > 1:
            print(f"WITH {batch_runs} BATCH RUNS")
        print("="*60)
        print_progress(f"Language: {profile['language']}")
        print_progress(f"Description: {profile.get('description', 'N/A')}")
        print_progress(f"Running single: {run_single}")
        print_progress(f"Running batch: {run_batch}")
        print_progress(f"Audio playback: {play_audio_enabled}")
        if batch_runs > 1:
            print_progress(f"Batch runs: {batch_runs} (outputs will be in run_1/, run_2/, etc.)")
        
        # Load model
        model = load_voice_design_model()
        print_progress(f"Supported languages: {model.get_supported_languages()}")
        
        # Process batch runs
        for run_num in range(1, batch_runs + 1):
            if batch_runs > 1:
                print("\n" + "="*80)
                print(f"BATCH RUN {run_num}/{batch_runs}")
                print("="*80)
            
            # Determine output directory
            if batch_runs > 1:
                base_output_dir = f"output/Voice_Design/run_{run_num}"
            else:
                base_output_dir = "output/Voice_Design"
            ensure_output_dir(base_output_dir)
            
            # Output paths
            output_single = f"{base_output_dir}/{profile_name}_single.wav"
            output_batch_prefix = f"{base_output_dir}/{profile_name}_batch"
            
            # Single voice generation
            if run_single:
                print("\n" + "="*60)
                if batch_runs > 1:
                    print(f"RUN {run_num} - SINGLE VOICE GENERATION")
                else:
                    print("SINGLE VOICE GENERATION")
                print("="*60)
                wavs, sr = generate_single_voice(
                    model=model,
                    text=profile['single_text'],
                    language=profile['language'],
                    instruct=profile['single_instruct'],
                    output_file=output_single,
                    output_format=args.output_format,
                    bitrate=args.bitrate
                )
                
                # Play the generated audio (only last run)
                if play_audio_enabled and run_num == batch_runs:
                    print("\n" + "="*60)
                    single_file_with_ext = str(Path(output_single).with_suffix(f'.{args.output_format}'))
                    play_audio(single_file_with_ext)
            
            # Batch voice generation
            if run_batch:
                batch_texts = profile.get('batch_texts', [])
                if batch_texts:
                    print("\n" + "="*60)
                    if batch_runs > 1:
                        print(f"RUN {run_num} - BATCH VOICE GENERATION")
                    else:
                        print("BATCH VOICE GENERATION")
                    print("="*60)
                    
                    batch_languages = profile.get('batch_languages', [profile['language']] * len(batch_texts))
                    batch_instructs = profile.get('batch_instructs', [''] * len(batch_texts))
                    
                    wavs, sr = generate_batch_voices(
                        model=model,
                        texts=batch_texts,
                        languages=batch_languages,
                        instructs=batch_instructs,
                        output_prefix=output_batch_prefix,
                        output_format=args.output_format,
                        bitrate=args.bitrate
                    )
                    
                    # Play the first batch output (only last run)
                    if play_audio_enabled and run_num == batch_runs:
                        print("\n" + "="*60)
                        play_audio(f"{output_batch_prefix}_1.{args.output_format}")
            
            if batch_runs > 1:
                print("\n" + "-"*80)
                print(f"Run {run_num} Complete")
                print(f"Output: {base_output_dir}")
                print("-"*80)
        
        total_duration = time.time() - start_time
        print("\n" + "="*60)
        if batch_runs > 1:
            print_progress(f"Total batch runs: {batch_runs}")
        print_progress(f"Total execution time: {total_duration:.2f} seconds")
        print("="*60)
        
    except Exception as e:
        handle_fatal_error(e, "running voice design generation")


if __name__ == "__main__":
    main()