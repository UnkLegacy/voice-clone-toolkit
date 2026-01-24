"""
Qwen3-TTS Voice Clone Generation Script

This script demonstrates how to use the Qwen3-TTS Base model to clone voices
and generate speech with the cloned voice characteristics.

Voice Clone Mode:
    To clone a voice, you need to provide:
    - ref_audio: Reference audio clip (local file path, URL, base64, or numpy array)
    - ref_text: Transcript of the reference audio
    
    Optional:
    - x_vector_only_mode: If True, only speaker embedding is used (ref_text not required,
                          but cloning quality may be reduced)
"""

import time
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import argparse

# Import utilities from our new modular structure
try:
    from .import_helper import get_utils
except ImportError:
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
load_voice_clone_profiles = utils.load_voice_clone_profiles
load_voice_clone_model = utils.load_voice_clone_model
create_base_parser = utils.create_base_parser
add_common_args = utils.add_common_args
add_voice_selection_args = utils.add_voice_selection_args
add_multi_voice_selection_args = utils.add_multi_voice_selection_args
get_generation_modes = utils.get_generation_modes
validate_file_exists = utils.validate_file_exists

from qwen_tts import Qwen3TTSModel

# Optional dependency for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# CONFIGURATION SECTION - Edit these to easily switch between different voices
# =============================================================================

# Path to voice profiles configuration file
VOICE_PROFILES_CONFIG = "config/voice_clone_profiles.json"

# Default settings (can be overridden by command-line arguments)
#DEFAULT_VOICE = "Example_Grandma"  # Change this to switch between voices (or use list: ["DougDoug", "Grandma"])
DEFAULT_VOICE = ["DougDoug", "Example_Grandma"]  # Process multiple voices by default
RUN_SINGLE = True                  # Set to False to skip single generation
RUN_BATCH = True                   # Set to False to skip batch generation
PLAY_AUDIO = True                  # Set to False to skip audio playback
COMPARE_MODE = False               # Set to True to use sample_transcript as single_text (for quality comparison)
BATCH_RUNS = 1                     # Number of complete runs to generate (for comparing different AI generations)

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================




def generate_voice_clone(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    ref_audio: Union[str, Tuple[np.ndarray, int]],
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    output_file: str = "output/Clone_Voice/output_voice_clone.wav",
    output_format: str = "wav",
    bitrate: str = "192k"
) -> tuple:
    """
    Generate speech with cloned voice characteristics.
    
    Args:
        model: The loaded Qwen3TTSModel
        text: Text to synthesize
        language: Target language
        ref_audio: Reference audio (file path, URL, base64, or numpy array tuple)
        ref_text: Transcript of reference audio (required unless x_vector_only_mode=True)
        x_vector_only_mode: If True, only use speaker embedding (lower quality)
        output_file: Output filename
        output_format: Output format ("wav" or "mp3")
        bitrate: Bitrate for MP3 encoding (e.g., "192k", "320k")
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating voice clone...")
    print_progress(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print_progress(f"  Language: {language}")
    print_progress(f"  Reference audio: {ref_audio if isinstance(ref_audio, str) else 'numpy array'}")
    if ref_text:
        print_progress(f"  Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")
    print_progress(f"  X-vector only mode: {x_vector_only_mode}")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=1, desc="Generating audio", unit="sample") as pbar:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            pbar.update(1)
    else:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
    
    duration = time.time() - start_time
    print_progress(f"Generation completed in {duration:.2f} seconds")
    
    print_progress(f"Saving to {output_file}...")
    final_path = save_audio(output_file, wavs[0], sr, output_format, bitrate)
    print_progress(f"Audio saved successfully: {final_path}")
    
    return wavs, sr


def generate_batch_voice_clone(
    model: Qwen3TTSModel,
    texts: list,
    languages: list,
    ref_audio: Union[str, Tuple[np.ndarray, int]],
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    output_prefix: str = "output/Clone_Voice/output_voice_clone",
    output_format: str = "wav",
    bitrate: str = "192k"
) -> tuple:
    """
    Generate multiple voice samples with the same cloned voice (efficient reuse of prompt).
    
    Args:
        model: The loaded Qwen3TTSModel
        texts: List of texts to synthesize
        languages: List of target languages
        ref_audio: Reference audio (file path, URL, base64, or numpy array tuple)
        ref_text: Transcript of reference audio
        x_vector_only_mode: If True, only use speaker embedding
        output_prefix: Prefix for output filenames
        output_format: Output format ("wav" or "mp3")
        bitrate: Bitrate for MP3 encoding (e.g., "192k", "320k")
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating batch of {len(texts)} voice clones...")
    print_progress(f"  Reference audio: {ref_audio if isinstance(ref_audio, str) else 'numpy array'}")
    if ref_text:
        print_progress(f"  Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")
    
    start_time = time.time()
    
    # Create voice clone prompt once for efficiency
    print_progress("Creating voice clone prompt (reusable for batch)...")
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )
    
    # Generate batch using the prompt
    if tqdm:
        with tqdm(total=len(texts), desc="Generating batch", unit="sample") as pbar:
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=prompt_items,
            )
            pbar.update(len(texts))
    else:
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=prompt_items,
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
    # Get default voice for help text
    default_voice_str = str(DEFAULT_VOICE[0]) if isinstance(DEFAULT_VOICE, list) else str(DEFAULT_VOICE)
    
    # Create parser with standard structure
    parser = create_base_parser(
        description="Qwen3-TTS Voice Clone Generation Script",
        script_name="src/clone_voice.py",
        available_profiles=voice_profiles
    )
    
    # Add voice selection arguments  
    add_voice_selection_args(
        parser, voice_profiles, default_voice_str,
        arg_name="voice", arg_short="v"
    )
    
    # Add multiple voice selection
    add_multi_voice_selection_args(
        parser, voice_profiles, 
        help_text="Multiple voice profiles to process (e.g., --voices DougDoug Grandma)"
    )
    
    # Add compare mode argument (specific to clone_voice)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare mode: use sample_transcript as single_text to compare against original audio"
    )
    
    # Add all common arguments
    add_common_args(parser, default_batch_runs=BATCH_RUNS, profile_type="voice profiles")
    
    return parser.parse_args()


def list_voice_profiles(voice_profiles: Dict[str, Any]):
    """List all available voice profiles."""
    print("\n" + "="*60)
    print("AVAILABLE VOICE PROFILES")
    print("="*60)
    for name, profile in voice_profiles.items():
        print(f"\n{name}:")
        print(f"  Voice sample file: {profile['voice_sample_file']}")
        print(f"  Sample transcript: {profile['sample_transcript'][:60]}...")
        print(f"  Single text: {profile['single_text'][:60]}...")
        print(f"  Batch texts: {len(profile['batch_texts'])} samples")
    print("="*60 + "\n")


def process_voice_profile(
    voice_name: str,
    voice_profiles: Dict[str, Any],
    model: Qwen3TTSModel,
    run_single: bool,
    run_batch: bool,
    play_audio_enabled: bool,
    compare_mode: bool,
    run_number: Optional[int] = None,
    total_runs: int = 1,
    args = None
):
    """
    Process a single voice profile for generation.
    
    Args:
        voice_name: Name of the voice profile
        voice_profiles: Dictionary of all voice profiles
        model: The loaded Qwen3TTSModel
        run_single: Whether to run single generation
        run_batch: Whether to run batch generation
        play_audio_enabled: Whether to play audio after generation
        compare_mode: Whether to use sample_transcript as single_text
        run_number: Current run number (for batch runs), None for single run
        total_runs: Total number of batch runs
    """
    if voice_name not in voice_profiles:
        print_error(f"Voice profile '{voice_name}' not found!")
        print_progress(f"Available profiles: {', '.join(voice_profiles.keys())}")
        return False
    
    profile = voice_profiles[voice_name]
    ref_audio = profile["voice_sample_file"]
    ref_text = profile["sample_transcript"]
    
    # In compare mode, use sample_transcript as single_text to compare against original
    if compare_mode:
        single_text = profile["sample_transcript"]
    else:
        single_text = profile["single_text"]
    
    batch_texts = profile["batch_texts"]
    
    start_time = time.time()
    
    try:
        print("\n" + "="*60)
        print(f"VOICE CLONING: {voice_name}")
        print("="*60)
        print_progress(f"Reference audio: {ref_audio}")
        print_progress(f"Running single: {run_single}")
        print_progress(f"Running batch: {run_batch}")
        print_progress(f"Audio playback: {play_audio_enabled}")
        print_progress(f"Compare mode: {compare_mode}" + (" (using sample_transcript as single_text)" if compare_mode else ""))
        
        # Ensure output directory exists
        ensure_output_dir("output/Clone_Voice")
        
        # Check if reference audio exists
        if not validate_file_exists(ref_audio, "reference audio"):
            return False
        
        # Extract input filename (without extension) for output naming
        input_name = Path(ref_audio).stem
        
        # Add run subdirectory if doing multiple batch runs
        if run_number is not None:
            base_output_dir = f"output/Clone_Voice/{input_name}/run_{run_number}"
        else:
            base_output_dir = f"output/Clone_Voice/{input_name}"
        
        output_single = f"{base_output_dir}/{input_name}_clone.wav"
        output_batch_prefix = f"{base_output_dir}/{input_name}_clone"
        
        # Single voice clone generation
        if run_single:
            print("\n" + "="*60)
            print("SINGLE VOICE CLONE GENERATION")
            print("="*60)
            wavs, sr = generate_voice_clone(
                model=model,
                text=single_text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False,
                output_file=output_single,
                output_format=args.output_format,
                bitrate=args.bitrate
            )
            
            # Play the generated audio
            if play_audio_enabled:
                print("\n" + "="*60)
                # Construct correct filename with extension
                single_file_with_ext = str(Path(output_single).with_suffix(f'.{args.output_format}'))
                play_audio(single_file_with_ext)
        
        # Batch voice clone generation (reusing prompt)
        if run_batch:
            print("\n" + "="*60)
            print("BATCH VOICE CLONE GENERATION (with prompt reuse)")
            print("="*60)
            
            languages = ["English"] * len(batch_texts)
            
            wavs, sr = generate_batch_voice_clone(
                model=model,
                texts=batch_texts,
                languages=languages,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False,
                output_prefix=output_batch_prefix,
                output_format=args.output_format,
                bitrate=args.bitrate
            )
            
            # Play the first batch output
            if play_audio_enabled:
                print("\n" + "="*60)
                play_audio(f"{output_batch_prefix}_1.{args.output_format}")
        
        duration = time.time() - start_time
        print("\n" + "="*60)
        print_progress(f"Voice '{voice_name}' completed in {duration:.2f} seconds")
        print("="*60)
        return True
        
    except Exception as e:
        return handle_processing_error(e, voice_name)


def main():
    """Main function to run the voice cloning pipeline."""
    try:
        # Load voice profiles from JSON config
        voice_profiles = load_voice_clone_profiles(VOICE_PROFILES_CONFIG)
        
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
        compare_mode = COMPARE_MODE or args.compare
        batch_runs = args.batch_runs if args.batch_runs is not None else BATCH_RUNS
        
        # Determine which voices to process
        if args.voices:
            # Multiple voices specified via --voices
            voice_names = args.voices
        elif args.voice:
            # Single voice specified via --voice
            voice_names = [args.voice]
        else:
            # Use DEFAULT_VOICE from config (can be string or list)
            if isinstance(DEFAULT_VOICE, list):
                voice_names = DEFAULT_VOICE
            else:
                voice_names = [DEFAULT_VOICE]
        
        total_start_time = time.time()
        
        # Load model once for all voices and runs
        print("\n" + "="*60)
        print(f"PROCESSING {len(voice_names)} VOICE PROFILE(S)")
        if batch_runs > 1:
            print(f"WITH {batch_runs} BATCH RUNS")
        print("="*60)
        print_progress(f"Voices to process: {', '.join(voice_names)}")
        if batch_runs > 1:
            print_progress(f"Batch runs: {batch_runs} (outputs will be in run_1/, run_2/, etc.)")
        
        model = load_voice_clone_model()
        
        # Process batch runs
        total_success_count = 0
        for run_num in range(1, batch_runs + 1):
            if batch_runs > 1:
                print("\n" + "="*80)
                print(f"BATCH RUN {run_num}/{batch_runs}")
                print("="*80)
            
            # Process each voice
            success_count = 0
            for i, voice_name in enumerate(voice_names, 1):
                print("\n" + "="*80)
                if batch_runs > 1:
                    print(f"RUN {run_num}/{batch_runs} - VOICE {i}/{len(voice_names)}: {voice_name}")
                else:
                    print(f"PROCESSING VOICE {i}/{len(voice_names)}: {voice_name}")
                print("="*80)
                
                success = process_voice_profile(
                    voice_name=voice_name,
                    voice_profiles=voice_profiles,
                    model=model,
                    run_single=run_single,
                    run_batch=run_batch,
                    play_audio_enabled=play_audio_enabled,
                    compare_mode=compare_mode,
                    run_number=run_num if batch_runs > 1 else None,
                    total_runs=batch_runs,
                    args=args
                )
                
                if success:
                    success_count += 1
            
            total_success_count += success_count
            
            if batch_runs > 1:
                print("\n" + "-"*80)
                print(f"Run {run_num} Summary: {success_count}/{len(voice_names)} voices successful")
                print("-"*80)
        
        # Print summary
        total_duration = time.time() - total_start_time
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        if batch_runs > 1:
            print_progress(f"Total batch runs: {batch_runs}")
            print_progress(f"Voices per run: {len(voice_names)}")
            print_progress(f"Total generations: {total_success_count}/{batch_runs * len(voice_names)}")
        else:
            print_progress(f"Successfully processed: {total_success_count}/{len(voice_names)} voices")
        print_progress(f"Total execution time: {total_duration:.2f} seconds")
        print("="*80)
        
    except Exception as e:
        handle_fatal_error(e, "running voice cloning pipeline")


if __name__ == "__main__":
    main()
