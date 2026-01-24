"""
Qwen3-TTS Voice Design + Clone Generation Script

This script demonstrates how to combine Voice Design and Voice Clone capabilities:
1. Use VoiceDesign model to create a reference audio with desired characteristics
2. Build a reusable clone prompt from that reference
3. Generate new content with the designed voice consistently

This workflow is ideal when you want a consistent voice across many lines
without re-extracting features every time.
"""

import time
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import argparse

# Import utilities from our new modular structure
# Handle both relative imports (when run as package) and absolute imports (when run directly)
try:
    from .utils.progress import print_progress, print_error, handle_fatal_error, handle_processing_error
    from .utils.audio_utils import save_audio, play_audio, ensure_output_dir
    from .utils.config_loader import load_voice_design_clone_profiles
    from .utils.model_utils import load_voice_design_model, load_voice_clone_model
    from .utils.cli_args import create_base_parser, add_common_args, add_voice_selection_args, get_generation_modes
    from .utils.file_utils import validate_file_exists
except ImportError:
    # Fallback to absolute imports when run directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from utils.progress import print_progress, print_error, handle_fatal_error, handle_processing_error
    from utils.audio_utils import save_audio, play_audio, ensure_output_dir
    from utils.config_loader import load_voice_design_clone_profiles
    from utils.model_utils import load_voice_design_model, load_voice_clone_model
    from utils.cli_args import create_base_parser, add_common_args, add_voice_selection_args, get_generation_modes
    from utils.file_utils import validate_file_exists

from qwen_tts import Qwen3TTSModel

# Optional dependency for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# CONFIGURATION SECTION - Edit these to easily switch between different voice designs
# =============================================================================

# Path to voice design + clone profiles configuration file
VOICE_DESIGN_CLONE_PROFILES_CONFIG = "config/voice_design_clone_profiles.json"

# Default settings (can be overridden by command-line arguments)
DEFAULT_PROFILE = "Nervous_Teen"  # Change this to switch between profiles
RUN_SINGLE = True                  # Set to False to skip single clone generation
RUN_BATCH = True                   # Set to False to skip batch clone generation
PLAY_AUDIO = True                  # Set to False to skip audio playback
BATCH_RUNS = 1                     # Number of complete runs to generate (for comparing different AI generations)

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================






def create_voice_design_reference(
    design_model: Qwen3TTSModel,
    ref_text: str,
    ref_instruct: str,
    language: str = "English",
    output_file: str = "output/Voice_Design_Clone/voice_design_reference.wav"
) -> tuple:
    """
    Create a reference audio using Voice Design model.
    
    Args:
        design_model: The loaded VoiceDesign model
        ref_text: Reference text to synthesize
        ref_instruct: Natural language description of desired voice characteristics
        language: Target language
        output_file: Output filename for reference audio
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress("Creating voice design reference...")
    print_progress(f"  Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")
    print_progress(f"  Language: {language}")
    print_progress(f"  Instruction: {ref_instruct[:60]}{'...' if len(ref_instruct) > 60 else ref_instruct}")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=1, desc="Generating reference", unit="sample") as pbar:
            wavs, sr = design_model.generate_voice_design(
                text=ref_text,
                language=language,
                instruct=ref_instruct,
            )
            pbar.update(1)
    else:
        wavs, sr = design_model.generate_voice_design(
            text=ref_text,
            language=language,
            instruct=ref_instruct,
        )
    
    duration = time.time() - start_time
    print_progress(f"Reference generation completed in {duration:.2f} seconds")
    
    print_progress(f"Saving reference to {output_file}...")
    final_path = save_audio(output_file, wavs[0], sr, "wav", "192k")  # Always save reference as WAV
    print_progress(f"Reference audio saved successfully: {final_path}")
    
    return wavs, sr


def generate_single_clone(
    clone_model: Qwen3TTSModel,
    text: str,
    language: str,
    voice_clone_prompt,
    output_file: str,
    output_format: str = "wav",
    bitrate: str = "192k"
) -> tuple:
    """
    Generate a single audio using the voice clone prompt.
    
    Args:
        clone_model: The loaded Clone model
        text: Text to synthesize
        language: Target language
        voice_clone_prompt: Pre-built voice clone prompt
        output_file: Output filename
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating single clone...")
    print_progress(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print_progress(f"  Language: {language}")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=1, desc="Generating clone", unit="sample") as pbar:
            wavs, sr = clone_model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
            )
            pbar.update(1)
    else:
        wavs, sr = clone_model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
    
    duration = time.time() - start_time
    print_progress(f"Clone generation completed in {duration:.2f} seconds")
    
    print_progress(f"Saving to {output_file}...")
    final_path = save_audio(output_file, wavs[0], sr, output_format, bitrate)
    print_progress(f"Audio saved successfully: {final_path}")
    
    return wavs, sr


def generate_batch_clone(
    clone_model: Qwen3TTSModel,
    texts: list,
    languages: list,
    voice_clone_prompt,
    output_prefix: str = "output/Voice_Design_Clone/clone_batch",
    output_format: str = "wav",
    bitrate: str = "192k"
) -> tuple:
    """
    Generate multiple audio samples using the voice clone prompt (batch mode).
    
    Args:
        clone_model: The loaded Clone model
        texts: List of texts to synthesize
        languages: List of target languages
        voice_clone_prompt: Pre-built voice clone prompt
        output_prefix: Prefix for output filenames
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating batch of {len(texts)} clones...")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=len(texts), desc="Generating batch", unit="sample") as pbar:
            wavs, sr = clone_model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=voice_clone_prompt,
            )
            pbar.update(len(texts))
    else:
        wavs, sr = clone_model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=voice_clone_prompt,
        )
    
    duration = time.time() - start_time
    print_progress(f"Batch generation completed in {duration:.2f} seconds")
    
    # Save all outputs
    print_progress("Saving audio files...")
    for i, wav in enumerate(wavs):
        output_file = f"{output_prefix}_{i}"
        final_path = save_audio(output_file, wav, sr, output_format, bitrate)
        print_progress(f"  Saved: {final_path}")
        print_progress(f"  Saved: {output_file}")
    
    return wavs, sr




def parse_args(voice_profiles: Dict[str, Any]):
    """Parse command-line arguments using shared utilities."""
    # Create parser with standard structure
    parser = create_base_parser(
        description="Qwen3-TTS Voice Design + Clone Generation Script",
        script_name="src/voice_design_clone.py",
        available_profiles=voice_profiles
    )
    
    # Add profile selection arguments  
    add_voice_selection_args(
        parser, voice_profiles, DEFAULT_PROFILE,
        arg_name="profile", arg_short="p",
        help_text=f"Voice design + clone profile to use (default: {DEFAULT_PROFILE})"
    )
    
    # Add all common arguments
    add_common_args(parser, default_batch_runs=BATCH_RUNS, profile_type="voice design + clone profiles")
    
    return parser.parse_args()


def list_voice_profiles(voice_profiles: Dict[str, Any]):
    """List all available voice profiles."""
    print("\n" + "="*60)
    print("AVAILABLE VOICE PROFILES")
    print("="*60)
    for name, profile in voice_profiles.items():
        print(f"\n{name}:")
        print(f"  Description: {profile.get('description', 'N/A')}")
        print(f"  Language: {profile['language']}")
        ref = profile.get('reference', {})
        ref_text = ref.get('text', '')
        display_ref = ref_text[:50] if len(ref_text) <= 50 else ref_text[:50] + "..."
        
        try:
            print(f"  Reference text: {display_ref}")
        except UnicodeEncodeError:
            print(f"  Reference text: [contains non-ASCII characters]")
        
        print(f"  Single texts: {len(profile.get('single_texts', []))} samples")
        print(f"  Batch texts: {len(profile.get('batch_texts', []))} samples")
    print("="*60 + "\n")


def main():
    """Main function to run the Voice Design + Clone pipeline."""
    try:
        # Load voice design + clone profiles from JSON config
        voice_profiles = load_voice_design_clone_profiles(VOICE_DESIGN_CLONE_PROFILES_CONFIG)
        
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
            print_error(f"Voice design + clone profile '{profile_name}' not found!")
            print_progress(f"Available profiles: {', '.join(voice_profiles.keys())}")
            sys.exit(1)
        
        profile = voice_profiles[profile_name]
        start_time = time.time()
        print("\n" + "="*60)
        print(f"VOICE DESIGN + CLONE: {profile_name}")
        if batch_runs > 1:
            print(f"WITH {batch_runs} BATCH RUNS")
        print("="*60)
        print_progress(f"Description: {profile.get('description', 'N/A')}")
        print_progress(f"Language: {profile['language']}")
        print_progress(f"Running single: {run_single}")
        print_progress(f"Running batch: {run_batch}")
        print_progress(f"Audio playback: {play_audio_enabled}")
        if batch_runs > 1:
            print_progress(f"Batch runs: {batch_runs} (outputs will be in run_1/, run_2/, etc.)")
        
        # Load models once
        design_model = load_voice_design_model()
        print_progress(f"VoiceDesign supported languages: {design_model.get_supported_languages()}")
        
        clone_model = load_voice_clone_model()
        print_progress("Voice clone model loaded successfully!")
        
        ref_data = profile['reference']
        ref_text = ref_data['text']
        ref_instruct = ref_data['instruct']
        ref_language = ref_data.get('language', profile['language'])
        
        # Process batch runs
        for run_num in range(1, batch_runs + 1):
            if batch_runs > 1:
                print("\n" + "="*80)
                print(f"BATCH RUN {run_num}/{batch_runs}")
                print("="*80)
            
            # Determine output directory
            if batch_runs > 1:
                base_output_dir = f"output/Voice_Design_Clone/run_{run_num}"
            else:
                base_output_dir = "output/Voice_Design_Clone"
            ensure_output_dir(base_output_dir)
            
            # Step 1: Create voice design reference
            print("\n" + "="*60)
            if batch_runs > 1:
                print(f"RUN {run_num} - STEP 1: CREATE VOICE DESIGN REFERENCE")
            else:
                print("STEP 1: CREATE VOICE DESIGN REFERENCE")
            print("="*60)
            
            ref_output = f"{base_output_dir}/{profile_name}_reference.wav"
            ref_wavs, sr = create_voice_design_reference(
                design_model=design_model,
                ref_text=ref_text,
                ref_instruct=ref_instruct,
                language=ref_language,
                output_file=ref_output
            )
            
            # Play the reference audio (only last run)
            if play_audio_enabled and run_num == batch_runs:
                print("\n" + "="*60)
                print("Playing reference audio...")
                print("="*60)
                play_audio(ref_output)
            
            # Step 2: Build reusable clone prompt
            print("\n" + "="*60)
            if batch_runs > 1:
                print(f"RUN {run_num} - STEP 2: BUILD REUSABLE CLONE PROMPT")
            else:
                print("STEP 2: BUILD REUSABLE CLONE PROMPT")
            print("="*60)
            
            print_progress("Creating voice clone prompt from reference...")
            voice_clone_prompt = clone_model.create_voice_clone_prompt(
                ref_audio=(ref_wavs[0], sr),
                ref_text=ref_text,
            )
            print_progress("Voice clone prompt created successfully!")
            
            # Step 3: Generate single clones using the reusable prompt
            if run_single:
                single_texts = profile.get('single_texts', [])
                if single_texts:
                    print("\n" + "="*60)
                    if batch_runs > 1:
                        print(f"RUN {run_num} - STEP 3: GENERATE SINGLE CLONES (Reusing Prompt)")
                    else:
                        print("STEP 3: GENERATE SINGLE CLONES (Reusing Prompt)")
                    print("="*60)
                    
                    for i, text in enumerate(single_texts, 1):
                        output_file = f"{base_output_dir}/{profile_name}_single_{i}.wav"
                        wavs, sr_clone = generate_single_clone(
                            clone_model=clone_model,
                            text=text,
                            language=profile['language'],
                            voice_clone_prompt=voice_clone_prompt,
                            output_file=output_file,
                            output_format=args.output_format,
                            bitrate=args.bitrate
                        )
                        
                        # Play first single clone (only last run)
                        if play_audio_enabled and i == 1 and run_num == batch_runs:
                            print("\n" + "="*60)
                            single_file_with_ext = str(Path(output_file).with_suffix(f'.{args.output_format}'))
                            play_audio(single_file_with_ext)
            
            # Step 4: Generate batch clones using the reusable prompt
            if run_batch:
                batch_texts = profile.get('batch_texts', [])
                if batch_texts:
                    print("\n" + "="*60)
                    if batch_runs > 1:
                        print(f"RUN {run_num} - STEP 4: GENERATE BATCH CLONES (Reusing Prompt)")
                    else:
                        print("STEP 4: GENERATE BATCH CLONES (Reusing Prompt)")
                    print("="*60)
                    
                    batch_languages = [profile['language']] * len(batch_texts)
                    
                    wavs_batch, sr_batch = generate_batch_clone(
                        clone_model=clone_model,
                        texts=batch_texts,
                        languages=batch_languages,
                        voice_clone_prompt=voice_clone_prompt,
                        output_prefix=f"{base_output_dir}/{profile_name}_batch",
                        output_format=args.output_format,
                        bitrate=args.bitrate
                    )
                    
                    # Play first batch clone (only last run)
                    if play_audio_enabled and run_num == batch_runs:
                        print("\n" + "="*60)
                        play_audio(f"{base_output_dir}/{profile_name}_batch_0.{args.output_format}")
            
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
        print_progress(f"Profile '{profile_name}' completed successfully")
        print("="*60)
        
    except Exception as e:
        handle_fatal_error(e, "running voice design + clone generation")


if __name__ == "__main__":
    main()
