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
import torch
import numpy as np
import wave
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import argparse
import json

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    tqdm = None

try:
    from pygame import mixer  # type: ignore
    mixer.init()
    def playsound(filepath: str):
        """Play audio using pygame."""
        mixer.music.load(filepath)
        mixer.music.play()
        while mixer.music.get_busy():
            import time
            time.sleep(0.1)
except (ImportError, Exception):
    try:
        import winsound  # type: ignore
        def playsound(filepath: str):
            """Play audio using Windows built-in winsound."""
            winsound.PlaySound(filepath, winsound.SND_FILENAME)
    except ImportError:
        print("Warning: No audio playback library available.")
        print("Install with 'pip install pygame' for audio playback.")
        playsound = None

from qwen_tts import Qwen3TTSModel


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


def load_voice_design_clone_profiles(config_path: str = VOICE_DESIGN_CLONE_PROFILES_CONFIG) -> Dict[str, Any]:
    """
    Load voice design + clone profiles from JSON configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary of voice design + clone profiles
    """
    if not os.path.exists(config_path):
        print_progress(f"Error: Voice design + clone profiles config not found: {config_path}")
        print_progress("Creating default config file...")
        
        # Create default config
        default_profiles = {
            "Example": {
                "description": "Example voice design + clone profile",
                "reference": {
                    "text": "This is an example reference text.",
                    "instruct": "Speak in a clear, neutral tone.",
                    "language": "English"
                },
                "single_texts": [
                    "This is the first single clone text.",
                    "This is the second single clone text."
                ],
                "batch_texts": [
                    "This is the first batch clone text.",
                    "This is the second batch clone text."
                ],
                "language": "English"
            }
        }
        
        # Ensure config directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_profiles, f, indent=2, ensure_ascii=False)
        
        print_progress(f"Created default config at: {config_path}")
        print_progress("Please edit this file to add your voice design + clone profiles.")
        return default_profiles
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print_progress(f"Error parsing JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        print_progress(f"Error loading voice design + clone profiles: {e}")
        sys.exit(1)


def print_progress(message: str):
    """Print a progress message with formatting."""
    print(f"[INFO] {message}")


def ensure_output_dir(output_dir: str = "output/Voice_Design_Clone"):
    """
    Ensure the output directory exists.
    
    Args:
        output_dir: Directory path to create
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def save_wav_pygame(filepath: str, audio_data: np.ndarray, sample_rate: int):
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


def load_design_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-VoiceDesign") -> Qwen3TTSModel:
    """
    Load the Qwen3-TTS VoiceDesign model with progress indication.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Loaded Qwen3TTSModel instance
    """
    print_progress("Loading VoiceDesign model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print_progress(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print_progress("Using CPU (CUDA not available)")
    
    print_progress(f"Loading model from {model_path}...")
    if tqdm:
        with tqdm(total=1, desc="Loading VoiceDesign model", unit="step") as pbar:
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map={"": device},
                dtype=torch.bfloat16,
            )
            pbar.update(1)
    else:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map={"": device},
            dtype=torch.bfloat16,
        )
    
    print_progress("VoiceDesign model loaded successfully!")
    
    return model


def load_clone_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-Base") -> Qwen3TTSModel:
    """
    Load the Qwen3-TTS Base (Clone) model with progress indication.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Loaded Qwen3TTSModel instance
    """
    print_progress("Loading Base (Clone) model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print_progress(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print_progress("Using CPU (CUDA not available)")
    
    print_progress(f"Loading model from {model_path}...")
    if tqdm:
        with tqdm(total=1, desc="Loading Clone model", unit="step") as pbar:
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map={"": device},
                dtype=torch.bfloat16,
            )
            pbar.update(1)
    else:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map={"": device},
            dtype=torch.bfloat16,
        )
    
    print_progress("Clone model loaded successfully!")
    
    return model


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
    save_wav_pygame(output_file, wavs[0], sr)
    print_progress(f"Reference audio saved successfully!")
    
    return wavs, sr


def generate_single_clone(
    clone_model: Qwen3TTSModel,
    text: str,
    language: str,
    voice_clone_prompt,
    output_file: str
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
    save_wav_pygame(output_file, wavs[0], sr)
    print_progress(f"Audio saved successfully!")
    
    return wavs, sr


def generate_batch_clone(
    clone_model: Qwen3TTSModel,
    texts: list,
    languages: list,
    voice_clone_prompt,
    output_prefix: str = "output/Voice_Design_Clone/clone_batch"
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
        output_file = f"{output_prefix}_{i}.wav"
        save_wav_pygame(output_file, wav, sr)
        print_progress(f"  Saved: {output_file}")
    
    return wavs, sr


def play_audio(filepath: str):
    """
    Play an audio file using the available audio library.
    
    Args:
        filepath: Path to the audio file
    """
    if not os.path.exists(filepath):
        print_progress(f"Warning: Audio file not found: {filepath}")
        return
    
    if playsound is None:
        print_progress("Warning: No audio playback library available. Cannot play audio.")
        print_progress(f"Audio file saved at: {os.path.abspath(filepath)}")
        return
    
    print_progress(f"Playing audio: {filepath}")
    try:
        playsound(filepath)
        print_progress("Playback completed")
    except Exception as e:
        print_progress(f"Error playing audio: {e}")
        print_progress(f"Audio file saved at: {os.path.abspath(filepath)}")


def parse_args(voice_profiles: Dict[str, Any]):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Design + Clone Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available voice design + clone profiles: {', '.join(voice_profiles.keys())}

Examples:
  python src/voice_design_clone.py                      # Use default settings from config
  python src/voice_design_clone.py --profile Nervous_Teen  # Use specific profile
  python src/voice_design_clone.py --no-batch           # Skip batch generation
  python src/voice_design_clone.py --only-single        # Only run single generation
  python src/voice_design_clone.py --list-voices        # List available voice profiles
        """
    )
    
    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        choices=list(voice_profiles.keys()),
        help=f"Voice design + clone profile to use (default: {DEFAULT_PROFILE})"
    )
    
    parser.add_argument(
        "--no-single",
        action="store_true",
        help="Skip single clone generation"
    )
    
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Skip batch clone generation"
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
        "--no-play",
        action="store_true",
        help="Skip audio playback"
    )
    
    parser.add_argument(
        "--batch-runs",
        type=int,
        default=None,
        help=f"Number of complete runs to generate for comparison (default: {BATCH_RUNS}). Creates run_1/, run_2/, etc. subdirectories"
    )
    
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice profiles and exit"
    )
    
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
    # Load voice design + clone profiles from JSON config
    voice_profiles = load_voice_design_clone_profiles()
    
    # Parse command-line arguments
    args = parse_args(voice_profiles)
    
    # Handle --list-voices
    if args.list_voices:
        list_voice_profiles(voice_profiles)
        return
    
    # Determine what to run
    run_single = RUN_SINGLE and not args.no_single and not args.only_batch
    run_batch = RUN_BATCH and not args.no_batch and not args.only_single
    play_audio_enabled = PLAY_AUDIO and not args.no_play
    batch_runs = args.batch_runs if args.batch_runs is not None else BATCH_RUNS
    
    # Determine which profile to use
    profile_name = args.profile if args.profile else DEFAULT_PROFILE
    
    if profile_name not in voice_profiles:
        print_progress(f"Error: Voice design + clone profile '{profile_name}' not found!")
        print_progress(f"Available profiles: {', '.join(voice_profiles.keys())}")
        sys.exit(1)
    
    profile = voice_profiles[profile_name]
    start_time = time.time()
    
    try:
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
        design_model = load_design_model()
        clone_model = load_clone_model()
        
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
                            output_file=output_file
                        )
                        
                        # Play first single clone (only last run)
                        if play_audio_enabled and i == 1 and run_num == batch_runs:
                            print("\n" + "="*60)
                            play_audio(output_file)
            
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
                        output_prefix=f"{base_output_dir}/{profile_name}_batch"
                    )
                    
                    # Play first batch clone (only last run)
                    if play_audio_enabled and run_num == batch_runs:
                        print("\n" + "="*60)
                        play_audio(f"{base_output_dir}/{profile_name}_batch_0.wav")
            
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
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
