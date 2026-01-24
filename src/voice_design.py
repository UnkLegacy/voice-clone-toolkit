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
import torch
import numpy as np
import wave
from pathlib import Path
from typing import Optional, Dict, Any
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
except ImportError:
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


def load_voice_design_profiles(config_path: str = VOICE_DESIGN_PROFILES_CONFIG) -> Dict[str, Any]:
    """
    Load voice design profiles from JSON configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary of voice design profiles
    """
    if not os.path.exists(config_path):
        print_progress(f"Error: Voice design profiles config not found: {config_path}")
        print_progress("Creating default config file...")
        
        # Create default config
        default_profiles = {
            "Example": {
                "language": "English",
                "description": "Example voice design profile",
                "single_text": "This is an example of voice design.",
                "single_instruct": "Speak in a clear, neutral tone.",
                "batch_texts": [
                    "This is the first example.",
                    "This is the second example."
                ],
                "batch_languages": ["English", "English"],
                "batch_instructs": ["", "Happy tone"]
            }
        }
        
        # Ensure config directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_profiles, f, indent=2, ensure_ascii=False)
        
        print_progress(f"Created default config at: {config_path}")
        print_progress("Please edit this file to add your voice design profiles.")
        return default_profiles
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print_progress(f"Error parsing JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        print_progress(f"Error loading voice design profiles: {e}")
        sys.exit(1)


def print_progress(message: str):
    """Print a progress message with formatting."""
    print(f"[INFO] {message}")


def ensure_output_dir(output_dir: str = "output/Voice_Design"):
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


def load_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-VoiceDesign") -> Qwen3TTSModel:
    """
    Load the Qwen3-TTS VoiceDesign model with progress indication.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Loaded Qwen3TTSModel instance
    """
    print_progress("Checking CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print_progress(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print_progress("Using CPU (CUDA not available)")
    
    print_progress(f"Loading model from {model_path}...")
    if tqdm:
        # Create a simple progress indicator for model loading
        with tqdm(total=1, desc="Loading model", unit="step") as pbar:
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
    
    print_progress("Model loaded successfully!")
    print_progress(f"Supported languages: {model.get_supported_languages()}")
    
    return model


def generate_single_voice(
    model: Qwen3TTSModel,
    text: str,
    language: str = "Chinese",
    instruct: str = "",
    output_file: str = "output/Voice_Design/output_voice_design.wav"
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
    save_wav_pygame(output_file, wavs[0], sr)
    print_progress(f"Audio saved successfully!")
    
    return wavs, sr


def generate_batch_voices(
    model: Qwen3TTSModel,
    texts: list,
    languages: list,
    instructs: list,
    output_prefix: str = "output/Voice_Design/output_voice_design"
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
        description="Qwen3-TTS Voice Design Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available voice design profiles: {', '.join(voice_profiles.keys())}

Examples:
  python src/voice_design.py                              # Use default settings from config
  python src/voice_design.py --profile Incredulous_Panic  # Use specific profile
  python src/voice_design.py --no-batch                   # Skip batch generation
  python src/voice_design.py --only-single                # Only run single generation
  python src/voice_design.py --list-profiles              # List available profiles
        """
    )
    
    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        choices=list(voice_profiles.keys()),
        help=f"Voice design profile to use (default: {DEFAULT_PROFILE})"
    )
    
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
        "--list-profiles",
        action="store_true",
        help="List available voice design profiles and exit"
    )
    
    return parser.parse_args()


def list_design_profiles(voice_profiles: Dict[str, Any]):
    """List all available voice design profiles."""
    print("\n" + "="*60)
    print("AVAILABLE VOICE DESIGN PROFILES")
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
    # Load voice design profiles from JSON config
    voice_profiles = load_voice_design_profiles()
    
    # Parse command-line arguments
    args = parse_args(voice_profiles)
    
    # Handle --list-profiles
    if args.list_profiles:
        list_design_profiles(voice_profiles)
        return
    
    # Determine what to run
    run_single = RUN_SINGLE and not args.no_single and not args.only_batch
    run_batch = RUN_BATCH and not args.no_batch and not args.only_single
    play_audio_enabled = PLAY_AUDIO and not args.no_play
    batch_runs = args.batch_runs if args.batch_runs is not None else BATCH_RUNS
    
    # Determine which profile to use
    profile_name = args.profile if args.profile else DEFAULT_PROFILE
    
    if profile_name not in voice_profiles:
        print_progress(f"Error: Voice design profile '{profile_name}' not found!")
        print_progress(f"Available profiles: {', '.join(voice_profiles.keys())}")
        sys.exit(1)
    
    profile = voice_profiles[profile_name]
    start_time = time.time()
    
    try:
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
        model = load_model()
        
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
                    output_file=output_single
                )
                
                # Play the generated audio (only last run)
                if play_audio_enabled and run_num == batch_runs:
                    print("\n" + "="*60)
                    play_audio(output_single)
            
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
                        output_prefix=output_batch_prefix
                    )
                    
                    # Play the first batch output (only last run)
                    if play_audio_enabled and run_num == batch_runs:
                        print("\n" + "="*60)
                        play_audio(f"{output_batch_prefix}_1.wav")
            
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
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()