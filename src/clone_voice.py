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
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import io
import wave
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


def load_text_from_file_or_string(value: Union[str, list]) -> Union[str, list]:
    """
    Load text from a file if the value is a file path, otherwise return the value as-is.
    Supports both single strings and lists of strings.
    
    Args:
        value: Either a string (text or file path) or a list of strings
        
    Returns:
        The text content (either from file or original value)
    """
    if isinstance(value, list):
        # Process each item in the list
        return [load_text_from_file_or_string(item) for item in value]
    
    if isinstance(value, str):
        # Check if it looks like a file path and exists
        if os.path.exists(value) and os.path.isfile(value):
            try:
                with open(value, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                print_progress(f"Loaded text from file: {value}")
                return content
            except Exception as e:
                print_progress(f"Warning: Could not read file '{value}': {e}")
                print_progress(f"Using value as literal text instead.")
                return value
        else:
            # Not a file path or doesn't exist, treat as literal text
            return value
    
    return value


def load_voice_profiles(config_path: str = VOICE_PROFILES_CONFIG) -> Dict[str, Any]:
    """
    Load voice profiles from JSON configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary of voice profiles
    """
    if not os.path.exists(config_path):
        print_progress(f"Error: Voice profiles config not found: {config_path}")
        print_progress("Creating default config file...")
        
        # Create default config
        default_profiles = {
            "Example": {
                "voice_sample_file": "./input/example.wav",
                "sample_transcript": "This is an example reference text.",
                "single_text": "This is a single generation example.",
                "batch_texts": [
                    "This is the first batch text.",
                    "This is the second batch text."
                ]
            }
        }
        
        # Ensure config directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_profiles, f, indent=2, ensure_ascii=False)
        
        print_progress(f"Created default config at: {config_path}")
        print_progress("Please edit this file to add your voice profiles.")
        return default_profiles
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        # Process text fields to load from files if specified
        for profile_name, profile in profiles.items():
            if 'single_text' in profile:
                profile['single_text'] = load_text_from_file_or_string(profile['single_text'])
            if 'batch_texts' in profile:
                profile['batch_texts'] = load_text_from_file_or_string(profile['batch_texts'])
            if 'sample_transcript' in profile:
                profile['sample_transcript'] = load_text_from_file_or_string(profile['sample_transcript'])
        
        return profiles
    except json.JSONDecodeError as e:
        print_progress(f"Error parsing JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        print_progress(f"Error loading voice profiles: {e}")
        sys.exit(1)


def print_progress(message: str):
    """Print a progress message with formatting."""
    print(f"[INFO] {message}")


def ensure_output_dir(output_dir: str = "output/Clone_Voice"):
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


def load_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-Base") -> Qwen3TTSModel:
    """
    Load the Qwen3-TTS model with progress indication.
    
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
    
    return model


def generate_voice_clone(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    ref_audio: Union[str, Tuple[np.ndarray, int]],
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    output_file: str = "output/Clone_Voice/output_voice_clone.wav"
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
    save_wav_pygame(output_file, wavs[0], sr)
    print_progress(f"Audio saved successfully!")
    
    return wavs, sr


def generate_batch_voice_clone(
    model: Qwen3TTSModel,
    texts: list,
    languages: list,
    ref_audio: Union[str, Tuple[np.ndarray, int]],
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    output_prefix: str = "output/Clone_Voice/output_voice_clone"
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
        output_file = f"{output_prefix}_{i}.wav"
        save_wav_pygame(output_file, wav, sr)
        print_progress(f"  Saved: {output_file}")
    
    return wavs, sr


def play_audio(filepath: str):
    """
    Play an audio file using playsound.
    
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
        print_progress(f"Error playing audio: {e}")
        print_progress(f"Audio file saved at: {os.path.abspath(filepath)}")


def parse_args(voice_profiles: Dict[str, Any]):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Clone Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available voice profiles: {', '.join(voice_profiles.keys())}

Examples:
  python src/clone_voice.py                               # Use default settings from config
  python src/clone_voice.py --voice DougDoug              # Use DougDoug voice profile
  python src/clone_voice.py --voices DougDoug Grandma     # Process multiple voice profiles
  python src/clone_voice.py --batch-runs 5                # Generate 5 different versions (run_1/, run_2/, etc.)
  python src/clone_voice.py --no-batch                    # Skip batch generation
  python src/clone_voice.py --only-single                 # Only run single generation
  python src/clone_voice.py --compare --only-single       # Compare mode: generate same text as reference
  python src/clone_voice.py --list-voices                 # List available voice profiles
        """
    )
    
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=None,
        choices=list(voice_profiles.keys()),
        help=f"Voice profile to use (default: {DEFAULT_VOICE})"
    )
    
    parser.add_argument(
        "--voices",
        type=str,
        nargs="+",
        choices=list(voice_profiles.keys()),
        help="Multiple voice profiles to process (e.g., --voices DougDoug Grandma)"
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
        "--compare",
        action="store_true",
        help="Compare mode: use sample_transcript as single_text to compare against original audio"
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
    total_runs: int = 1
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
        print_progress(f"Error: Voice profile '{voice_name}' not found!")
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
        ensure_output_dir()
        
        # Check if reference audio exists
        if not os.path.exists(ref_audio):
            print_progress(f"Error: Reference audio not found: {ref_audio}")
            print_progress("Please provide a valid reference audio file.")
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
                output_file=output_single
            )
            
            # Play the generated audio
            if play_audio_enabled:
                print("\n" + "="*60)
                play_audio(output_single)
        
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
                output_prefix=output_batch_prefix
            )
            
            # Play the first batch output
            if play_audio_enabled:
                print("\n" + "="*60)
                play_audio(f"{output_batch_prefix}_1.wav")
        
        duration = time.time() - start_time
        print("\n" + "="*60)
        print_progress(f"Voice '{voice_name}' completed in {duration:.2f} seconds")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error processing voice '{voice_name}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the voice cloning pipeline."""
    # Load voice profiles from JSON config
    voice_profiles = load_voice_profiles()
    
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
    
    try:
        # Load model once for all voices and runs
        print("\n" + "="*60)
        print(f"PROCESSING {len(voice_names)} VOICE PROFILE(S)")
        if batch_runs > 1:
            print(f"WITH {batch_runs} BATCH RUNS")
        print("="*60)
        print_progress(f"Voices to process: {', '.join(voice_names)}")
        if batch_runs > 1:
            print_progress(f"Batch runs: {batch_runs} (outputs will be in run_1/, run_2/, etc.)")
        
        model = load_model()
        
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
                    total_runs=batch_runs
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
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
