"""
Qwen3-TTS Custom Voice Generation Script

This script demonstrates how to use the Qwen3-TTS CustomVoice model to generate
speech with different speakers and languages.

Speaker profiles are configured in config/custom_voice_profiles.json
Use --list-speakers to see all available speakers and their descriptions.
"""

import time
import sys
import os
import torch
import numpy as np
import wave
from pathlib import Path
from typing import Optional, Dict, Any, Union
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
# CONFIGURATION SECTION - Edit these to easily switch between different speakers
# =============================================================================

# Path to custom voice profiles configuration file
CUSTOM_VOICE_PROFILES_CONFIG = "config/custom_voice_profiles.json"

# Default settings (can be overridden by command-line arguments)
DEFAULT_SPEAKER = "Ryan"  # Change this to switch between speakers
RUN_SINGLE = True          # Set to False to skip single generation
RUN_BATCH = True           # Set to False to skip batch generation
PLAY_AUDIO = True          # Set to False to skip audio playback

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================


def load_custom_voice_profiles(config_path: str = CUSTOM_VOICE_PROFILES_CONFIG) -> Dict[str, Any]:
    """
    Load custom voice profiles from JSON configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary of custom voice profiles
    """
    if not os.path.exists(config_path):
        print_progress(f"Error: Custom voice profiles config not found: {config_path}")
        print_progress("Creating default config file...")
        
        # Create default config
        default_profiles = {
            "Ryan": {
                "speaker": "Ryan",
                "language": "English",
                "description": "Dynamic male voice with strong rhythmic drive",
                "single_text": "Hello everyone, this is a test of the custom voice system.",
                "single_instruct": "",
                "batch_texts": [
                    "This is the first example.",
                    "This is the second example."
                ],
                "batch_languages": ["English", "English"],
                "batch_instructs": ["", "Happy"]
            }
        }
        
        # Ensure config directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_profiles, f, indent=2, ensure_ascii=False)
        
        print_progress(f"Created default config at: {config_path}")
        print_progress("Please edit this file to add your custom voice profiles.")
        return default_profiles
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print_progress(f"Error parsing JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        print_progress(f"Error loading custom voice profiles: {e}")
        sys.exit(1)


def print_progress(message: str):
    """Print a progress message with formatting."""
    print(f"[INFO] {message}")


def ensure_output_dir(output_dir: str = "output/Custom_Voice"):
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


def load_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-CustomVoice") -> Qwen3TTSModel:
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
    print_progress(f"Supported speakers: {model.get_supported_speakers()}")
    print_progress(f"Supported languages: {model.get_supported_languages()}")
    
    return model


def generate_single_voice(
    model: Qwen3TTSModel,
    text: str,
    language: str = "English",
    speaker: str = "Vivian",
    instruct: Optional[str] = None,
    output_file: str = "output/Custom_Voice/output_custom_voice.wav"
) -> tuple:
    """
    Generate a single voice sample.
    
    Args:
        model: The loaded Qwen3TTSModel
        text: Text to synthesize
        language: Target language
        speaker: Speaker name
        instruct: Optional instruction for tone/style
        output_file: Output filename
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating single voice sample...")
    print_progress(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print_progress(f"  Speaker: {speaker}, Language: {language}")
    if instruct:
        print_progress(f"  Instruction: {instruct}")
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=1, desc="Generating audio", unit="sample") as pbar:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or "",
            )
            pbar.update(1)
    else:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct or "",
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
    speakers: list,
    instructs: Optional[list] = None,
    output_prefix: str = "output/Custom_Voice/output_custom_voice"
) -> tuple:
    """
    Generate multiple voice samples in batch.
    
    Args:
        model: The loaded Qwen3TTSModel
        texts: List of texts to synthesize
        languages: List of target languages
        speakers: List of speaker names
        instructs: Optional list of instructions
        output_prefix: Prefix for output filenames
        
    Returns:
        Tuple of (wavs, sample_rate)
    """
    print_progress(f"Generating batch of {len(texts)} voice samples...")
    
    if instructs is None:
        instructs = [""] * len(texts)
    
    start_time = time.time()
    
    if tqdm:
        with tqdm(total=len(texts), desc="Generating batch", unit="sample") as pbar:
            wavs, sr = model.generate_custom_voice(
                text=texts,
                language=languages,
                speaker=speakers,
                instruct=instructs
            )
            pbar.update(len(texts))
    else:
        wavs, sr = model.generate_custom_voice(
            text=texts,
            language=languages,
            speaker=speakers,
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
        description="Qwen3-TTS Custom Voice Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available voice profiles: {', '.join(voice_profiles.keys())}

Examples:
  python src/custom_voice.py                        # Use default settings from config
  python src/custom_voice.py --speaker Ryan         # Use Ryan speaker profile
  python src/custom_voice.py --no-batch             # Skip batch generation
  python src/custom_voice.py --only-single          # Only run single generation
  python src/custom_voice.py --list-speakers        # List available speaker profiles
        """
    )
    
    parser.add_argument(
        "--speaker", "-s",
        type=str,
        default=None,
        choices=list(voice_profiles.keys()),
        help=f"Speaker profile to use (default: {DEFAULT_SPEAKER})"
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
        "--list-speakers",
        action="store_true",
        help="List available speaker profiles and exit"
    )
    
    return parser.parse_args()


def list_speaker_profiles(voice_profiles: Dict[str, Any]):
    """List all available speaker profiles."""
    print("\n" + "="*60)
    print("AVAILABLE SPEAKER PROFILES")
    print("="*60)
    for name, profile in voice_profiles.items():
        print(f"\n{name}:")
        print(f"  Speaker: {profile['speaker']}")
        print(f"  Language: {profile['language']}")
        print(f"  Description: {profile.get('description', 'N/A')}")
        single_text = profile['single_text']
        # Truncate safely for display
        display_text = single_text[:60] if len(single_text) <= 60 else single_text[:60] + "..."
        try:
            print(f"  Single text: {display_text}")
        except UnicodeEncodeError:
            print(f"  Single text: [contains non-ASCII characters]")
        print(f"  Batch texts: {len(profile.get('batch_texts', []))} samples")
    print("="*60 + "\n")


def main():
    """Main function to run the TTS generation pipeline."""
    # Load custom voice profiles from JSON config
    voice_profiles = load_custom_voice_profiles()
    
    # Parse command-line arguments
    args = parse_args(voice_profiles)
    
    # Handle --list-speakers
    if args.list_speakers:
        list_speaker_profiles(voice_profiles)
        return
    
    # Determine what to run
    run_single = RUN_SINGLE and not args.no_single and not args.only_batch
    run_batch = RUN_BATCH and not args.no_batch and not args.only_single
    play_audio_enabled = PLAY_AUDIO and not args.no_play
    
    # Determine which speaker to use
    speaker_name = args.speaker if args.speaker else DEFAULT_SPEAKER
    
    if speaker_name not in voice_profiles:
        print_progress(f"Error: Speaker profile '{speaker_name}' not found!")
        print_progress(f"Available profiles: {', '.join(voice_profiles.keys())}")
        sys.exit(1)
    
    profile = voice_profiles[speaker_name]
    start_time = time.time()
    
    try:
        print("\n" + "="*60)
        print(f"CUSTOM VOICE GENERATION: {speaker_name}")
        print("="*60)
        print_progress(f"Speaker: {profile['speaker']}")
        print_progress(f"Language: {profile['language']}")
        print_progress(f"Description: {profile.get('description', 'N/A')}")
        print_progress(f"Running single: {run_single}")
        print_progress(f"Running batch: {run_batch}")
        print_progress(f"Audio playback: {play_audio_enabled}")
        
        # Ensure output directory exists
        ensure_output_dir()
        
        # Load model
        model = load_model()
        
        # Output paths
        output_single = f"output/Custom_Voice/{speaker_name}_single.wav"
        output_batch_prefix = f"output/Custom_Voice/{speaker_name}_batch"
        
        # Single voice generation
        if run_single:
            print("\n" + "="*60)
            print("SINGLE VOICE GENERATION")
            print("="*60)
            wavs, sr = generate_single_voice(
                model=model,
                text=profile['single_text'],
                language=profile['language'],
                speaker=profile['speaker'],
                instruct=profile.get('single_instruct', ''),
                output_file=output_single
            )
            
            # Play the generated audio
            if play_audio_enabled:
                print("\n" + "="*60)
                play_audio(output_single)
        
        # Batch voice generation
        if run_batch:
            batch_texts = profile.get('batch_texts', [])
            if batch_texts:
                print("\n" + "="*60)
                print("BATCH VOICE GENERATION")
                print("="*60)
                
                batch_languages = profile.get('batch_languages', [profile['language']] * len(batch_texts))
                batch_instructs = profile.get('batch_instructs', [''] * len(batch_texts))
                batch_speakers = [profile['speaker']] * len(batch_texts)
                
                wavs, sr = generate_batch_voices(
                    model=model,
                    texts=batch_texts,
                    languages=batch_languages,
                    speakers=batch_speakers,
                    instructs=batch_instructs,
                    output_prefix=output_batch_prefix
                )
                
                # Play the first batch output
                if play_audio_enabled:
                    print("\n" + "="*60)
                    play_audio(f"{output_batch_prefix}_1.wav")
        
        total_duration = time.time() - start_time
        print("\n" + "="*60)
        print_progress(f"Total execution time: {total_duration:.2f} seconds")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()