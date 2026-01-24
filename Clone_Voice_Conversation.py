"""
Qwen3-TTS Voice Clone Conversation Script Generator

This script enables you to create conversations between multiple cloned voices.
It combines the voice cloning capability with conversation scripting to generate
realistic multi-character dialogues.

Features:
- Load multiple actor voice profiles
- Parse conversation scripts with [Actor] dialogue format
- Generate audio for each line in order
- Optional audio concatenation for full conversation playback
- Support for both inline scripts and script files
"""

import time
import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import io
import wave
import argparse
import json
import re

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
# CONFIGURATION SECTION
# =============================================================================

VOICE_PROFILES_CONFIG = "config/voice_clone_profiles.json"
CONVERSATION_SCRIPTS_CONFIG = "config/conversation_scripts.json"

# Default settings
DEFAULT_SCRIPT = "script_file_example"  # Which script to use by default
PLAY_AUDIO = True                        # Set to False to skip audio playback
CONCATENATE_AUDIO = True                 # Set to False to keep lines separate only

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================


def print_progress(message: str):
    """Print a progress message with formatting."""
    print(f"[INFO] {message}")


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        config_path: Path to the JSON config file
        
    Returns:
        Dictionary of configuration data
    """
    if not os.path.exists(config_path):
        print_progress(f"Error: Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print_progress(f"Error parsing JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        print_progress(f"Error loading config: {e}")
        sys.exit(1)


def load_text_from_file_or_string(value: Union[str, list]) -> Union[str, list]:
    """
    Load text from a file if the value is a file path, otherwise return the value as-is.
    
    Args:
        value: Either a string (text or file path) or a list of strings
        
    Returns:
        The text content (either from file or original value)
    """
    if isinstance(value, list):
        return [load_text_from_file_or_string(item) for item in value]
    
    if isinstance(value, str):
        if os.path.exists(value) and os.path.isfile(value):
            try:
                with open(value, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                print_progress(f"Loaded text from file: {value}")
                return content
            except Exception as e:
                print_progress(f"Warning: Could not read file '{value}': {e}")
                return value
        else:
            return value
    
    return value


def parse_script_format(script_text: str) -> List[Tuple[str, str]]:
    """
    Parse script text with [Actor] dialogue format into a list of (actor, line) tuples.
    
    Args:
        script_text: Script text with lines in format "[Actor] dialogue text"
        
    Returns:
        List of (actor_name, dialogue_text) tuples
    """
    lines = []
    pattern = r'\[([^\]]+)\]\s*(.+)'
    
    for line in script_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            actor = match.group(1).strip()
            dialogue = match.group(2).strip()
            lines.append((actor, dialogue))
        else:
            print_progress(f"Warning: Skipping malformed line: {line[:50]}...")
    
    return lines


def parse_script_list(script_list: List[str]) -> List[Tuple[str, str]]:
    """
    Parse script from a list format with [Actor] dialogue.
    
    Args:
        script_list: List of strings in format "[Actor] dialogue text"
        
    Returns:
        List of (actor_name, dialogue_text) tuples
    """
    lines = []
    pattern = r'\[([^\]]+)\]\s*(.+)'
    
    for line in script_list:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            actor = match.group(1).strip()
            dialogue = match.group(2).strip()
            lines.append((actor, dialogue))
        else:
            print_progress(f"Warning: Skipping malformed line: {line[:50]}...")
    
    return lines


def ensure_output_dir(output_dir: str = "output/Conversations"):
    """Ensure the output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def save_wav(filepath: str, audio_data: np.ndarray, sample_rate: int):
    """
    Save audio data to WAV file.
    
    Args:
        filepath: Output file path
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if audio_data.dtype != np.int16:
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def concatenate_audio_files(audio_files: List[str], output_file: str, sample_rate: int, silence_duration: float = 0.3):
    """
    Concatenate multiple audio files with optional silence between them.
    
    Args:
        audio_files: List of audio file paths to concatenate
        output_file: Output file path for concatenated audio
        sample_rate: Sample rate in Hz
        silence_duration: Duration of silence between audio clips in seconds
    """
    print_progress(f"Concatenating {len(audio_files)} audio files...")
    
    silence_samples = int(silence_duration * sample_rate)
    silence = np.zeros(silence_samples, dtype=np.int16)
    
    concatenated = []
    for i, audio_file in enumerate(audio_files):
        with wave.open(audio_file, 'rb') as wav:
            audio_data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            concatenated.append(audio_data)
            
            # Add silence between clips (but not after the last one)
            if i < len(audio_files) - 1:
                concatenated.append(silence)
    
    full_audio = np.concatenate(concatenated)
    save_wav(output_file, full_audio, sample_rate)
    print_progress(f"Concatenated audio saved to: {output_file}")


def load_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-Base") -> Qwen3TTSModel:
    """Load the Qwen3-TTS model."""
    print_progress("Checking CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print_progress(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print_progress("Using CPU (CUDA not available)")
    
    print_progress(f"Loading model from {model_path}...")
    if tqdm:
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


def create_actor_prompts(
    model: Qwen3TTSModel,
    actors: List[str],
    voice_profiles: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create voice clone prompts for all actors in the conversation.
    
    Args:
        model: The loaded Qwen3TTSModel
        actors: List of actor names to prepare
        voice_profiles: Dictionary of voice profiles
        
    Returns:
        Dictionary mapping actor names to their voice clone prompts
    """
    print_progress(f"Creating voice prompts for {len(actors)} actors...")
    actor_prompts = {}
    
    for actor in actors:
        if actor not in voice_profiles:
            print_progress(f"Warning: Actor '{actor}' not found in voice profiles!")
            continue
        
        profile = voice_profiles[actor]
        ref_audio = load_text_from_file_or_string(profile["voice_sample_file"])
        ref_text = load_text_from_file_or_string(profile["sample_transcript"])
        
        if not os.path.exists(ref_audio):
            print_progress(f"Warning: Reference audio not found for '{actor}': {ref_audio}")
            continue
        
        print_progress(f"Creating prompt for {actor}...")
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        actor_prompts[actor] = prompt
        print_progress(f"  ✓ {actor} prompt created")
    
    return actor_prompts


def generate_conversation(
    model: Qwen3TTSModel,
    script_lines: List[Tuple[str, str]],
    actor_prompts: Dict[str, Any],
    output_dir: str,
    conversation_name: str
) -> List[str]:
    """
    Generate audio for each line in the conversation script.
    
    Args:
        model: The loaded Qwen3TTSModel
        script_lines: List of (actor, dialogue) tuples
        actor_prompts: Dictionary of actor voice prompts
        output_dir: Output directory for audio files
        conversation_name: Name of the conversation (for file naming)
        
    Returns:
        List of generated audio file paths
    """
    print_progress(f"Generating conversation with {len(script_lines)} lines...")
    audio_files = []
    
    for i, (actor, dialogue) in enumerate(script_lines, 1):
        if actor not in actor_prompts:
            print_progress(f"Warning: Skipping line {i} - actor '{actor}' has no voice prompt")
            continue
        
        print_progress(f"Line {i}/{len(script_lines)}: [{actor}] {dialogue[:50]}{'...' if len(dialogue) > 50 else ''}")
        
        try:
            # Generate audio for this line
            if tqdm:
                with tqdm(total=1, desc=f"Generating line {i}", unit="sample", leave=False) as pbar:
                    wavs, sr = model.generate_voice_clone(
                        text=dialogue,
                        language="English",
                        voice_clone_prompt=actor_prompts[actor],
                    )
                    pbar.update(1)
            else:
                wavs, sr = model.generate_voice_clone(
                    text=dialogue,
                    language="English",
                    voice_clone_prompt=actor_prompts[actor],
                )
            
            # Save the line
            output_file = f"{output_dir}/{conversation_name}_line_{i:03d}_{actor}.wav"
            save_wav(output_file, wavs[0], sr)
            audio_files.append(output_file)
            print_progress(f"  ✓ Saved: {output_file}")
            
        except Exception as e:
            print_progress(f"Error generating line {i}: {e}")
            continue
    
    return audio_files


def play_audio(filepath: str):
    """Play an audio file."""
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Clone Conversation Script Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Clone_Voice_Conversation.py                                # Use default script
  python Clone_Voice_Conversation.py --script example_conversation  # Use specific script
  python Clone_Voice_Conversation.py --list-scripts                 # List available scripts
  python Clone_Voice_Conversation.py --no-play                      # Skip audio playback
  python Clone_Voice_Conversation.py --no-concatenate               # Keep lines separate only
        """
    )
    
    parser.add_argument(
        "--script", "-s",
        type=str,
        default=None,
        help=f"Conversation script to use (default: {DEFAULT_SCRIPT})"
    )
    
    parser.add_argument(
        "--list-scripts",
        action="store_true",
        help="List available conversation scripts and exit"
    )
    
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip audio playback"
    )
    
    parser.add_argument(
        "--no-concatenate",
        action="store_true",
        help="Don't create concatenated audio file (keep lines separate only)"
    )
    
    return parser.parse_args()


def list_scripts(conversation_scripts: Dict[str, Any]):
    """List all available conversation scripts."""
    print("\n" + "="*60)
    print("AVAILABLE CONVERSATION SCRIPTS")
    print("="*60)
    for name, script_data in conversation_scripts.items():
        print(f"\n{name}:")
        print(f"  Actors: {', '.join(script_data.get('actors', []))}")
        script = script_data.get('script', [])
        if isinstance(script, str):
            lines_count = len([l for l in script.split('\n') if l.strip()])
        else:
            lines_count = len(script)
        print(f"  Lines: {lines_count}")
    print("="*60 + "\n")


def main():
    """Main function to run the conversation generation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configurations
    voice_profiles = load_json_config(VOICE_PROFILES_CONFIG)
    conversation_scripts = load_json_config(CONVERSATION_SCRIPTS_CONFIG)
    
    # Handle --list-scripts
    if args.list_scripts:
        list_scripts(conversation_scripts)
        return
    
    # Determine settings
    script_name = args.script if args.script else DEFAULT_SCRIPT
    play_audio_enabled = PLAY_AUDIO and not args.no_play
    concatenate_enabled = CONCATENATE_AUDIO and not args.no_concatenate
    
    if script_name not in conversation_scripts:
        print_progress(f"Error: Script '{script_name}' not found!")
        print_progress(f"Available scripts: {', '.join(conversation_scripts.keys())}")
        sys.exit(1)
    
    script_data = conversation_scripts[script_name]
    actors = script_data.get("actors", [])
    script_content = script_data.get("script", [])
    
    # Load script content from file if needed
    script_content = load_text_from_file_or_string(script_content)
    
    start_time = time.time()
    
    try:
        print("\n" + "="*60)
        print(f"CONVERSATION SCRIPT: {script_name}")
        print("="*60)
        print_progress(f"Actors: {', '.join(actors)}")
        print_progress(f"Audio playback: {play_audio_enabled}")
        print_progress(f"Concatenate audio: {concatenate_enabled}")
        
        # Parse script
        if isinstance(script_content, str):
            script_lines = parse_script_format(script_content)
        else:
            script_lines = parse_script_list(script_content)
        
        print_progress(f"Parsed {len(script_lines)} dialogue lines")
        
        # Ensure output directory
        output_dir = f"output/Conversations/{script_name}"
        ensure_output_dir(output_dir)
        
        # Load model
        model = load_model()
        
        # Create voice prompts for all actors
        print("\n" + "="*60)
        print("PREPARING ACTOR VOICES")
        print("="*60)
        actor_prompts = create_actor_prompts(model, actors, voice_profiles)
        
        if not actor_prompts:
            print_progress("Error: No valid actor prompts created!")
            sys.exit(1)
        
        # Generate conversation
        print("\n" + "="*60)
        print("GENERATING CONVERSATION")
        print("="*60)
        audio_files = generate_conversation(
            model=model,
            script_lines=script_lines,
            actor_prompts=actor_prompts,
            output_dir=output_dir,
            conversation_name=script_name
        )
        
        if not audio_files:
            print_progress("Error: No audio files generated!")
            sys.exit(1)
        
        # Concatenate audio if requested
        concatenated_file = None
        if concatenate_enabled:
            print("\n" + "="*60)
            print("CONCATENATING AUDIO")
            print("="*60)
            concatenated_file = f"{output_dir}/{script_name}_full.wav"
            
            # Get sample rate from first audio file
            with wave.open(audio_files[0], 'rb') as wav:
                sample_rate = wav.getframerate()
            
            concatenate_audio_files(audio_files, concatenated_file, sample_rate)
        
        # Summary
        duration = time.time() - start_time
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print_progress(f"Generated {len(audio_files)} audio lines")
        print_progress(f"Output directory: {output_dir}")
        print_progress(f"Total execution time: {duration:.2f} seconds")
        
        # Play concatenated audio if available
        if concatenated_file and play_audio_enabled:
            print("\n" + "="*60)
            print("PLAYING FULL CONVERSATION")
            print("="*60)
            play_audio(concatenated_file)
        
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
