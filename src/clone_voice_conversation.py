"""
Qwen3-TTS Voice Clone Conversation Script Generator

This script enables you to create conversations between multiple voices using
ANY voice profile type. It combines all voice generation capabilities with
conversation scripting to generate realistic multi-voice dialogues.

Features:
- Support for ALL profile types: voice clones, custom voices, voice designs, and voice design+clone
- Mix different voice types in the same conversation (e.g., DougDoug + Sohee + Incredulous_Panic)
- Parse conversation scripts with [Voice] dialogue format
- Generate audio for each line in order
- Optional audio concatenation for full conversation playback
- Support for both inline scripts and script files

Voice profiles are automatically loaded from:
- config/voice_clone_profiles.json (reference audio + transcript)
- config/custom_voice_profiles.json (pre-built speaker voices)
- config/voice_design_profiles.json (custom voice characteristics)
- config/voice_design_clone_profiles.json (designed voices)
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
# CONFIGURATION SECTION
# =============================================================================

VOICE_CLONE_PROFILES_CONFIG = "config/voice_clone_profiles.json"
CUSTOM_VOICE_PROFILES_CONFIG = "config/custom_voice_profiles.json"
VOICE_DESIGN_PROFILES_CONFIG = "config/voice_design_profiles.json"
VOICE_DESIGN_CLONE_PROFILES_CONFIG = "config/voice_design_clone_profiles.json"
CONVERSATION_SCRIPTS_CONFIG = "config/conversation_scripts.json"

# Default settings
DEFAULT_SCRIPT = "long_script_file_example"  # Which script to use by default
PLAY_AUDIO = True                        # Set to False to skip audio playback
CONCATENATE_AUDIO = True                 # Set to False to keep lines separate only
BATCH_RUNS = 1                           # Number of complete runs to generate (for comparing different AI generations)

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
    Parse script text with [Voice] dialogue format into a list of (voice, line) tuples.
    
    Args:
        script_text: Script text with lines in format "[Voice] dialogue text"
    
    Returns:
        List of (voice_name, dialogue_text) tuples
    """
    lines = []
    pattern = r'\[([^\]]+)\]\s*(.+)'
    
    for line in script_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            voice = match.group(1).strip()
            dialogue = match.group(2).strip()
            lines.append((voice, dialogue))
        else:
            print_progress(f"Warning: Skipping malformed line: {line[:50]}...")
    
    return lines


def parse_script_list(script_list: List[str]) -> List[Tuple[str, str]]:
    """
    Parse script from a list format with [Voice] dialogue.
    
    Args:
        script_list: List of strings in format "[Voice] dialogue text"
    
    Returns:
        List of (voice_name, dialogue_text) tuples
    """
    lines = []
    pattern = r'\[([^\]]+)\]\s*(.+)'
    
    for line in script_list:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            voice = match.group(1).strip()
            dialogue = match.group(2).strip()
            lines.append((voice, dialogue))
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


def load_all_profiles() -> Dict[str, Any]:
    """
    Load all profile types and merge them into a single dictionary.
    Tracks the profile type for each voice.
    
    Returns:
        Dictionary with voice names as keys and profile data (with 'profile_type' field) as values
    """
    all_profiles = {}
    
    # Load voice clone profiles
    if os.path.exists(VOICE_CLONE_PROFILES_CONFIG):
        clone_profiles = load_json_config(VOICE_CLONE_PROFILES_CONFIG)
        for name, profile in clone_profiles.items():
            profile['profile_type'] = 'voice_clone'
            all_profiles[name] = profile
    
    # Load custom voice profiles
    if os.path.exists(CUSTOM_VOICE_PROFILES_CONFIG):
        custom_profiles = load_json_config(CUSTOM_VOICE_PROFILES_CONFIG)
        for name, profile in custom_profiles.items():
            profile['profile_type'] = 'custom_voice'
            all_profiles[name] = profile
    
    # Load voice design profiles
    if os.path.exists(VOICE_DESIGN_PROFILES_CONFIG):
        design_profiles = load_json_config(VOICE_DESIGN_PROFILES_CONFIG)
        for name, profile in design_profiles.items():
            profile['profile_type'] = 'voice_design'
            all_profiles[name] = profile
    
    # Load voice design + clone profiles
    if os.path.exists(VOICE_DESIGN_CLONE_PROFILES_CONFIG):
        design_clone_profiles = load_json_config(VOICE_DESIGN_CLONE_PROFILES_CONFIG)
        for name, profile in design_clone_profiles.items():
            profile['profile_type'] = 'voice_design_clone'
            all_profiles[name] = profile
    
    return all_profiles


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


def create_voice_prompts(
    base_model: Qwen3TTSModel,
    voices: List[str],
    voice_profiles: Dict[str, Any],
    temp_dir: str = "output/Conversations/_temp"
) -> Dict[str, Any]:
    """
    Create voice clone prompts for all voices in the conversation.
    Handles different profile types by converting them to voice clone prompts.
    
    Args:
        base_model: The loaded Base model (for cloning)
        voices: List of voice names to prepare
        voice_profiles: Dictionary of voice profiles (all types)
        temp_dir: Temporary directory for generated reference audio
        
    Returns:
        Dictionary mapping voice names to their voice clone prompts
    """
    print_progress(f"Creating voice prompts for {len(voices)} voices...")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    voice_prompts = {}
    
    for voice in voices:
        if voice not in voice_profiles:
            print_progress(f"Warning: Voice '{voice}' not found in voice profiles!")
            continue
        
        profile = voice_profiles[voice]
        profile_type = profile.get('profile_type', 'voice_clone')
        
        print_progress(f"Creating prompt for {voice} (type: {profile_type})...")
        
        try:
            if profile_type == 'voice_clone':
                # Standard voice clone profile
                ref_audio = profile["voice_sample_file"]  # Audio file path, don't try to read as text
                ref_text = load_text_from_file_or_string(profile["sample_transcript"])
                
                if not os.path.exists(ref_audio):
                    print_progress(f"Warning: Reference audio not found for '{voice}': {ref_audio}")
                    continue
                
                prompt = base_model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                
            elif profile_type == 'custom_voice':
                # Custom voice profile - generate reference audio, then create prompt
                print_progress(f"  Generating custom voice reference for {voice}...")
                custom_model = load_custom_voice_model()
                
                ref_text = profile['single_text']
                ref_audio_path = f"{temp_dir}/{voice}_custom_ref.wav"
                
                wavs, sr = custom_model.generate_custom_voice(
                    text=ref_text,
                    language=profile['language'],
                    speaker=profile['speaker'],
                    instruct=profile.get('single_instruct', ''),
                )
                save_wav(ref_audio_path, wavs[0], sr)
                
                prompt = base_model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                )
                
            elif profile_type == 'voice_design':
                # Voice design profile - generate reference audio, then create prompt
                print_progress(f"  Generating voice design reference for {voice}...")
                design_model = load_voice_design_model()
                
                ref_text = profile['single_text']
                ref_audio_path = f"{temp_dir}/{voice}_design_ref.wav"
                
                wavs, sr = design_model.generate_voice_design(
                    text=ref_text,
                    language=profile['language'],
                    instruct=profile['single_instruct'],
                )
                save_wav(ref_audio_path, wavs[0], sr)
                
                prompt = base_model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                )
                
            elif profile_type == 'voice_design_clone':
                # Voice design + clone profile - generate reference, then create prompt
                print_progress(f"  Generating voice design+clone reference for {voice}...")
                design_model = load_voice_design_model()
                
                ref_data = profile['reference']
                ref_text = ref_data['text']
                ref_audio_path = f"{temp_dir}/{voice}_design_clone_ref.wav"
                
                wavs, sr = design_model.generate_voice_design(
                    text=ref_text,
                    language=ref_data.get('language', profile['language']),
                    instruct=ref_data['instruct'],
                )
                save_wav(ref_audio_path, wavs[0], sr)
                
                prompt = base_model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                )
            else:
                print_progress(f"Warning: Unknown profile type '{profile_type}' for voice '{voice}'")
                continue
            
            voice_prompts[voice] = prompt
            print_progress(f"  ✓ {voice} prompt created")
        
        except Exception as e:
            print_progress(f"Error creating prompt for '{voice}': {e}")
            continue
    
    return voice_prompts


def load_custom_voice_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-CustomVoice") -> Qwen3TTSModel:
    """Load the CustomVoice model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map={"": device},
        dtype=torch.bfloat16,
    )
    return model


def load_voice_design_model(model_path: str = "Qwen_Models/Qwen3-TTS-12Hz-1.7B-VoiceDesign") -> Qwen3TTSModel:
    """Load the VoiceDesign model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map={"": device},
        dtype=torch.bfloat16,
    )
    return model


def generate_conversation(
    model: Qwen3TTSModel,
    script_lines: List[Tuple[str, str]],
    voice_prompts: Dict[str, Any],
    output_dir: str,
    conversation_name: str
) -> List[str]:
    """
    Generate audio for each line in the conversation script.
    
    Args:
        model: The loaded Qwen3TTSModel
        script_lines: List of (voice, dialogue) tuples
        voice_prompts: Dictionary of voice prompts
        output_dir: Output directory for audio files
        conversation_name: Name of the conversation (for file naming)
        
    Returns:
        List of generated audio file paths
    """
    print_progress(f"Generating conversation with {len(script_lines)} lines...")
    audio_files = []
    
    for i, (voice, dialogue) in enumerate(script_lines, 1):
        if voice not in voice_prompts:
            print_progress(f"Warning: Skipping line {i} - voice '{voice}' has no voice prompt")
            continue
        
        print_progress(f"Line {i}/{len(script_lines)}: [{voice}] {dialogue[:50]}{'...' if len(dialogue) > 50 else ''}")
        
        try:
            # Generate audio for this line
            if tqdm:
                with tqdm(total=1, desc=f"Generating line {i}", unit="sample", leave=False, file=sys.stderr) as pbar:
                    wavs, sr = model.generate_voice_clone(
                        text=dialogue,
                        language="English",
                        voice_clone_prompt=voice_prompts[voice],
                    )
                    pbar.update(1)
            else:
                wavs, sr = model.generate_voice_clone(
                    text=dialogue,
                    language="English",
                    voice_clone_prompt=voice_prompts[voice],
                )
            
            # Save the line
            output_file = f"{output_dir}/{conversation_name}_line_{i:03d}_{voice}.wav"
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
  python src/clone_voice_conversation.py                                # Use default script
  python src/clone_voice_conversation.py --script example_conversation  # Use specific script
  python src/clone_voice_conversation.py --batch-runs 3                 # Generate 3 different conversation versions
  python src/clone_voice_conversation.py --list-scripts                 # List available scripts
  python src/clone_voice_conversation.py --list-voices                  # List all available voices
  python src/clone_voice_conversation.py --no-play                      # Skip audio playback
  python src/clone_voice_conversation.py --no-concatenate               # Keep lines separate only
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
        "--list-voices",
        action="store_true",
        help="List all available voice profiles from all types and exit"
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
    
    parser.add_argument(
        "--batch-runs",
        type=int,
        default=None,
        help=f"Number of complete runs to generate for comparison (default: {BATCH_RUNS}). Creates run_1/, run_2/, etc. subdirectories"
    )
    
    return parser.parse_args()


def list_scripts(conversation_scripts: Dict[str, Any]):
    """List all available conversation scripts."""
    print("\n" + "="*60)
    print("AVAILABLE CONVERSATION SCRIPTS")
    print("="*60)
    for name, script_data in conversation_scripts.items():
        print(f"\n{name}:")
        print(f"  Voices: {', '.join(script_data.get('voices', []))}")
        script = script_data.get('script', [])
        if isinstance(script, str):
            lines_count = len([l for l in script.split('\n') if l.strip()])
        else:
            lines_count = len(script)
        print(f"  Lines: {lines_count}")
    print("="*60 + "\n")


def list_all_voices(voice_profiles: Dict[str, Any]):
    """List all available voice profiles from all profile types."""
    print("\n" + "="*60)
    print("AVAILABLE VOICE PROFILES (ALL TYPES)")
    print("="*60)
    
    # Group by profile type
    by_type = {}
    for name, profile in voice_profiles.items():
        profile_type = profile.get('profile_type', 'unknown')
        if profile_type not in by_type:
            by_type[profile_type] = []
        by_type[profile_type].append(name)
    
    # Display by type
    type_names = {
        'voice_clone': 'Voice Clone Profiles',
        'custom_voice': 'Custom Voice Profiles',
        'voice_design': 'Voice Design Profiles',
        'voice_design_clone': 'Voice Design+Clone Profiles'
    }
    
    for ptype, names in sorted(by_type.items()):
        print(f"\n{type_names.get(ptype, ptype.upper())}:")
        for name in sorted(names):
            print(f"  - {name}")
    
    print("\n" + "="*60)
    print(f"Total: {len(voice_profiles)} voice profiles available")
    print("="*60 + "\n")


def main():
    """Main function to run the conversation generation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load all profile types and conversation scripts
    print_progress("Loading all voice profiles...")
    voice_profiles = load_all_profiles()
    print_progress(f"Loaded {len(voice_profiles)} total profiles from all sources")
    
    conversation_scripts = load_json_config(CONVERSATION_SCRIPTS_CONFIG)
    
    # Handle --list-scripts
    if args.list_scripts:
        list_scripts(conversation_scripts)
        return
    
    # Handle --list-voices
    if args.list_voices:
        voice_profiles = load_all_profiles()
        list_all_voices(voice_profiles)
        return
    
    # Determine settings
    script_name = args.script if args.script else DEFAULT_SCRIPT
    play_audio_enabled = PLAY_AUDIO and not args.no_play
    concatenate_enabled = CONCATENATE_AUDIO and not args.no_concatenate
    batch_runs = args.batch_runs if args.batch_runs is not None else BATCH_RUNS
    
    if script_name not in conversation_scripts:
        print_progress(f"Error: Script '{script_name}' not found!")
        print_progress(f"Available scripts: {', '.join(conversation_scripts.keys())}")
        sys.exit(1)
    
    script_data = conversation_scripts[script_name]
    voices = script_data.get("voices", [])
    script_content = script_data.get("script", [])
    
    # Load script content from file if needed
    script_content = load_text_from_file_or_string(script_content)
    
    start_time = time.time()
    
    try:
        print("\n" + "="*60)
        print(f"CONVERSATION SCRIPT: {script_name}")
        if batch_runs > 1:
            print(f"WITH {batch_runs} BATCH RUNS")
        print("="*60)
        print_progress(f"Voices: {', '.join(voices)}")
        print_progress(f"Audio playback: {play_audio_enabled}")
        print_progress(f"Concatenate audio: {concatenate_enabled}")
        if batch_runs > 1:
            print_progress(f"Batch runs: {batch_runs} (outputs will be in run_1/, run_2/, etc.)")
        
        # Parse script
        if isinstance(script_content, str):
            script_lines = parse_script_format(script_content)
        else:
            script_lines = parse_script_list(script_content)
        
        print_progress(f"Parsed {len(script_lines)} dialogue lines")
        
        # Load model
        model = load_model()
        
        # Process batch runs
        total_generated = 0
        last_concatenated_file = None
        
        for run_num in range(1, batch_runs + 1):
            if batch_runs > 1:
                print("\n" + "="*80)
                print(f"BATCH RUN {run_num}/{batch_runs}")
                print("="*80)
            
            # Determine output directory
            if batch_runs > 1:
                output_dir = f"output/Conversations/{script_name}/run_{run_num}"
            else:
                output_dir = f"output/Conversations/{script_name}"
            ensure_output_dir(output_dir)
            
            # Create voice prompts for all voices
            if run_num == 1 or batch_runs > 1:
                print("\n" + "="*60)
                if batch_runs > 1:
                    print(f"RUN {run_num} - PREPARING VOICES")
                else:
                    print("PREPARING VOICES")
                print("="*60)
            
            voice_prompts = create_voice_prompts(model, voices, voice_profiles, temp_dir=output_dir)
            
            if not voice_prompts:
                print_progress("Error: No valid voice prompts created!")
                continue
            
            # Generate conversation
            print("\n" + "="*60)
            if batch_runs > 1:
                print(f"RUN {run_num} - GENERATING CONVERSATION")
            else:
                print("GENERATING CONVERSATION")
            print("="*60)
            audio_files = generate_conversation(
                model=model,
                script_lines=script_lines,
                voice_prompts=voice_prompts,
                output_dir=output_dir,
                conversation_name=script_name
            )
            
            if not audio_files:
                print_progress(f"Warning: No audio files generated for run {run_num}!")
                continue
            
            total_generated += len(audio_files)
            
            # Concatenate audio if requested
            concatenated_file = None
            if concatenate_enabled:
                print("\n" + "="*60)
                if batch_runs > 1:
                    print(f"RUN {run_num} - CONCATENATING AUDIO")
                else:
                    print("CONCATENATING AUDIO")
                print("="*60)
                concatenated_file = f"{output_dir}/{script_name}_full.wav"
                
                # Get sample rate from first audio file
                with wave.open(audio_files[0], 'rb') as wav:
                    sample_rate = wav.getframerate()
                
                concatenate_audio_files(audio_files, concatenated_file, sample_rate)
                last_concatenated_file = concatenated_file
            
            if batch_runs > 1:
                print("\n" + "-"*80)
                print(f"Run {run_num} Complete: {len(audio_files)} lines generated")
                print(f"Output: {output_dir}")
                print("-"*80)
        
        # Summary
        duration = time.time() - start_time
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        if batch_runs > 1:
            print_progress(f"Total batch runs: {batch_runs}")
            print_progress(f"Lines per run: {len(script_lines)}")
            print_progress(f"Total audio files generated: {total_generated}")
        else:
            print_progress(f"Generated {total_generated} audio lines")
        base_output = f"output/Conversations/{script_name}"
        if batch_runs > 1:
            print_progress(f"Output base directory: {base_output} (run_1/, run_2/, etc.)")
        else:
            print_progress(f"Output directory: {base_output}")
        print_progress(f"Total execution time: {duration:.2f} seconds")
        
        # Play last concatenated audio if available
        if last_concatenated_file and play_audio_enabled:
            print("\n" + "="*60)
            if batch_runs > 1:
                print(f"PLAYING LAST RUN CONVERSATION (run_{batch_runs})")
            else:
                print("PLAYING FULL CONVERSATION")
            print("="*60)
            play_audio(last_concatenated_file)
        
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
