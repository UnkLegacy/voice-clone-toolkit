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
from typing import Optional

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


def main():
    """Main function to run the Voice Design TTS generation pipeline."""
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        ensure_output_dir()
        
        # Load model
        model = load_model()
        
        # Single inference
        print("\n" + "="*60)
        print("SINGLE VOICE GENERATION")
        print("="*60)
        wavs, sr = generate_single_voice(
            model=model,
            text="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
            language="English",
            instruct="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
            output_file="output/Voice_Design/output_voice_design.wav"
        )
        
        # Play the generated audio
        print("\n" + "="*60)
        play_audio("output/Voice_Design/output_voice_design.wav")
        
        # Batch inference
        # print("\n" + "="*60)
        # print("BATCH VOICE GENERATION")
        # print("="*60)
        # wavs, sr = generate_batch_voices(
        #     model=model,
        #     texts=[
        #         "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        #         "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
        #     ],
        #     languages=["Chinese", "English"],
        #     instructs=[
        #         "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        #         "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
        #     ]
        # )
        
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