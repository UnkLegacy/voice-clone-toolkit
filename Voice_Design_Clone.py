"""
Qwen3-TTS Voice Design + Clone Generation Script

This script demonstrates how to combine Voice Design and Voice Clone capabilities:
1. Use VoiceDesign model to create a reference audio with desired characteristics
2. Build a reusable clone prompt from that reference
3. Generate new content with the designed voice consistently

This workflow is ideal when you want a consistent character voice across many lines
without re-extracting features every time.
"""

import time
import sys
import os
import torch
import numpy as np
import wave
from pathlib import Path
from typing import Optional, Union, Tuple

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


def main():
    """Main function to run the Voice Design + Clone pipeline."""
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        ensure_output_dir()
        
        # Step 1: Load VoiceDesign model and create reference audio
        print("\n" + "="*60)
        print("STEP 1: CREATE VOICE DESIGN REFERENCE")
        print("="*60)
        
        design_model = load_design_model()
        
        ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
        ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
        
        ref_wavs, sr = create_voice_design_reference(
            design_model=design_model,
            ref_text=ref_text,
            ref_instruct=ref_instruct,
            language="English",
            output_file="output/Voice_Design_Clone/voice_design_reference.wav"
        )
        
        # Play the reference audio
        print("\n" + "="*60)
        print("Playing reference audio...")
        print("="*60)
        play_audio("output/Voice_Design_Clone/voice_design_reference.wav")
        
        # Step 2: Load Clone model and build reusable clone prompt
        print("\n" + "="*60)
        print("STEP 2: BUILD REUSABLE CLONE PROMPT")
        print("="*60)
        
        clone_model = load_clone_model()
        
        print_progress("Creating voice clone prompt from reference...")
        voice_clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=(ref_wavs[0], sr),  # Can also use file path: "voice_design_reference.wav"
            ref_text=ref_text,
        )
        print_progress("Voice clone prompt created successfully!")
        
        # Step 3: Generate single clones using the reusable prompt
        print("\n" + "="*60)
        print("STEP 3: GENERATE SINGLE CLONES (Reusing Prompt)")
        print("="*60)
        
        sentences = [
            "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
            "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
        ]
        
        # Generate first single clone
        wavs1, sr1 = generate_single_clone(
            clone_model=clone_model,
            text=sentences[0],
            language="English",
            voice_clone_prompt=voice_clone_prompt,
            output_file="output/Voice_Design_Clone/clone_single_1.wav"
        )
        
        # Generate second single clone
        wavs2, sr2 = generate_single_clone(
            clone_model=clone_model,
            text=sentences[1],
            language="English",
            voice_clone_prompt=voice_clone_prompt,
            output_file="output/Voice_Design_Clone/clone_single_2.wav"
        )
        
        # Play first single clone
        print("\n" + "="*60)
        print("Playing clone_single_1.wav...")
        print("="*60)
        play_audio("output/Voice_Design_Clone/clone_single_1.wav")
        
        # Step 4: Generate batch clones using the reusable prompt
        print("\n" + "="*60)
        print("STEP 4: GENERATE BATCH CLONES (Reusing Prompt)")
        print("="*60)
        
        additional_sentences = [
            "I mean, not that I was watching you or anything! I was just... adjacent to where you were working!",
            "Sure! Yeah! That would be... I'd like that. A lot. Too much? No, just the right amount!",
        ]
        
        wavs_batch, sr_batch = generate_batch_clone(
            clone_model=clone_model,
            texts=additional_sentences,
            languages=["English", "English"],
            voice_clone_prompt=voice_clone_prompt,
            output_prefix="output/Voice_Design_Clone/clone_batch"
        )
        
        # Play first batch clone
        print("\n" + "="*60)
        print("Playing clone_batch_0.wav...")
        print("="*60)
        play_audio("output/Voice_Design_Clone/clone_batch_0.wav")
        
        total_duration = time.time() - start_time
        print("\n" + "="*60)
        print_progress(f"Total execution time: {total_duration:.2f} seconds")
        print_progress("Generated files:")
        print_progress("  - output/Voice_Design_Clone/voice_design_reference.wav (original designed voice)")
        print_progress("  - output/Voice_Design_Clone/clone_single_1.wav (first single clone)")
        print_progress("  - output/Voice_Design_Clone/clone_single_2.wav (second single clone)")
        print_progress("  - output/Voice_Design_Clone/clone_batch_0.wav (first batch clone)")
        print_progress("  - output/Voice_Design_Clone/clone_batch_1.wav (second batch clone)")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
