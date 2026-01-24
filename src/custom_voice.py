"""
Qwen3-TTS Custom Voice Generation Script

This script demonstrates how to use the Qwen3-TTS CustomVoice model to generate
speech with different speakers and languages.

Available Speakers:
    Speaker     | Voice Description                                   | Native Language
    ------------|----------------------------------------------------|-------------------------
    Vivian      | Bright, slightly edgy young female voice.          | Chinese
    Serena      | Warm, gentle young female voice.                  | Chinese
    Uncle_Fu    | Seasoned male voice with a low, mellow timbre.    | Chinese
    Dylan       | Youthful Beijing male voice with clear, natural timbre. | Chinese (Beijing Dialect)
    Eric        | Lively Chengdu male voice with slightly husky brightness. | Chinese (Sichuan Dialect)
    Ryan        | Dynamic male voice with strong rhythmic drive.    | English
    Aiden       | Sunny American male voice with clear midrange.   | English
    Ono_Anna    | Playful Japanese female voice with light, nimble timbre. | Japanese
    Sohee       | Warm Korean female voice with rich emotion.      | Korean
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


def main():
    """Main function to run the TTS generation pipeline."""
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
            text="Hello Jonathan, I want to gobble your cock.  Please shove it deep in my mouth and fuck me hard.",
            language="English",
            speaker="Sohee",
            instruct="Speak in a deep, breathy, seductive tone",
            output_file="output/Custom_Voice/output_custom_voice.wav"
        )
        
        # Play the generated audio
        print("\n" + "="*60)
        play_audio("output/Custom_Voice/output_custom_voice.wav")
        
        # Batch inference
        # print("\n" + "="*60)
        # print("BATCH VOICE GENERATION")
        # print("="*60)
        # wavs, sr = generate_batch_voices(
        #     model=model,
        #     texts=[
        #         "She sells seashells by the seashore. She sells seashells by the seashore. She sells seashells by the seashore. She sells seashells by the seashore. She sells seashells by the seashore.",
        #         "Peter Piper picked a peck of pickled peppers",
        #         "Fischers Fritze fischt frische Fische, frische Fische fischt Fischers Fritze.",
        #         "Blaukraut bleibt Blaukraut und Brautkleid bleibt Brautkleid."
        #     ],
        #     languages=["English", "English", "German", "German"],
        #     speakers=["Ryan", "Aiden", "Uncle_fu", "Sohee"],
        #     instructs=["", "Very happy.", "Serious.", "Silly."]
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