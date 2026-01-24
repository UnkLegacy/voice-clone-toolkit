# Input Audio Files

This directory should contain your reference audio files for voice cloning.

## Supported Formats

- **WAV** (recommended): Best quality, no compression
- **MP3**: Supported but may affect voice quality
- **Other formats**: Check Qwen3-TTS documentation for full list

## Audio Requirements

For best voice cloning results:

- **Duration**: 5-30 seconds is ideal
- **Quality**: High quality, clear audio
- **Background**: No background noise or music
- **Sample Rate**: 24kHz or 48kHz recommended
- **Bit Depth**: 16-bit or higher

## How to Use

1. **Place your audio file** in this directory:
   ```
   input/
   ├── MyVoice.wav
   ├── DougDoug.wav
   └── Grandma.wav
   ```

2. **Reference it** in `config/voice_clone_profiles.json`:
   ```json
   {
     "MyVoice": {
       "voice_sample_file": "./input/MyVoice.wav",
       "sample_transcript": "Accurate transcript of the audio...",
       "single_text": "Text to generate",
       "batch_texts": ["Text 1", "Text 2"]
     }
   }
   ```

3. **Run voice cloning:**
   ```bash
   python Clone_Voice.py --voice MyVoice
   ```

## Tips for Quality

- ✅ Use a quiet environment for recordings
- ✅ Speak naturally and clearly
- ✅ Avoid background music or sound effects
- ✅ Use a good quality microphone
- ✅ Ensure the transcript matches the audio exactly
- ✅ Test with short samples first

## Example Files

You can place your voice samples here:

- `DougDoug.wav` - Example voice sample
- `Grandma.wav` - Example voice sample
- `MyVoice.wav` - Your custom voice sample

**Note**: Audio files are not committed to git (see `.gitignore`)
