# Clone Voice Conversation - Complete Guide

Generate realistic multi-character conversations using cloned voices with the `Clone_Voice_Conversation.py` script.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Script Format](#script-format)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Overview

The Conversation Script Generator combines voice cloning with dialogue generation to create realistic multi-character conversations. It:

- ✅ Loads multiple voice profiles simultaneously
- ✅ Parses conversation scripts with `[Actor] dialogue` format
- ✅ Generates audio for each line in sequence
- ✅ Creates individual line files AND a combined conversation file
- ✅ Supports both inline scripts and external script files

## Quick Start

### 1. Basic Usage

```bash
# Run the example conversation
python src/clone_voice_conversation.py
```

This will generate a conversation using the default `example_conversation` script.

### 2. List Available Scripts

```bash
python src/clone_voice_conversation.py --list-scripts
```

Output:
```
============================================================
AVAILABLE CONVERSATION SCRIPTS
============================================================

example_conversation:
  Actors: DougDoug, Example_Grandma
  Lines: 8

tech_discussion:
  Actors: DougDoug, Example_Grandma
  Lines: 9

script_file_example:
  Actors: DougDoug, Example_Grandma
  Lines: 9
============================================================
```

### 3. Run a Specific Script

```bash
python src/clone_voice_conversation.py --script tech_discussion
```

## Configuration

### Configuration Files

The system uses two configuration files:

1. **`config/voice_clone_profiles.json`** - Voice profiles (shared with Clone_Voice.py)
2. **`config/conversation_scripts.json`** - Conversation scripts (specific to conversations)

### conversation_scripts.json Structure

```json
{
  "script_name": {
    "actors": ["Actor1", "Actor2"],
    "script": [
      "[Actor1] First line",
      "[Actor2] Second line"
    ]
  }
}
```

### Fields Explained

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `actors` | array | Yes | List of actor names (must match voice profiles) |
| `script` | array or string | Yes | Either inline dialogue or path to script file |

## Script Format

### Method 1: Inline Scripts (JSON Array)

Define the script directly in the JSON config:

```json
{
  "my_conversation": {
    "actors": ["DougDoug", "Example_Grandma"],
    "script": [
      "[DougDoug] Hey Grandma!",
      "[Example_Grandma] Hello dear!",
      "[DougDoug] How are you?",
      "[Example_Grandma] I'm wonderful, thank you!"
    ]
  }
}
```

**Pros:**
- Everything in one place
- Easy to see the whole conversation
- No additional files needed

**Cons:**
- Can get cluttered for long conversations
- Harder to edit with proper text editor features

### Method 2: External Script Files

Store scripts in separate `.txt` files:

**In `config/conversation_scripts.json`:**
```json
{
  "podcast_episode": {
    "actors": ["Host", "Guest"],
    "script": "./conversation_scripts/podcast_episode_1.txt"
  }
}
```

**In `scripts/podcast_episode_1.txt`:**
```
[Host] Welcome to the podcast!
[Guest] Thanks for having me!
[Host] Let's dive right into it. What's your story?
[Guest] Well, it all started when I was just a kid...
```

**Pros:**
- Cleaner for long conversations
- Easier to edit with text editors
- Better for version control
- Can reuse scripts across configurations

**Cons:**
- Additional files to manage
- Path must be correct

### Script Syntax Rules

1. **Format:** `[ActorName] Dialogue text`
2. **Actor names** must be enclosed in square brackets `[` `]`
3. **Actor names** must match voice profile names exactly (case-sensitive)
4. **Space** required between `]` and dialogue text
5. **Empty lines** are ignored
6. **Malformed lines** are skipped with a warning

**Valid:**
```
[DougDoug] This is a valid line.
[Example_Grandma] This is also valid!
```

**Invalid:**
```
DougDoug This won't work - missing brackets
[DougDoug]This won't work - missing space
This line has no actor - will be skipped
```

## Usage Examples

### Example 1: Simple Conversation

```bash
# Generate the default conversation
python src/clone_voice_conversation.py
```

Output files:
```
output/Conversations/example_conversation/
├── example_conversation_line_001_DougDoug.wav
├── example_conversation_line_002_Example_Grandma.wav
├── example_conversation_line_003_DougDoug.wav
└── example_conversation_full.wav
```

### Example 2: No Playback (Silent Mode)

```bash
# Generate without playing audio
python src/clone_voice_conversation.py --script tech_discussion --no-play
```

Useful for:
- Batch processing
- Remote/headless servers
- When you just want the files

### Example 3: Individual Lines Only

```bash
# Generate without concatenating
python src/clone_voice_conversation.py --no-concatenate
```

This creates individual line files but skips creating the `_full.wav` file.

Useful for:
- Manual editing in audio software
- Selective line replacement
- Custom post-processing

### Example 4: Create Your Own Script

**Step 1:** Add voice profiles (if needed)

```json
// In config/voice_clone_profiles.json
{
  "Narrator": {
    "voice_sample_file": "./input/Narrator.wav",
    "sample_transcript": "./texts/narrator_transcript.txt",
    "single_text": "Sample text",
    "batch_texts": ["Text 1"]
  }
}
```

**Step 2:** Create script file

```
// In scripts/my_story.txt
[Narrator] Once upon a time, in a land far away...
[DougDoug] Wait, are we really starting with 'once upon a time'?
[Narrator] Yes, it's a classic opening!
[Example_Grandma] Oh let him tell the story, dear.
[DougDoug] Fine, fine. Continue, Narrator.
[Narrator] As I was saying... Once upon a time...
```

**Step 3:** Add to config

```json
// In config/conversation_scripts.json
{
  "my_story": {
    "actors": ["Narrator", "DougDoug", "Example_Grandma"],
    "script": "./conversation_scripts/my_story.txt"
  }
}
```

**Step 4:** Generate

```bash
python src/clone_voice_conversation.py --script my_story
```

## Advanced Features

### Silence Between Lines

By default, 0.3 seconds of silence is added between lines in the concatenated audio. This is configurable in the code:

```python
# In Clone_Voice_Conversation.py
concatenate_audio_files(audio_files, concatenated_file, sample_rate, silence_duration=0.3)
```

Change `silence_duration` to adjust:
- `0.0` - No silence (rapid-fire dialogue)
- `0.3` - Default (natural conversation)
- `0.5` - Slower pacing
- `1.0` - Dramatic pauses

### Multiple Actors

The system supports conversations with any number of actors:

```json
{
  "panel_discussion": {
    "actors": ["Host", "Guest1", "Guest2", "Guest3"],
    "script": [
      "[Host] Welcome everyone to the panel!",
      "[Guest1] Thanks for having us!",
      "[Guest2] Great to be here!",
      "[Guest3] Looking forward to it!",
      "[Host] Let's begin with our first topic..."
    ]
  }
}
```

### Mixed Format Scripts

You can mix inline and file-based scripts in the same config:

```json
{
  "quick_greeting": {
    "actors": ["A", "B"],
    "script": [
      "[A] Hi!",
      "[B] Hey!"
    ]
  },
  "long_story": {
    "actors": ["Narrator"],
    "script": "./conversation_scripts/long_story.txt"
  }
}
```

## Troubleshooting

### "Actor not found in voice profiles"

**Problem:** The script references an actor that doesn't have a voice profile.

**Solution:**
1. Check spelling and capitalization in both files
2. Add the actor to `config/voice_clone_profiles.json`
3. Ensure the voice sample file exists

### "Reference audio not found"

**Problem:** The voice profile points to a non-existent audio file.

**Solution:**
1. Verify the path in the voice profile is correct
2. Ensure the audio file exists in the specified location
3. Check file permissions

### "Skipping malformed line"

**Problem:** A script line doesn't match the `[Actor] dialogue` format.

**Solution:**
1. Check that actor name is in square brackets: `[ActorName]`
2. Ensure there's a space after the closing bracket: `[Actor] text`
3. Remove any empty lines if they're causing issues
4. Check for special characters that might break parsing

### Lines Generated Out of Order

**Problem:** The concatenated audio has lines in wrong order.

**Solution:** 
This shouldn't happen - the script processes lines sequentially. If it does:
1. Check the individual line files (they have numbers: `001`, `002`, etc.)
2. The issue might be in the concatenation - check file sorting

### Audio Quality Issues

**Problem:** Generated voices don't sound right.

**Solution:**
1. Verify voice sample files are high quality
2. Check that `sample_transcript` accurately matches the audio
3. Try the voice in `Clone_Voice.py` first to test quality
4. Use `--compare` mode in `Clone_Voice.py` to verify cloning quality

### Memory Issues

**Problem:** Running out of memory with long conversations.

**Solution:**
1. Process conversations in smaller batches
2. Close other applications
3. The model will automatically use CPU if GPU memory is insufficient
4. Break very long scripts into multiple conversation scripts

## Tips for Best Results

### Writing Natural Dialogue

✓ **Use contractions:** "I'm" instead of "I am"  
✓ **Add hesitations:** "Well, I think... maybe?"  
✓ **Include interruptions:** "But wait—"  
✓ **Vary sentence length:** Mix short and long sentences  
✓ **Use punctuation:** Commas for pauses, exclamation for emphasis  

### Voice Consistency

✓ **Test voices individually first** with `Clone_Voice.py`  
✓ **Use high-quality reference audio** (clear, no background noise)  
✓ **Keep reference transcripts accurate** (exact match to audio)  
✓ **Use appropriate text** for each character's personality  

### Technical Tips

✓ **Start small:** Test with 3-4 lines before generating long scripts  
✓ **Name consistently:** Use same actor names throughout  
✓ **Organize files:** Keep scripts in `scripts/` directory  
✓ **Version control:** Track both config and script files  
✓ **Backup outputs:** Save successful generations  

## Performance

**Typical Generation Times:**
- Model loading: 10-30 seconds (one-time)
- Voice prompt creation: 2-5 seconds per actor (one-time)
- Line generation: 1-3 seconds per line
- Concatenation: <1 second

**Example:** A 10-line conversation with 2 actors:
- Initial setup: ~20 seconds
- Generation: ~15-30 seconds
- Total: ~35-50 seconds

**Efficiency Tips:**
- Actors' voice prompts are created once and reused
- GPU is significantly faster than CPU
- Longer lines take slightly more time but not proportionally

## Example Workflow

1. **Prepare voices:**
   ```bash
   # Test your voices first
   python Clone_Voice.py --voice DougDoug --compare --only-single
   ```

2. **Write script:**
   ```bash
   # Create your script file
   nano scripts/my_conversation.txt
   ```

3. **Add to config:**
   ```json
   {
     "my_conversation": {
       "actors": ["DougDoug", "Example_Grandma"],
       "script": "./conversation_scripts/my_conversation.txt"
     }
   }
   ```

4. **Test generation:**
   ```bash
   # Generate and listen
   python src/clone_voice_conversation.py --script my_conversation
   ```

5. **Iterate:**
   - Adjust script based on results
   - Regenerate as needed
   - Individual line files let you replace specific lines

## Support

For issues with:
- **Script functionality:** Check this guide and the README.md
- **Voice quality:** Verify with `Clone_Voice.py` first
- **Model issues:** Visit [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)

---

Happy conversation generating! 🎭🎙️
