# Conversation Scripts Directory

This directory contains script files for conversation generation.

## Script Format

Each line should follow this format:

```
[ActorName] Dialogue text here
```

### Example

```
[DougDoug] Hey there, how's it going?
[Grandma] Oh wonderful, dear! How are you?
[DougDoug] I'm doing great, thanks for asking!
```

## Rules

1. Each line must start with `[ActorName]` where ActorName matches a voice profile
2. The actor name must be enclosed in square brackets
3. After the closing bracket, include a space and then the dialogue
4. Empty lines are ignored
5. Lines without proper `[Actor]` format will be skipped with a warning

## Actor Names

Actor names must match voice profiles defined in `config/voice_clone_profiles.json`

**Available actors (as of now):**
- `DougDoug`
- `Grandma`
- `Example_Grandma`

## How to Use

1. **Create a script file** in this directory (e.g., `my_conversation.txt`)

2. **Add it to** `config/conversation_scripts.json`:

```json
{
  "my_conversation": {
    "actors": ["DougDoug", "Grandma"],
    "script": "./scripts/my_conversation.txt"
  }
}
```

3. **Run the conversation generator:**

```bash
python Clone_Voice_Conversation.py --script my_conversation
```

## Tips

- ✅ Keep dialogue natural and conversational
- ✅ Use punctuation to control pacing and emphasis
- ✅ Break long monologues into multiple lines
- ✅ Test with short scripts first
- ✅ You can mix inline scripts and file-based scripts in the config

## Example Files

- `example_script.txt` - Example conversation between DougDoug and Example_Grandma
