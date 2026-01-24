# Text Files Directory

This directory contains text files that can be referenced in your voice profile configuration.

## Usage

Instead of putting long text directly in the JSON config file, you can create `.txt` files here and reference them with their path.

## Supported Fields

- **sample_transcript**: The transcript of your reference audio
- **single_text**: Text to generate in single generation mode
- **batch_texts**: Texts to generate in batch mode (can reference multiple files)

## Example Files

- `example_transcript.txt` - Example transcript file for reference audio
- `example_single.txt` - Example single generation text
- `example_batch_1.txt` - Example batch text #1
- `dougdoug_transcript.txt` - DougDoug reference audio transcript

## How to Use

1. Create a `.txt` file with your text content

2. In `config/voice_clone_profiles.json`, use the file path instead of inline text:

```json
"sample_transcript": "./texts/my_transcript.txt"
```

Or for batch texts:

```json
"batch_texts": [
  "./texts/chapter_1.txt",
  "./texts/chapter_2.txt",
  "You can also mix inline text!"
]
```

## Benefits

- ✅ Easier to edit longer texts
- ✅ No need to escape quotes or special characters
- ✅ Keep your JSON config clean and readable
- ✅ Reuse the same text across multiple profiles
- ✅ Better for version control
