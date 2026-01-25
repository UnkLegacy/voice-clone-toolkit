# Voice Design Examples - Complete Guide

Learn how to create custom voice characteristics using natural language instructions with the Voice Design feature.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Voice Attribute Control](#voice-attribute-control)
- [Control Methods](#control-methods)
- [Examples by Use Case](#examples-by-use-case)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Voice Design allows you to describe voice characteristics using natural language, giving you precise control over:

- **Acoustic attributes**: Pitch, speed, volume, clarity, fluency
- **Voice characteristics**: Gender, age, accent, texture
- **Emotional expression**: Emotion, tone, personality
- **Dynamic changes**: Gradual shifts, emotional transitions

Unlike predefined speakers, Voice Design lets you create unique voices tailored to your specific needs.

## Quick Start

### Basic Voice Design Instruction

```json
{
  "single_instruct": "Speak in a clear, professional narrator voice with confidence and authority."
}
```

### Multi-Attribute Control

```json
{
  "single_instruct": "Gender: Female. Pitch: Mid-range female pitch. Speed: Moderate pace. Volume: Conversational. Age: Mid-30s. Clarity: Clear articulation. Accent: American English. Emotion: Warm and friendly. Tone: Professional yet approachable."
}
```

## Voice Attribute Control

### Available Attributes

| Attribute | Description | Example Values |
|-----------|-------------|----------------|
| **Gender** | Voice gender | Male, Female |
| **Pitch** | Voice pitch range | Low, Mid-range, High, Low male pitch, High female pitch |
| **Speed** | Speaking rate | Fast-paced, Slow, Moderate, Deliberate pace |
| **Volume** | Voice loudness | Loud, Soft, Conversational, Projecting |
| **Age** | Perceived age | Young adult, Middle-aged, Elderly, Late 20s |
| **Clarity** | Articulation quality | Clear, Distinct, Muffled, Highly articulate |
| **Fluency** | Speech flow | Fluent, Hesitant, Very fluent, Natural |
| **Accent** | Regional accent | American English, British English, General American |
| **Texture** | Voice quality | Bright, Gravelly, Nasal, Resonant, Breathy |
| **Emotion** | Emotional state | Happy, Sad, Angry, Excited, Calm, Enthusiastic |
| **Tone** | Overall tone | Upbeat, Authoritative, Playful, Professional |
| **Personality** | Character traits | Confident, Shy, Outgoing, Assertive, Friendly |

## Control Methods

### Method 1: Single Attribute Control

Simple, direct instructions for one characteristic:

**Examples:**
- `"Speak with a very sad and tearful voice."`
- `"Very happy."`
- `"Said in a very angry tone."`
- `"Please speak very softly and quietly."`
- `"Speaking at an extremely slow pace."`
- `"Low-pitched."`

**Use when:** You need a quick, simple voice modification.

### Method 2: Multi-Attribute Control

Comprehensive control using structured format:

**Format:**
```
Attribute: Description.
Attribute: Description.
...
```

**Example:**
```
Gender: Male.
Pitch: Low male pitch with significant upward inflections for emphasis and excitement.
Speed: Fast-paced delivery with deliberate pauses for dramatic effect.
Volume: Loud and projecting, increasing notably during moments of praise and announcements.
Age: Young adult to middle-aged adult.
Clarity: Highly articulate and distinct pronunciation.
Fluency: Very fluent speech with no hesitations.
Accent: British English.
Texture: Bright and clear vocal texture.
Emotion: Enthusiastic and excited, especially when complimenting.
Tone: Upbeat, authoritative, and performative.
Personality: Confident, extroverted, and engaging.
```

**Use when:** You need precise control over multiple voice characteristics.

### Method 3: Natural Language Description

Fluid, narrative-style descriptions:

**Example:**
```
A relaxed, naturally expressive male voice in his late twenties to early thirties, with a moderately low pitch, casual speaking rate, and conversational volume; deliver lines with a light, self-deprecating tone, breaking into genuine, easygoing laughter at moments of embarrassment, while maintaining clear articulation and an overall warm, approachable clarity.
```

**Use when:** You want a more natural, flowing description that captures the overall voice character.

### Method 4: Character-Based Description

Rich descriptions based on character background:

**Example:**
```
Character Name: Marcus Cole
Voice Profile: A bright, agile male voice with a natural upward lift, delivering lines at a brisk, energetic pace. Pitch leans high with spark, volume projects clearly—near-shouting at peaks—to convey urgency and excitement. Speech flows seamlessly, fluently, each word sharply defined, riding a current of dynamic rhythm.
Background: Longtime broadcast booth announcer for national television, specializing in live interstitials and public engagement spots. His voice bridges segments, rallies action, and keeps momentum alive—from voter drives to entertainment news.
Personality: Energetic, precise, inherently engaging. He doesn't just read—he propels. Behind the speed is intent: to inform fast, to move people to act.
```

**Use when:** You're creating voices for specific characters with detailed backgrounds.

## Examples by Use Case

### Professional Narrator

**Multi-Attribute Format:**
```
Gender: Male.
Pitch: Low male pitch, generally stable.
Speed: Deliberate pace, slowing slightly after the initial exclamation.
Volume: Starts loud, then transitions to a projected conversational volume.
Age: Middle-aged adult.
Clarity: High clarity with distinct pronunciation.
Fluency: Highly fluent.
Accent: American English.
Texture: Resonant and slightly gravelly.
Emotion: Initially commanding, shifting to narrative amusement.
Tone: Authoritative start, moving to an engaging, descriptive tone.
Personality: Confident and performative.
```

### Emotional Character

**Gradual Control Example:**
```
Gender: Female.
Pitch: Mid-range female pitch, rising sharply with frustration.
Speed: Starts measured, then accelerates rapidly during emotional outburst.
Volume: Begins conversational, escalates quickly to loud and forceful.
Age: Young adult to middle-aged.
Clarity: High clarity and distinct articulation throughout.
Fluency: Highly fluent with no significant pauses or fillers.
Accent: General American English.
Texture: Bright and clear vocal quality.
Emotion: Shifts abruptly from neutral acceptance to intense resentment and anger.
Tone: Initially accepting, becomes sharply accusatory and confrontational.
Personality: Assertive and emotionally expressive when provoked.
```

### Playful Character Voice

**Multi-Attribute Format:**
```
Gender: Male.
Pitch: Artificially high-pitched, slightly lowering after the initial laugh.
Speed: Rapid during the laugh, then slowing to a deliberate pace.
Volume: Loud laugh transitioning to a standard conversational level.
Age: Young adult to middle-aged, performing a character voice.
Clarity: Clear and distinct articulation.
Fluency: Fluent delivery without hesitation.
Accent: American English.
Texture: Slightly strained and somewhat nasal quality.
Emotion: Forced amusement shifting to feigned resignation.
Tone: Initially playful, then shifts to a slightly put-upon tone.
Personality: Theatrical and expressive.
```

### Sarcastic Teenage Character

**Natural Language Format:**
```
Speak as a sarcastic, assertive teenage girl: crisp enunciation, controlled volume, with vocal emphasis that conveys disdain and authority.
```

### Emotional Storytelling

**Narrative Format:**
```
Speaking in a deeply sorrowful tone, with a noticeable tremor in her voice, she spoke slowly and softly, as if each word carried immense pain. Her voice trembled and was subdued, yet her words, though softly spoken, were clear and distinct, revealing the profound sadness and helplessness hidden deep within her heart.
```

### Persuasive Speaker

**Dynamic Control:**
```
The voice maintains the characteristics of a young woman, exhibiting a clear and slightly urgent tone. The speaking speed starts steadily and gradually increases during the narration, the volume increases with emotional fluctuations, and the intonation rises at the end of sentences to emphasize the persuasive tone.
```

### Cross-Lingual Voice

**International Character:**
```
Gender: Female voice.
Pitch: Female mid-to-high range, with varied intonation.
Speaking speed: Fast-paced, occasionally accelerating.
Volume: Normal conversational volume, with loud laughter.
Clarity: Clear articulation and standard pronunciation.
Fluency: Fluent and natural expression.
Accent: Mandarin Chinese.
Timbre/Tone quality: Bright and slightly cheerful.
Emotion: Pleasant and friendly, accompanied by cheerful laughter.
Intonation: Upbeat and lively intonation, especially noticeable when asking questions.
Personality: Outgoing, cheerful, and talkative.
```

### Confident Middle-Aged Woman

**Natural Language:**
```
A deep, rich, and solid vocal register characteristic of a middle-aged woman, with full and powerful volume. Speech is delivered at a steady pace, articulation clear and precise, with fluent and confident intonation that rises slightly at the end of sentences.
```

### Assertive Professional

**Structured Format:**
```
The voice should be that of a straightforward and slightly assertive middle-aged woman, with a slightly sharp tone. The delivery should be fluent, with occasional pauses to emphasize certain points. The overall emotion should convey a slight dissatisfaction, and the volume should increase slightly with emotional intensity.
```

## Best Practices

### 1. Start Simple

Begin with basic attributes and add complexity as needed:

```json
"single_instruct": "Speak in a warm, friendly tone."
```

Then refine:
```json
"single_instruct": "Gender: Female. Pitch: Mid-range. Speed: Moderate. Emotion: Warm and friendly. Tone: Approachable and caring."
```

### 2. Be Specific

Vague instructions produce inconsistent results:

❌ **Bad:** `"Speak nicely."`
✅ **Good:** `"Speak in a warm, friendly tone with clear articulation and a moderate pace."`

### 3. Use Consistent Format

Choose one format style and stick with it for similar voices:

- Use **multi-attribute format** for technical/professional voices
- Use **natural language** for character voices
- Use **single attribute** for quick modifications

### 4. Test and Iterate

Voice design is iterative. Generate samples, listen, and refine:

1. Start with basic instruction
2. Generate a test sample
3. Identify what needs adjustment
4. Refine the instruction
5. Repeat until satisfied

### 5. Combine with Batch Instructions

Use different instructions for different texts to show voice range:

```json
{
  "batch_texts": [
    "Hello, welcome to our presentation.",
    "This is exciting news!",
    "I'm sorry to hear about that."
  ],
  "batch_instructs": [
    "Speak in a professional, welcoming tone.",
    "Speak with enthusiasm and excitement, volume rising slightly.",
    "Speak softly and sympathetically, with a gentle, caring tone."
  ]
}
```

### 6. Document Your Voices

Keep notes on what works:

```json
{
  "description": "Professional narrator - works best with clear, authoritative instructions",
  "single_instruct": "Gender: Male. Pitch: Low. Speed: Deliberate. Tone: Authoritative."
}
```

## Troubleshooting

### Voice Doesn't Match Description

**Problem:** Generated voice doesn't match your instructions.

**Solutions:**
- Be more specific about attributes
- Use multi-attribute format for complex voices
- Check that conflicting attributes aren't used (e.g., "very fast" and "very slow")
- Try breaking complex instructions into simpler parts

### Inconsistent Results

**Problem:** Same instruction produces different voices each time.

**Solutions:**
- Use more specific attribute descriptions
- Include more attributes to constrain the voice space
- Use structured format instead of natural language
- Test with batch runs to see variation range

### Voice Sounds Unnatural

**Problem:** Generated voice sounds robotic or artificial.

**Solutions:**
- Add fluency and naturalness attributes: "Fluent and natural expression"
- Include personality traits: "Warm and approachable"
- Use natural language descriptions
- Add emotional context: "Speak with genuine emotion"

### Too Many Attributes

**Problem:** Instruction is too long and complex.

**Solutions:**
- Focus on the 3-5 most important attributes
- Use natural language for complex descriptions
- Break into separate batch instructions for different contexts
- Simplify and test incrementally

### Accent Not Working

**Problem:** Accent doesn't come through in the generated voice.

**Solutions:**
- Be explicit: "British English accent" not just "British"
- Combine with other attributes: "British English accent with clear articulation"
- Use accent-specific examples in batch texts
- Check supported languages and accents

## Advanced Techniques

### Timbre Reuse

Reference existing voice characteristics:

```
"Lucas": "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
"Mia": "Female, 16 years old, mezzo-soprano range, softening - lowering register to intimate speaking voice, consonants softening"
```

### Dynamic Voice Changes

Describe how the voice changes over time or context:

```
Gender: Female.
Pitch: Mid-range female pitch, rising sharply with frustration.
Speed: Starts measured, then accelerates rapidly during emotional outburst.
Volume: Begins conversational, escalates quickly to loud and forceful.
Emotion: Shifts abruptly from neutral acceptance to intense resentment and anger.
```

### Context-Aware Instructions

Use batch instructions to show voice range:

```json
{
  "batch_texts": [
    "I'm so excited about this!",
    "Let me explain the details.",
    "I understand your concern."
  ],
  "batch_instructs": [
    "Speak with high energy and enthusiasm, pitch rising with excitement.",
    "Speak clearly and professionally.",
    "Speak warmly and empathetically, showing understanding."
  ]
}
```

## Related Documentation

- [Project Structure](PROJECT_STRUCTURE.md) - Learn about configuration files
- [Conversation Guide](CONVERSATION_GUIDE.md) - Use voice design in conversations
- [Main README](../README.md) - General usage and quick start

## Summary

Voice Design gives you powerful control over voice characteristics through natural language instructions. Key takeaways:

- ✅ Use **multi-attribute format** for precise control
- ✅ Use **natural language** for character voices
- ✅ Start simple and iterate
- ✅ Be specific and consistent
- ✅ Test with batch instructions to show voice range
- ✅ Document what works for future reference

Experiment, test, and refine to create the perfect voice for your needs!
