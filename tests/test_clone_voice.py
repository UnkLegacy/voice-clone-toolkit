"""
Unit tests for Clone_Voice.py
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clone_voice import (
    list_voice_profiles,
    parse_args,
)

# Import utilities from the new modular structure
from src.utils.config_loader import load_voice_clone_profiles
from src.utils.file_utils import load_text_from_file_or_string
from src.utils.audio_utils import ensure_output_dir, save_wav, save_audio, PYDUB_AVAILABLE


class TestLoadVoiceProfiles(unittest.TestCase):
    """Test voice profile loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_profiles.json")
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_valid_profiles(self):
        """Test loading valid voice profiles."""
        test_profiles = {
            "TestVoice": {
                "voice_sample_file": "./input/test.wav",
                "sample_transcript": "Test transcript",
                "single_text": "Test single",
                "batch_texts": ["Test 1", "Test 2"]
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_profiles, f)
        
        profiles = load_voice_clone_profiles(self.config_path)
        
        self.assertIn("TestVoice", profiles)
        self.assertEqual(profiles["TestVoice"]["sample_transcript"], "Test transcript")
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file creates default."""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.json")
        profiles = load_voice_clone_profiles(nonexistent_path)
        
        self.assertIn("Example", profiles)
        self.assertTrue(os.path.exists(nonexistent_path))
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with open(self.config_path, 'w') as f:
            f.write("{invalid json}")
        
        with self.assertRaises(SystemExit):
            load_voice_clone_profiles(self.config_path)


class TestLoadTextFromFile(unittest.TestCase):
    """Test text loading from file or string."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.txt")
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_from_file(self):
        """Test loading text from existing file."""
        test_content = "This is test content"
        with open(self.test_file, 'w') as f:
            f.write(test_content)
        
        result = load_text_from_file_or_string(self.test_file)
        self.assertEqual(result, test_content)
    
    def test_load_inline_string(self):
        """Test returning inline string when not a file."""
        test_string = "This is inline text"
        result = load_text_from_file_or_string(test_string)
        self.assertEqual(result, test_string)
    
    def test_load_list_of_strings(self):
        """Test loading list with mixed file paths and strings."""
        with open(self.test_file, 'w') as f:
            f.write("File content")
        
        test_list = [self.test_file, "inline text"]
        result = load_text_from_file_or_string(test_list)
        
        self.assertEqual(result[0], "File content")
        self.assertEqual(result[1], "inline text")
    
    def test_load_nonexistent_file(self):
        """Test that nonexistent file path is treated as string."""
        fake_path = "./nonexistent/file.txt"
        result = load_text_from_file_or_string(fake_path)
        self.assertEqual(result, fake_path)
    
    def test_binary_file_returns_path(self):
        """Test that binary files (like .wav) return the path without UTF-8 error."""
        # Create a fake binary WAV file with non-UTF-8 bytes
        wav_file = os.path.join(self.test_dir, "test.wav")
        
        # Write binary data that will fail UTF-8 decoding
        with open(wav_file, 'wb') as f:
            # Write WAV-like header with bytes that aren't valid UTF-8
            f.write(b'RIFF\xfc\x00\x00\x00WAVEfmt ')
        
        # Should return the path without raising UTF-8 decode error
        result = load_text_from_file_or_string(wav_file)
        
        # Should return the original path since it can't be read as text
        self.assertEqual(result, wav_file)


class TestEnsureOutputDir(unittest.TestCase):
    """Test output directory creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_create_directory(self):
        """Test creating a new directory."""
        test_path = os.path.join(self.test_dir, "output", "test")
        ensure_output_dir(test_path)
        self.assertTrue(os.path.exists(test_path))
    
    def test_existing_directory(self):
        """Test with existing directory doesn't raise error."""
        test_path = os.path.join(self.test_dir, "existing")
        os.makedirs(test_path)
        ensure_output_dir(test_path)  # Should not raise
        self.assertTrue(os.path.exists(test_path))


class TestSaveWav(unittest.TestCase):
    """Test WAV file saving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.wav")
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_save_int16_audio(self):
        """Test saving int16 audio data."""
        audio_data = np.array([0, 1000, -1000, 0], dtype=np.int16)
        sample_rate = 24000
        
        save_wav(self.test_file, audio_data, sample_rate)
        self.assertTrue(os.path.exists(self.test_file))
    
    def test_save_float_audio(self):
        """Test saving float audio data (should convert to int16)."""
        audio_data = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
        sample_rate = 24000
        
        save_wav(self.test_file, audio_data, sample_rate)
        self.assertTrue(os.path.exists(self.test_file))
    
    def test_save_creates_parent_dirs(self):
        """Test that saving creates parent directories."""
        nested_path = os.path.join(self.test_dir, "nested", "dir", "test.wav")
        audio_data = np.array([0, 1000], dtype=np.int16)
        
        save_wav(nested_path, audio_data, 24000)
        self.assertTrue(os.path.exists(nested_path))


class TestListVoiceProfiles(unittest.TestCase):
    """Test voice profile listing."""
    
    @patch('builtins.print')
    def test_list_profiles(self, mock_print):
        """Test listing voice profiles."""
        test_profiles = {
            "Voice1": {
                "voice_sample_file": "./input/voice1.wav",
                "sample_transcript": "Transcript 1",
                "single_text": "Single 1",
                "batch_texts": ["Batch 1", "Batch 2"]
            },
            "Voice2": {
                "voice_sample_file": "./input/voice2.wav",
                "sample_transcript": "Transcript 2",
                "single_text": "Single 2",
                "batch_texts": ["Batch 1"]
            }
        }
        
        list_voice_profiles(test_profiles)
        
        # Check that print was called
        self.assertTrue(mock_print.called)
        
        # Check that voice names were printed
        call_args = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Voice1" in str(arg) for arg in call_args))
        self.assertTrue(any("Voice2" in str(arg) for arg in call_args))


class TestSaveAudio(unittest.TestCase):
    """Test audio file saving with format conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_save_wav_format(self):
        """Test saving in WAV format."""
        output_file = os.path.join(self.test_dir, "test")
        audio_data = np.array([0, 1000, -1000, 0], dtype=np.int16)
        
        result_path = save_audio(output_file, audio_data, 24000, output_format="wav")
        
        self.assertTrue(result_path.endswith('.wav'))
        self.assertTrue(os.path.exists(result_path))
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_save_mp3_format(self):
        """Test saving in MP3 format."""
        output_file = os.path.join(self.test_dir, "test")
        audio_data = np.array([0, 1000, -1000, 0], dtype=np.int16)
        
        result_path = save_audio(output_file, audio_data, 24000, output_format="mp3", bitrate="192k")
        
        self.assertTrue(result_path.endswith('.mp3'))
        self.assertTrue(os.path.exists(result_path))
        # WAV file should be deleted after conversion
        wav_path = os.path.join(self.test_dir, "test.wav")
        self.assertFalse(os.path.exists(wav_path))
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_mp3_bitrate_option(self):
        """Test MP3 encoding with different bitrate."""
        output_file = os.path.join(self.test_dir, "test")
        # Generate 1 second of audio
        audio_data = np.array([0] * 24000, dtype=np.int16)
        
        # Test with 320k bitrate
        result_path = save_audio(output_file, audio_data, 24000, output_format="mp3", bitrate="320k")
        
        self.assertTrue(result_path.endswith('.mp3'))
        self.assertTrue(os.path.exists(result_path))


class TestParseArgs(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_profiles = {
            "TestVoice": {
                "voice_sample_file": "./input/test.wav",
                "sample_transcript": "Test",
                "single_text": "Test",
                "batch_texts": ["Test"]
            }
        }
    
    @patch('sys.argv', ['Clone_Voice.py', '--voice', 'TestVoice'])
    def test_parse_voice_argument(self):
        """Test parsing --voice argument."""
        args = parse_args(self.test_profiles)
        self.assertEqual(args.voice, 'TestVoice')
    
    @patch('sys.argv', ['Clone_Voice.py', '--voices', 'TestVoice', 'TestVoice'])
    def test_parse_voices_argument(self):
        """Test parsing --voices argument."""
        args = parse_args(self.test_profiles)
        self.assertEqual(args.voices, ['TestVoice', 'TestVoice'])
    
    @patch('sys.argv', ['Clone_Voice.py', '--no-batch'])
    def test_parse_no_batch(self):
        """Test parsing --no-batch flag."""
        args = parse_args(self.test_profiles)
        self.assertTrue(args.no_batch)
    
    @patch('sys.argv', ['Clone_Voice.py', '--compare'])
    def test_parse_compare(self):
        """Test parsing --compare flag."""
        args = parse_args(self.test_profiles)
        self.assertTrue(args.compare)
    
    @patch('sys.argv', ['Clone_Voice.py', '--list-voices'])
    def test_parse_list_voices(self):
        """Test parsing --list-voices flag."""
        args = parse_args(self.test_profiles)
        self.assertTrue(args.list_voices)
    
    @patch('sys.argv', ['Clone_Voice.py', '--output-format', 'mp3'])
    def test_parse_output_format_mp3(self):
        """Test parsing --output-format mp3."""
        args = parse_args(self.test_profiles)
        self.assertEqual(args.output_format, 'mp3')
    
    @patch('sys.argv', ['Clone_Voice.py', '--output-format', 'wav'])
    def test_parse_output_format_wav(self):
        """Test parsing --output-format wav."""
        args = parse_args(self.test_profiles)
        self.assertEqual(args.output_format, 'wav')
    
    @patch('sys.argv', ['Clone_Voice.py', '--bitrate', '320k'])
    def test_parse_bitrate(self):
        """Test parsing --bitrate argument."""
        args = parse_args(self.test_profiles)
        self.assertEqual(args.bitrate, '320k')


if __name__ == '__main__':
    unittest.main()
