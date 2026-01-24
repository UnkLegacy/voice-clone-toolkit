"""
Unit tests for Clone_Voice_Conversation.py
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clone_voice_conversation import (
    parse_script_format,
    parse_script_list,
    parse_args,
    list_scripts,
    list_all_voices,
)

# Import utilities from the new modular structure
from src.utils.config_loader import load_json_config
from src.utils.file_utils import load_text_from_file_or_string
from src.utils.audio_utils import ensure_output_dir, save_wav, save_audio, PYDUB_AVAILABLE


class TestLoadJsonConfig(unittest.TestCase):
    """Test JSON configuration loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_valid_config(self):
        """Test loading valid JSON config."""
        test_config = {"key": "value", "number": 42}
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        result = load_json_config(self.config_path)
        self.assertEqual(result, test_config)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns empty dict."""
        result = load_json_config("nonexistent.json")
        self.assertEqual(result, {})
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON exits."""
        with open(self.config_path, 'w') as f:
            f.write("{invalid}")
        
        with self.assertRaises(SystemExit):
            load_json_config(self.config_path)


class TestLoadTextFromFileOrString(unittest.TestCase):
    """Test text loading with proper binary file handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_text_file(self):
        """Test loading actual text files works correctly."""
        text_file = os.path.join(self.test_dir, "test.txt")
        test_content = "This is test content"
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = load_text_from_file_or_string(text_file)
        self.assertEqual(result, test_content)
    
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
    
    def test_nonexistent_file_returns_string(self):
        """Test that non-existent paths are treated as literal strings."""
        fake_path = "./nonexistent/file.wav"
        result = load_text_from_file_or_string(fake_path)
        self.assertEqual(result, fake_path)
    
    def test_literal_string_returned(self):
        """Test that literal strings are returned as-is."""
        literal = "This is just a string, not a file path"
        result = load_text_from_file_or_string(literal)
        self.assertEqual(result, literal)


class TestParseScriptFormat(unittest.TestCase):
    """Test script format parsing."""
    
    def test_parse_valid_script(self):
        """Test parsing valid script format."""
        script = """[Voice1] Hello there!
[Voice2] Hi, how are you?
[Voice1] I'm doing great, thanks!"""
        
        result = parse_script_format(script)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("Voice1", "Hello there!"))
        self.assertEqual(result[1], ("Voice2", "Hi, how are you?"))
        self.assertEqual(result[2], ("Voice1", "I'm doing great, thanks!"))
    
    def test_parse_with_empty_lines(self):
        """Test parsing script with empty lines."""
        script = """[Voice1] First line

[Voice2] Second line

"""
        result = parse_script_format(script)
        self.assertEqual(len(result), 2)
    
    def test_parse_malformed_lines(self):
        """Test that malformed lines are skipped."""
        script = """[Voice1] Valid line
This line has no voice name
[Voice2] Another valid line"""
        
        result = parse_script_format(script)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "Voice1")
        self.assertEqual(result[1][0], "Voice2")
    
    def test_parse_with_extra_brackets(self):
        """Test parsing lines with brackets in dialogue."""
        script = "[Voice] This is [important] information!"
        result = parse_script_format(script)
        self.assertEqual(result[0], ("Voice", "This is [important] information!"))


class TestParseScriptList(unittest.TestCase):
    """Test script list parsing."""
    
    def test_parse_valid_list(self):
        """Test parsing valid script list."""
        script_list = [
            "[Voice1] Line 1",
            "[Voice2] Line 2",
            "[Voice1] Line 3"
        ]
        
        result = parse_script_list(script_list)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("Voice1", "Line 1"))
        self.assertEqual(result[1], ("Voice2", "Line 2"))
    
    def test_parse_empty_strings(self):
        """Test that empty strings are skipped."""
        script_list = [
            "[Voice1] Line 1",
            "",
            "[Voice2] Line 2"
        ]
        
        result = parse_script_list(script_list)
        self.assertEqual(len(result), 2)
    
    def test_parse_whitespace_handling(self):
        """Test proper whitespace handling."""
        script_list = [
            "  [Voice1]   Line with spaces  ",
            "[Voice2] Normal line"
        ]
        
        result = parse_script_list(script_list)
        self.assertEqual(result[0], ("Voice1", "Line with spaces"))


class TestSaveWav(unittest.TestCase):
    """Test WAV file saving."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_save_valid_audio(self):
        """Test saving valid audio data."""
        output_file = os.path.join(self.test_dir, "test.wav")
        audio_data = np.array([0, 1000, -1000, 0], dtype=np.int16)
        
        save_wav(output_file, audio_data, 24000)
        self.assertTrue(os.path.exists(output_file))
    
    def test_save_float_conversion(self):
        """Test conversion of float audio to int16."""
        output_file = os.path.join(self.test_dir, "test.wav")
        audio_data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        
        save_wav(output_file, audio_data, 24000)
        self.assertTrue(os.path.exists(output_file))
    
    def test_save_creates_directories(self):
        """Test that parent directories are created."""
        output_file = os.path.join(self.test_dir, "nested", "dir", "test.wav")
        audio_data = np.array([0, 100], dtype=np.int16)
        
        save_wav(output_file, audio_data, 24000)
        self.assertTrue(os.path.exists(output_file))


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


class TestListScripts(unittest.TestCase):
    """Test script listing functionality."""
    
    @patch('builtins.print')
    def test_list_inline_scripts(self, mock_print):
        """Test listing scripts with inline content."""
        test_scripts = {
            "script1": {
                "voices": ["Voice1", "Voice2"],
                "script": [
                    "[Voice1] Line 1",
                    "[Voice2] Line 2"
                ]
            }
        }
        
        list_scripts(test_scripts)
        self.assertTrue(mock_print.called)
    
    @patch('builtins.print')
    def test_list_file_scripts(self, mock_print):
        """Test listing scripts with file references."""
        test_scripts = {
            "script1": {
                "voices": ["Voice1"],
                "script": "./scripts/test.txt"
            }
        }
        
        list_scripts(test_scripts)
        self.assertTrue(mock_print.called)


class TestListAllVoices(unittest.TestCase):
    """Test listing all voice profiles from all sources."""
    
    @patch('builtins.print')
    def test_list_all_voices(self, mock_print):
        """Test listing all voice profiles from all types."""
        test_profiles = {
            "clone": {
                "Voice1": {
                    "voice_sample_file": "./input/voice1.wav",
                    "sample_transcript": "Transcript 1"
                }
            },
            "custom": {
                "Voice2": {
                    "speaker": "Voice2",
                    "language": "English"
                }
            },
            "design": {
                "Voice3": {
                    "description": "Voice 3",
                    "language": "English"
                }
            },
            "design_clone": {
                "Voice4": {
                    "description": "Voice 4",
                    "language": "English"
                }
            }
        }
        
        # Should not raise an exception
        try:
            list_all_voices(test_profiles)
            success = True
        except Exception as e:
            success = False
            print(f"Exception raised: {e}")
        
        self.assertTrue(success)
        self.assertTrue(mock_print.called)


class TestParseArgs(unittest.TestCase):
    """Test command-line argument parsing."""
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--script', 'test_script'])
    def test_parse_script_argument(self):
        """Test parsing --script argument."""
        args = parse_args()
        self.assertEqual(args.script, 'test_script')
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--list-scripts'])
    def test_parse_list_scripts(self):
        """Test parsing --list-scripts flag."""
        args = parse_args()
        self.assertTrue(args.list_scripts)
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--list-voices'])
    def test_parse_list_voices(self):
        """Test parsing --list-voices flag."""
        args = parse_args()
        self.assertTrue(args.list_voices)
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--no-play'])
    def test_parse_no_play(self):
        """Test parsing --no-play flag."""
        args = parse_args()
        self.assertTrue(args.no_play)
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--no-concatenate'])
    def test_parse_no_concatenate(self):
        """Test parsing --no-concatenate flag."""
        args = parse_args()
        self.assertTrue(args.no_concatenate)
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--output-format', 'mp3'])
    def test_parse_output_format_mp3(self):
        """Test parsing --output-format mp3."""
        args = parse_args()
        self.assertEqual(args.output_format, 'mp3')
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--output-format', 'wav'])
    def test_parse_output_format_wav(self):
        """Test parsing --output-format wav."""
        args = parse_args()
        self.assertEqual(args.output_format, 'wav')
    
    @patch('sys.argv', ['Clone_Voice_Conversation.py', '--bitrate', '320k'])
    def test_parse_bitrate(self):
        """Test parsing --bitrate argument."""
        args = parse_args()
        self.assertEqual(args.bitrate, '320k')


class TestEnsureOutputDir(unittest.TestCase):
    """Test output directory creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_create_new_directory(self):
        """Test creating a new directory."""
        test_path = os.path.join(self.test_dir, "conversations", "test")
        ensure_output_dir(test_path)
        self.assertTrue(os.path.exists(test_path))
    
    def test_existing_directory(self):
        """Test that existing directory doesn't cause error."""
        test_path = os.path.join(self.test_dir, "existing")
        os.makedirs(test_path)
        ensure_output_dir(test_path)
        self.assertTrue(os.path.exists(test_path))


if __name__ == '__main__':
    unittest.main()
