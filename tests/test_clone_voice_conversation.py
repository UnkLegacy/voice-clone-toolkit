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

from Clone_Voice_Conversation import (
    load_json_config,
    load_text_from_file_or_string,
    parse_script_format,
    parse_script_list,
    ensure_output_dir,
    save_wav,
    parse_args,
    list_scripts,
)


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


class TestParseScriptFormat(unittest.TestCase):
    """Test script format parsing."""
    
    def test_parse_valid_script(self):
        """Test parsing valid script format."""
        script = """[Actor1] Hello there!
[Actor2] Hi, how are you?
[Actor1] I'm doing great, thanks!"""
        
        result = parse_script_format(script)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("Actor1", "Hello there!"))
        self.assertEqual(result[1], ("Actor2", "Hi, how are you?"))
        self.assertEqual(result[2], ("Actor1", "I'm doing great, thanks!"))
    
    def test_parse_with_empty_lines(self):
        """Test parsing script with empty lines."""
        script = """[Actor1] First line

[Actor2] Second line

"""
        result = parse_script_format(script)
        self.assertEqual(len(result), 2)
    
    def test_parse_malformed_lines(self):
        """Test that malformed lines are skipped."""
        script = """[Actor1] Valid line
This line has no actor
[Actor2] Another valid line"""
        
        result = parse_script_format(script)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "Actor1")
        self.assertEqual(result[1][0], "Actor2")
    
    def test_parse_with_extra_brackets(self):
        """Test parsing lines with brackets in dialogue."""
        script = "[Actor] This is [important] information!"
        result = parse_script_format(script)
        self.assertEqual(result[0], ("Actor", "This is [important] information!"))


class TestParseScriptList(unittest.TestCase):
    """Test script list parsing."""
    
    def test_parse_valid_list(self):
        """Test parsing valid script list."""
        script_list = [
            "[Actor1] Line 1",
            "[Actor2] Line 2",
            "[Actor1] Line 3"
        ]
        
        result = parse_script_list(script_list)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("Actor1", "Line 1"))
        self.assertEqual(result[1], ("Actor2", "Line 2"))
    
    def test_parse_empty_strings(self):
        """Test that empty strings are skipped."""
        script_list = [
            "[Actor1] Line 1",
            "",
            "[Actor2] Line 2"
        ]
        
        result = parse_script_list(script_list)
        self.assertEqual(len(result), 2)
    
    def test_parse_whitespace_handling(self):
        """Test proper whitespace handling."""
        script_list = [
            "  [Actor1]   Line with spaces  ",
            "[Actor2] Normal line"
        ]
        
        result = parse_script_list(script_list)
        self.assertEqual(result[0], ("Actor1", "Line with spaces"))


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


class TestListScripts(unittest.TestCase):
    """Test script listing functionality."""
    
    @patch('builtins.print')
    def test_list_inline_scripts(self, mock_print):
        """Test listing scripts with inline content."""
        test_scripts = {
            "script1": {
                "actors": ["Actor1", "Actor2"],
                "script": [
                    "[Actor1] Line 1",
                    "[Actor2] Line 2"
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
                "actors": ["Actor1"],
                "script": "./scripts/test.txt"
            }
        }
        
        list_scripts(test_scripts)
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
