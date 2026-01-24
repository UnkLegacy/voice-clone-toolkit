"""
Unit tests for src/custom_voice.py

Tests the Custom Voice generation script functionality including:
- Configuration loading
- Profile listing
- Argument parsing
- Output directory creation
- WAV file saving
"""

import unittest
import sys
import os
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.custom_voice import (
    load_custom_voice_profiles,
    ensure_output_dir,
    save_wav_pygame,
    list_speaker_profiles,
    parse_args,
)


class TestLoadCustomVoiceProfiles(unittest.TestCase):
    """Test loading custom voice profiles from JSON."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_profiles.json")
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_load_valid_profiles(self):
        """Test loading valid custom voice profiles."""
        test_profiles = {
            "TestSpeaker": {
                "speaker": "TestSpeaker",
                "language": "English",
                "description": "Test speaker",
                "single_text": "Test text",
                "single_instruct": "",
                "batch_texts": ["Text 1", "Text 2"],
                "batch_languages": ["English", "English"],
                "batch_instructs": ["", "Happy"]
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(test_profiles, f)
        
        profiles = load_custom_voice_profiles(self.config_path)
        
        self.assertIn("TestSpeaker", profiles)
        self.assertEqual(profiles["TestSpeaker"]["speaker"], "TestSpeaker")
        self.assertEqual(profiles["TestSpeaker"]["language"], "English")
        self.assertEqual(len(profiles["TestSpeaker"]["batch_texts"]), 2)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file creates default."""
        profiles = load_custom_voice_profiles(self.config_path)
        
        # Should create default config
        self.assertTrue(os.path.exists(self.config_path))
        self.assertIn("Ryan", profiles)
        self.assertEqual(profiles["Ryan"]["speaker"], "Ryan")
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with open(self.config_path, 'w') as f:
            f.write("{invalid json")
        
        with self.assertRaises(SystemExit):
            load_custom_voice_profiles(self.config_path)


class TestEnsureOutputDir(unittest.TestCase):
    """Test output directory creation."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_directory(self):
        """Test creating a new directory."""
        output_dir = os.path.join(self.test_dir, "test_output")
        
        self.assertFalse(os.path.exists(output_dir))
        ensure_output_dir(output_dir)
        self.assertTrue(os.path.exists(output_dir))
    
    def test_existing_directory(self):
        """Test with existing directory doesn't raise error."""
        output_dir = os.path.join(self.test_dir, "existing")
        os.makedirs(output_dir)
        
        # Should not raise an exception
        ensure_output_dir(output_dir)
        self.assertTrue(os.path.exists(output_dir))


class TestSaveWav(unittest.TestCase):
    """Test WAV file saving functionality."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_int16_audio(self):
        """Test saving int16 audio data."""
        audio_data = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        sample_rate = 12000
        output_file = os.path.join(self.test_dir, "test.wav")
        
        save_wav_pygame(output_file, audio_data, sample_rate)
        
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_save_float_audio(self):
        """Test saving float audio data (should convert to int16)."""
        audio_data = np.random.uniform(-1.0, 1.0, size=1000).astype(np.float32)
        sample_rate = 12000
        output_file = os.path.join(self.test_dir, "test_float.wav")
        
        save_wav_pygame(output_file, audio_data, sample_rate)
        
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_save_creates_parent_dirs(self):
        """Test that saving creates parent directories."""
        nested_path = os.path.join(self.test_dir, "nested", "dir", "test.wav")
        audio_data = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        sample_rate = 12000
        
        self.assertFalse(os.path.exists(os.path.dirname(nested_path)))
        save_wav_pygame(nested_path, audio_data, sample_rate)
        
        self.assertTrue(os.path.exists(nested_path))


class TestListSpeakerProfiles(unittest.TestCase):
    """Test listing speaker profiles."""
    
    def test_list_profiles(self):
        """Test listing speaker profiles."""
        test_profiles = {
            "TestSpeaker1": {
                "speaker": "TestSpeaker1",
                "language": "English",
                "description": "Test speaker 1",
                "single_text": "Test text 1",
                "batch_texts": ["Text 1"]
            },
            "TestSpeaker2": {
                "speaker": "TestSpeaker2",
                "language": "Chinese",
                "description": "Test speaker 2",
                "single_text": "测试文本",
                "batch_texts": ["文本1", "文本2"]
            }
        }
        
        # Should not raise an exception
        try:
            list_speaker_profiles(test_profiles)
        except Exception as e:
            self.fail(f"list_speaker_profiles raised an exception: {e}")


class TestParseArgs(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def setUp(self):
        """Set up test profiles."""
        self.test_profiles = {
            "Ryan": {
                "speaker": "Ryan",
                "language": "English",
                "description": "Test",
                "single_text": "Test",
                "batch_texts": []
            },
            "Sohee": {
                "speaker": "Sohee",
                "language": "Korean",
                "description": "Test",
                "single_text": "Test",
                "batch_texts": []
            }
        }
    
    def test_parse_speaker_argument(self):
        """Test parsing --speaker argument."""
        test_args = ['--speaker', 'Ryan']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertEqual(args.speaker, 'Ryan')
    
    def test_parse_list_speakers(self):
        """Test parsing --list-speakers flag."""
        test_args = ['--list-speakers']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertTrue(args.list_speakers)
    
    def test_parse_no_batch(self):
        """Test parsing --no-batch flag."""
        test_args = ['--no-batch']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertTrue(args.no_batch)
    
    def test_parse_only_single(self):
        """Test parsing --only-single flag."""
        test_args = ['--only-single']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertTrue(args.only_single)
    
    def test_parse_no_play(self):
        """Test parsing --no-play flag."""
        test_args = ['--no-play']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertTrue(args.no_play)


if __name__ == '__main__':
    unittest.main()
