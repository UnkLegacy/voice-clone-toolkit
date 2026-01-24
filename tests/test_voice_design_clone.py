"""
Unit tests for src/voice_design_clone.py

Tests the Voice Design + Clone generation script functionality including:
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

from src.voice_design_clone import (
    load_voice_design_clone_profiles,
    ensure_output_dir,
    save_wav_pygame,
    list_design_clone_profiles,
    parse_args,
)


class TestLoadVoiceDesignCloneProfiles(unittest.TestCase):
    """Test loading voice design + clone profiles from JSON."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_profiles.json")
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_load_valid_profiles(self):
        """Test loading valid voice design + clone profiles."""
        test_profiles = {
            "TestCharacter": {
                "description": "Test character",
                "reference": {
                    "text": "Test reference text",
                    "instruct": "Test instruction",
                    "language": "English"
                },
                "single_texts": ["Text 1", "Text 2"],
                "batch_texts": ["Batch 1", "Batch 2"],
                "language": "English"
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(test_profiles, f)
        
        profiles = load_voice_design_clone_profiles(self.config_path)
        
        self.assertIn("TestCharacter", profiles)
        self.assertEqual(profiles["TestCharacter"]["language"], "English")
        self.assertEqual(len(profiles["TestCharacter"]["single_texts"]), 2)
        self.assertEqual(len(profiles["TestCharacter"]["batch_texts"]), 2)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file creates default."""
        profiles = load_voice_design_clone_profiles(self.config_path)
        
        # Should create default config
        self.assertTrue(os.path.exists(self.config_path))
        self.assertIn("Example", profiles)
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with open(self.config_path, 'w') as f:
            f.write("{invalid json")
        
        with self.assertRaises(SystemExit):
            load_voice_design_clone_profiles(self.config_path)


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


class TestListDesignCloneProfiles(unittest.TestCase):
    """Test listing voice design + clone profiles."""
    
    def test_list_profiles(self):
        """Test listing voice design + clone profiles."""
        test_profiles = {
            "TestCharacter1": {
                "description": "Test character 1",
                "reference": {
                    "text": "Reference text",
                    "instruct": "Instruction",
                    "language": "English"
                },
                "single_texts": ["Text 1"],
                "batch_texts": ["Batch 1"],
                "language": "English"
            },
            "TestCharacter2": {
                "description": "Test character 2",
                "reference": {
                    "text": "参考文本",
                    "instruct": "指令",
                    "language": "Chinese"
                },
                "single_texts": ["文本1"],
                "batch_texts": ["批次1"],
                "language": "Chinese"
            }
        }
        
        # Should not raise an exception
        try:
            list_design_clone_profiles(test_profiles)
        except Exception as e:
            self.fail(f"list_design_clone_profiles raised an exception: {e}")


class TestParseArgs(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def setUp(self):
        """Set up test profiles."""
        self.test_profiles = {
            "Nervous_Teen": {
                "description": "Test",
                "reference": {
                    "text": "Test",
                    "instruct": "Test",
                    "language": "English"
                },
                "single_texts": ["Test"],
                "batch_texts": ["Test"],
                "language": "English"
            },
            "Confident_Professional": {
                "description": "Test",
                "reference": {
                    "text": "Test",
                    "instruct": "Test",
                    "language": "English"
                },
                "single_texts": ["Test"],
                "batch_texts": ["Test"],
                "language": "English"
            }
        }
    
    def test_parse_profile_argument(self):
        """Test parsing --profile argument."""
        test_args = ['--profile', 'Nervous_Teen']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertEqual(args.profile, 'Nervous_Teen')
    
    def test_parse_list_profiles(self):
        """Test parsing --list-profiles flag."""
        test_args = ['--list-profiles']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertTrue(args.list_profiles)
    
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
