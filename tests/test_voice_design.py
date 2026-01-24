"""
Unit tests for src/voice_design.py

Tests the Voice Design generation script functionality including:
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

from src.voice_design import (
    list_voice_profiles,
    parse_args,
)

# Import utilities from the new modular structure
from src.utils.config_loader import load_voice_design_profiles
from src.utils.audio_utils import ensure_output_dir, save_wav, save_audio, PYDUB_AVAILABLE


class TestLoadVoiceDesignProfiles(unittest.TestCase):
    """Test loading voice design profiles from JSON."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_profiles.json")
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_load_valid_profiles(self):
        """Test loading valid voice design profiles."""
        test_profiles = {
            "TestProfile": {
                "language": "English",
                "description": "Test profile",
                "single_text": "Test text",
                "single_instruct": "Test instruction",
                "batch_texts": ["Text 1", "Text 2"],
                "batch_languages": ["English", "English"],
                "batch_instructs": ["", "Happy"]
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(test_profiles, f)
        
        profiles = load_voice_design_profiles(self.config_path)
        
        self.assertIn("TestProfile", profiles)
        self.assertEqual(profiles["TestProfile"]["language"], "English")
        self.assertEqual(len(profiles["TestProfile"]["batch_texts"]), 2)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file creates default."""
        profiles = load_voice_design_profiles(self.config_path)
        
        # Should create default config
        self.assertTrue(os.path.exists(self.config_path))
        self.assertIn("Example", profiles)
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with open(self.config_path, 'w') as f:
            f.write("{invalid json")
        
        with self.assertRaises(SystemExit):
            load_voice_design_profiles(self.config_path)


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
        
        save_wav(output_file, audio_data, sample_rate)
        
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_save_float_audio(self):
        """Test saving float audio data (should convert to int16)."""
        audio_data = np.random.uniform(-1.0, 1.0, size=1000).astype(np.float32)
        sample_rate = 12000
        output_file = os.path.join(self.test_dir, "test_float.wav")
        
        save_wav(output_file, audio_data, sample_rate)
        
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_save_creates_parent_dirs(self):
        """Test that saving creates parent directories."""
        nested_path = os.path.join(self.test_dir, "nested", "dir", "test.wav")
        audio_data = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        sample_rate = 12000
        
        self.assertFalse(os.path.exists(os.path.dirname(nested_path)))
        save_wav(nested_path, audio_data, sample_rate)
        
        self.assertTrue(os.path.exists(nested_path))


class TestListVoiceProfiles(unittest.TestCase):
    """Test listing voice profiles."""
    
    def test_list_profiles(self):
        """Test listing voice profiles."""
        test_profiles = {
            "TestProfile1": {
                "language": "English",
                "description": "Test profile 1",
                "single_text": "Test text 1",
                "single_instruct": "Test instruction 1",
                "batch_texts": ["Text 1"]
            },
            "TestProfile2": {
                "language": "Chinese",
                "description": "Test profile 2",
                "single_text": "测试文本",
                "single_instruct": "测试指令",
                "batch_texts": ["文本1", "文本2"]
            }
        }
        
        # Should not raise an exception
        try:
            list_voice_profiles(test_profiles)
        except Exception as e:
            self.fail(f"list_voice_profiles raised an exception: {e}")


class TestSaveAudio(unittest.TestCase):
    """Test audio file saving with format conversion."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_save_wav_format(self):
        output_file = os.path.join(self.test_dir, "test")
        audio_data = np.array([0, 1000, -1000, 0], dtype=np.int16)
        result_path = save_audio(output_file, audio_data, 24000, output_format="wav")
        self.assertTrue(result_path.endswith('.wav'))
        self.assertTrue(os.path.exists(result_path))
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_save_mp3_format(self):
        output_file = os.path.join(self.test_dir, "test")
        audio_data = np.array([0, 1000, -1000, 0], dtype=np.int16)
        result_path = save_audio(output_file, audio_data, 24000, output_format="mp3", bitrate="192k")
        self.assertTrue(result_path.endswith('.mp3'))
        self.assertTrue(os.path.exists(result_path))
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_mp3_bitrate_option(self):
        output_file = os.path.join(self.test_dir, "test")
        audio_data = np.array([0] * 24000, dtype=np.int16)
        result_path = save_audio(output_file, audio_data, 24000, output_format="mp3", bitrate="320k")
        self.assertTrue(result_path.endswith('.mp3'))
        self.assertTrue(os.path.exists(result_path))


class TestParseArgs(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def setUp(self):
        """Set up test profiles."""
        self.test_profiles = {
            "Incredulous_Panic": {
                "language": "English",
                "description": "Test",
                "single_text": "Test",
                "single_instruct": "Test",
                "batch_texts": []
            },
            "Professional_Narrator": {
                "language": "English",
                "description": "Test",
                "single_text": "Test",
                "single_instruct": "Test",
                "batch_texts": []
            }
        }
    
    def test_parse_profile_argument(self):
        """Test parsing --profile argument."""
        test_args = ['--profile', 'Incredulous_Panic']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertEqual(args.profile, 'Incredulous_Panic')
    
    def test_parse_list_voices(self):
        """Test parsing --list-voices flag."""
        test_args = ['--list-voices']
        sys.argv = ['test'] + test_args
        
        args = parse_args(self.test_profiles)
        
        self.assertTrue(args.list_voices)
    
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
    
    def test_parse_output_format_mp3(self):
        """Test parsing --output-format mp3."""
        test_args = ['--output-format', 'mp3']
        sys.argv = ['test'] + test_args
        args = parse_args(self.test_profiles)
        self.assertEqual(args.output_format, 'mp3')
    
    def test_parse_output_format_wav(self):
        """Test parsing --output-format wav."""
        test_args = ['--output-format', 'wav']
        sys.argv = ['test'] + test_args
        args = parse_args(self.test_profiles)
        self.assertEqual(args.output_format, 'wav')
    
    def test_parse_bitrate(self):
        """Test parsing --bitrate argument."""
        test_args = ['--bitrate', '320k']
        sys.argv = ['test'] + test_args
        args = parse_args(self.test_profiles)
        self.assertEqual(args.bitrate, '320k')


if __name__ == '__main__':
    unittest.main()
