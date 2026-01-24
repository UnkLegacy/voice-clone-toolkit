"""
Unit tests for convert_audio_format.py
"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.convert_audio_format import convert_audio_file, convert_directory, PYDUB_AVAILABLE


class TestConvertAudioFile(unittest.TestCase):
    """Test audio file conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_wav_to_mp3_conversion(self):
        """Test converting WAV to MP3."""
        import wave
        
        # Create a test WAV file
        wav_file = os.path.join(self.test_dir, "test.wav")
        with wave.open(wav_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            # Write 1 second of silence
            audio_data = np.zeros(24000, dtype=np.int16)
            wav.writeframes(audio_data.tobytes())
        
        # Convert to MP3 with delete_original=True
        mp3_file = os.path.join(self.test_dir, "test.mp3")
        result = convert_audio_file(wav_file, mp3_file, "mp3", delete_original=True)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(mp3_file))
        self.assertFalse(os.path.exists(wav_file))  # Original should be deleted
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_auto_extension(self):
        """Test that output extension is automatically set."""
        import wave
        
        wav_file = os.path.join(self.test_dir, "test.wav")
        with wave.open(wav_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            audio_data = np.zeros(24000, dtype=np.int16)
            wav.writeframes(audio_data.tobytes())
        
        # Convert without specifying output path
        result = convert_audio_file(wav_file, output_format="mp3")
        
        self.assertTrue(result)
        mp3_file = os.path.join(self.test_dir, "test.mp3")
        self.assertTrue(os.path.exists(mp3_file))
    
    def test_nonexistent_file(self):
        """Test that conversion fails gracefully for nonexistent file."""
        result = convert_audio_file("nonexistent.wav", output_format="mp3")
        self.assertFalse(result)


class TestConvertDirectory(unittest.TestCase):
    """Test directory conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipIf(not PYDUB_AVAILABLE, "pydub not available")
    def test_directory_conversion(self):
        """Test converting all files in a directory."""
        import wave
        
        # Create multiple test WAV files
        for i in range(3):
            wav_file = os.path.join(self.test_dir, f"test{i}.wav")
            with wave.open(wav_file, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                audio_data = np.zeros(24000, dtype=np.int16)
                wav.writeframes(audio_data.tobytes())
        
        # Convert all
        successful, failed = convert_directory(self.test_dir, "mp3", "wav")
        
        self.assertEqual(successful, 3)
        self.assertEqual(failed, 0)
        
        # Check MP3 files exist
        for i in range(3):
            mp3_file = os.path.join(self.test_dir, f"test{i}.mp3")
            self.assertTrue(os.path.exists(mp3_file))
    
    def test_nonexistent_directory(self):
        """Test that conversion handles nonexistent directory."""
        successful, failed = convert_directory("nonexistent_dir", "mp3")
        self.assertEqual(successful, 0)
        self.assertEqual(failed, 0)


if __name__ == '__main__':
    unittest.main()
