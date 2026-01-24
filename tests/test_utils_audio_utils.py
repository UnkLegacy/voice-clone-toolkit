"""
Unit tests for utils/audio_utils.py

Tests audio processing, saving, and playback utilities.
"""

import unittest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.utils.audio_utils import (
    ensure_output_dir,
    save_wav,
    save_audio,
    play_audio,
    get_audio_info,
    PYDUB_AVAILABLE,
    playsound
)


class TestAudioUtilities(unittest.TestCase):
    """Test audio processing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_audio_data = np.array([0.1, 0.2, 0.3, -0.1, -0.2], dtype=np.float32)
        self.test_sample_rate = 22050
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_output_dir_creates_directory(self):
        """Test ensure_output_dir creates directory."""
        test_dir = os.path.join(self.temp_dir, "test_output")
        self.assertFalse(os.path.exists(test_dir))
        
        ensure_output_dir(test_dir)
        
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.isdir(test_dir))
    
    def test_ensure_output_dir_existing_directory(self):
        """Test ensure_output_dir with existing directory."""
        # Should not raise an error
        ensure_output_dir(self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_save_wav_creates_file(self):
        """Test save_wav creates WAV file."""
        output_path = os.path.join(self.temp_dir, "test.wav")
        
        # Convert to int16 for the test
        test_data_int16 = (self.test_audio_data * 32767).astype(np.int16)
        
        save_wav(output_path, test_data_int16, self.test_sample_rate)
        
        self.assertTrue(os.path.exists(output_path))
        
        # Verify it's a valid WAV file by checking file size
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_wav_converts_float_to_int16(self):
        """Test save_wav properly converts float data to int16."""
        output_path = os.path.join(self.temp_dir, "test_float.wav")
        
        save_wav(output_path, self.test_audio_data, self.test_sample_rate)
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_save_audio_wav_format(self):
        """Test save_audio with WAV format."""
        output_path = os.path.join(self.temp_dir, "test_audio")
        
        result_path = save_audio(output_path, self.test_audio_data, 
                               self.test_sample_rate, "wav", "192k")
        
        expected_path = os.path.join(self.temp_dir, "test_audio.wav")
        self.assertEqual(result_path, expected_path)
        self.assertTrue(os.path.exists(expected_path))
    
    @patch('src.utils.audio_utils.PYDUB_AVAILABLE', True)
    @patch('src.utils.audio_utils.AudioSegment')
    def test_save_audio_mp3_format_success(self, mock_audiosegment):
        """Test save_audio with MP3 format when pydub is available."""
        output_path = os.path.join(self.temp_dir, "test_audio")
        
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audiosegment.from_wav.return_value = mock_audio
        
        with patch('os.remove') as mock_remove:
            result_path = save_audio(output_path, self.test_audio_data,
                                   self.test_sample_rate, "mp3", "192k")
            
            expected_path = os.path.join(self.temp_dir, "test_audio.mp3")
            self.assertEqual(result_path, expected_path)
            
            # Verify AudioSegment was used correctly
            mock_audiosegment.from_wav.assert_called_once()
            mock_audio.export.assert_called_once_with(expected_path, format="mp3", bitrate="192k")
            
            # Verify WAV file was removed after conversion
            mock_remove.assert_called_once()
    
    @patch('src.utils.audio_utils.PYDUB_AVAILABLE', False)
    def test_save_audio_mp3_format_no_pydub(self):
        """Test save_audio with MP3 format when pydub is unavailable."""
        output_path = os.path.join(self.temp_dir, "test_audio")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result_path = save_audio(output_path, self.test_audio_data,
                                   self.test_sample_rate, "mp3", "192k")
            
            # Should fallback to WAV
            expected_path = os.path.join(self.temp_dir, "test_audio.wav")
            self.assertEqual(result_path, expected_path)
            self.assertTrue(os.path.exists(expected_path))
    
    @patch('src.utils.audio_utils.PYDUB_AVAILABLE', True)
    @patch('src.utils.audio_utils.AudioSegment')
    def test_save_audio_mp3_conversion_error(self, mock_audiosegment):
        """Test save_audio MP3 conversion error handling."""
        output_path = os.path.join(self.temp_dir, "test_audio")
        
        # Mock AudioSegment to raise an exception
        mock_audiosegment.from_wav.side_effect = Exception("Conversion failed")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result_path = save_audio(output_path, self.test_audio_data,
                                   self.test_sample_rate, "mp3", "192k")
            
            # Should fallback to WAV
            expected_path = os.path.join(self.temp_dir, "test_audio.wav")
            self.assertEqual(result_path, expected_path)
    
    @patch('src.utils.audio_utils.playsound')
    def test_play_audio_success(self, mock_playsound):
        """Test play_audio with successful playback."""
        # Create a dummy file
        test_file = os.path.join(self.temp_dir, "test.wav")
        with open(test_file, 'wb') as f:
            f.write(b"dummy audio data")
        
        with patch('sys.stdout'):  # Suppress progress messages
            play_audio(test_file)
            
        mock_playsound.assert_called_once_with(test_file)
    
    def test_play_audio_file_not_found(self):
        """Test play_audio with non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, "nonexistent.wav")
        
        with patch('sys.stdout') as mock_stdout:
            play_audio(non_existent_file)
            # Should print warning message
    
    @patch('src.utils.audio_utils.playsound', None)
    def test_play_audio_no_playsound(self):
        """Test play_audio when playsound is unavailable."""
        # Create a dummy file
        test_file = os.path.join(self.temp_dir, "test.wav")
        with open(test_file, 'wb') as f:
            f.write(b"dummy audio data")
        
        with patch('sys.stdout'):  # Suppress progress messages
            play_audio(test_file)
            # Should print warning about unavailable playsound
    
    def test_get_audio_info_nonexistent_file(self):
        """Test get_audio_info with non-existent file."""
        result = get_audio_info("nonexistent.wav")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "File not found")
    
    def test_get_audio_info_wav_file(self):
        """Test get_audio_info with WAV file."""
        # Create a simple WAV file
        output_path = os.path.join(self.temp_dir, "test.wav")
        test_data_int16 = (self.test_audio_data * 32767).astype(np.int16)
        save_wav(output_path, test_data_int16, self.test_sample_rate)
        
        result = get_audio_info(output_path)
        
        self.assertNotIn("error", result)
        self.assertEqual(result["format"], "WAV")
        self.assertEqual(result["sample_rate"], self.test_sample_rate)
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["bit_depth"], 16)


if __name__ == '__main__':
    unittest.main()