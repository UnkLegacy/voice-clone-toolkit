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
    normalize_audio,
    adjust_volume,
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


class TestVolumeNormalization(unittest.TestCase):
    """Test volume normalization and adjustment functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_normalize_audio_peak_normalization(self):
        """Test peak normalization of audio."""
        # Create quiet audio (low amplitude)
        quiet_audio = np.array([0.1, 0.2, -0.1, -0.2, 0.15], dtype=np.float32)
        
        # Normalize to 95% peak
        normalized = normalize_audio(quiet_audio, target_level=0.95, method="peak")
        
        # Check that peak is approximately 0.95
        peak = np.max(np.abs(normalized))
        self.assertAlmostEqual(peak, 0.95, places=2)
        # Check that values are clipped to [-1, 1]
        self.assertTrue(np.all(normalized >= -1.0))
        self.assertTrue(np.all(normalized <= 1.0))
    
    def test_normalize_audio_int16(self):
        """Test normalization with int16 audio data."""
        # Create quiet int16 audio
        quiet_audio = np.array([1000, 2000, -1000, -2000], dtype=np.int16)
        
        # Normalize
        normalized = normalize_audio(quiet_audio, target_level=0.95, method="peak")
        
        # Should still be int16
        self.assertEqual(normalized.dtype, np.int16)
        # Peak should be approximately 95% of int16 max
        peak = np.max(np.abs(normalized))
        expected_peak = int(0.95 * 32767)
        self.assertAlmostEqual(peak, expected_peak, delta=100)
    
    def test_normalize_audio_silent_audio(self):
        """Test normalization handles silent audio gracefully."""
        silent_audio = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Should return original (no division by zero)
        normalized = normalize_audio(silent_audio, target_level=0.95)
        
        np.testing.assert_array_equal(normalized, silent_audio)
    
    def test_adjust_volume_boost(self):
        """Test volume adjustment with boost factor."""
        audio = np.array([0.5, 0.3, -0.4], dtype=np.float32)
        
        # Boost by 50%
        adjusted = adjust_volume(audio, volume_factor=1.5)
        
        # Check values are multiplied correctly
        expected = audio * 1.5
        np.testing.assert_array_almost_equal(adjusted, expected, decimal=5)
        # Check clipping
        self.assertTrue(np.all(adjusted >= -1.0))
        self.assertTrue(np.all(adjusted <= 1.0))
    
    def test_adjust_volume_reduction(self):
        """Test volume adjustment with reduction factor."""
        audio = np.array([0.8, 0.6, -0.7], dtype=np.float32)
        
        # Reduce by 20%
        adjusted = adjust_volume(audio, volume_factor=0.8)
        
        # Check values are multiplied correctly
        expected = audio * 0.8
        np.testing.assert_array_almost_equal(adjusted, expected, decimal=5)
    
    def test_adjust_volume_int16(self):
        """Test volume adjustment with int16 audio."""
        audio = np.array([10000, 20000, -15000], dtype=np.int16)
        
        # Boost by 50%
        adjusted = adjust_volume(audio, volume_factor=1.5)
        
        # Should still be int16
        self.assertEqual(adjusted.dtype, np.int16)
        # Values should be approximately 1.5x (clipped to int16 range)
        self.assertTrue(np.all(adjusted >= -32767))
        self.assertTrue(np.all(adjusted <= 32767))
    
    def test_adjust_volume_clipping(self):
        """Test that volume adjustment prevents clipping."""
        # Create audio that would clip if not handled
        audio = np.array([0.8, 0.9, -0.85], dtype=np.float32)
        
        # Boost by 2x (would exceed 1.0)
        adjusted = adjust_volume(audio, volume_factor=2.0)
        
        # Should be clipped to [-1, 1]
        self.assertTrue(np.all(adjusted >= -1.0))
        self.assertTrue(np.all(adjusted <= 1.0))
        # Some values should be at the limit
        self.assertTrue(np.any(np.abs(adjusted) >= 0.99))
    
    def test_save_audio_with_normalization(self):
        """Test save_audio with normalization enabled."""
        audio_data = np.array([0.1, 0.2, -0.1, -0.2], dtype=np.float32)
        output_path = os.path.join(self.temp_dir, "normalized.wav")
        
        # Save with normalization
        result_path = save_audio(
            output_path, audio_data, 22050, 
            output_format="wav", normalize=True, target_level=0.95
        )
        
        # File should be created
        self.assertTrue(os.path.exists(result_path))
        # Should be WAV format
        self.assertTrue(result_path.endswith('.wav'))
    
    def test_save_audio_with_volume_adjust(self):
        """Test save_audio with volume adjustment."""
        audio_data = np.array([0.3, 0.4, -0.3], dtype=np.float32)
        output_path = os.path.join(self.temp_dir, "adjusted.wav")
        
        # Save with volume boost
        result_path = save_audio(
            output_path, audio_data, 22050,
            output_format="wav", volume_adjust=1.5
        )
        
        # File should be created
        self.assertTrue(os.path.exists(result_path))


if __name__ == '__main__':
    unittest.main()