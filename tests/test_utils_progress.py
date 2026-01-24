"""
Unit tests for utils/progress.py

Tests progress reporting and error handling utilities.
"""

import unittest
import sys
from io import StringIO
from unittest.mock import patch, MagicMock
import argparse

from src.utils.progress import (
    print_progress,
    print_error,
    print_warning,
    handle_mp3_conversion_error,
    handle_audio_playback_error,
    handle_fatal_error,
    handle_processing_error,
    with_error_handling
)


class TestProgressUtilities(unittest.TestCase):
    """Test progress reporting utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_message = "Test message"
        self.test_exception = Exception("Test exception")
        
    def test_print_progress(self):
        """Test print_progress formats messages correctly."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_progress(self.test_message)
            output = mock_stdout.getvalue().strip()
            self.assertEqual(output, f"[INFO] {self.test_message}")
    
    def test_print_error(self):
        """Test print_error formats error messages correctly."""
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            print_error(self.test_message)
            output = mock_stderr.getvalue()
            self.assertEqual(output, f"\n[ERROR] {self.test_message}\n")
    
    def test_print_error_with_traceback(self):
        """Test print_error with traceback enabled."""
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with patch('traceback.print_exc') as mock_traceback:
                print_error(self.test_message, show_traceback=True)
                mock_traceback.assert_called_once()
    
    def test_print_warning(self):
        """Test print_warning formats warning messages correctly."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_warning(self.test_message)
            output = mock_stdout.getvalue().strip()
            self.assertEqual(output, f"[INFO] Warning: {self.test_message}")
    
    def test_handle_mp3_conversion_error(self):
        """Test MP3 conversion error handling returns WAV path."""
        wav_path = "/path/to/file.wav"
        with patch('sys.stdout', new_callable=StringIO):
            result = handle_mp3_conversion_error(self.test_exception, wav_path)
            self.assertEqual(result, wav_path)
    
    def test_handle_audio_playback_error(self):
        """Test audio playback error handling."""
        filepath = "/path/to/audio.wav"
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout, \
             patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            handle_audio_playback_error(self.test_exception, filepath)
            stdout_output = mock_stdout.getvalue()
            stderr_output = mock_stderr.getvalue()
            # Error message goes to stderr, file path goes to stdout
            self.assertIn("Error playing audio", stderr_output)
            self.assertIn(filepath, stdout_output)
    
    def test_handle_fatal_error(self):
        """Test fatal error handling exits with code 1."""
        with patch('sys.stderr', new_callable=StringIO):
            with patch('traceback.print_exc'):
                with patch('sys.exit') as mock_exit:
                    handle_fatal_error(self.test_exception, "test operation")
                    mock_exit.assert_called_once_with(1)
    
    def test_handle_processing_error(self):
        """Test processing error handling returns False."""
        item_name = "test_item"
        with patch('sys.stderr', new_callable=StringIO):
            with patch('traceback.print_exc'):
                result = handle_processing_error(self.test_exception, item_name)
                self.assertFalse(result)
    
    def test_with_error_handling_decorator_success(self):
        """Test error handling decorator with successful function."""
        @with_error_handling(exit_on_error=False)
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
    
    def test_with_error_handling_decorator_error_no_exit(self):
        """Test error handling decorator with error, no exit."""
        @with_error_handling(exit_on_error=False)
        def test_function():
            raise self.test_exception
        
        with patch('sys.stderr', new_callable=StringIO):
            with patch('traceback.print_exc'):
                result = test_function()
                self.assertIsNone(result)
    
    def test_with_error_handling_decorator_error_with_exit(self):
        """Test error handling decorator with error and exit."""
        @with_error_handling(exit_on_error=True)
        def test_function():
            raise self.test_exception
        
        with patch('sys.stderr', new_callable=StringIO):
            with patch('traceback.print_exc'):
                with patch('sys.exit') as mock_exit:
                    test_function()
                    mock_exit.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()