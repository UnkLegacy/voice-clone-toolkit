"""
Unit tests for custom voices functionality.

Tests loading and filtering custom voices from custom/custom_voices.json
"""

import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock

from src.utils.config_loader import load_custom_voices_by_type


class TestCustomVoices(unittest.TestCase):
    """Test custom voices loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.custom_voices_path = os.path.join(self.temp_dir, "custom_voices.json")
        
        # Create custom directory structure
        self.custom_dir = os.path.join(self.temp_dir, "custom")
        os.makedirs(self.custom_dir, exist_ok=True)
        self.custom_voices_file = os.path.join(self.custom_dir, "custom_voices.json")
        
        self.test_custom_voices = {
            "Amanda": {
                "profile_type": "custom_voice",
                "speaker": "Sohee",
                "language": "English",
                "description": "Test custom voice",
                "single_text": "Hello",
                "single_instruct": "",
                "batch_texts": ["Text 1"],
                "batch_languages": ["English"],
                "batch_instructs": [""]
            },
            "Jonathan": {
                "profile_type": "custom_voice",
                "speaker": "Ryan",
                "language": "English",
                "description": "Test custom voice 2",
                "single_text": "Hi",
                "single_instruct": "",
                "batch_texts": ["Text 2"],
                "batch_languages": ["English"],
                "batch_instructs": [""]
            },
            "Amanda_Design": {
                "profile_type": "voice_design",
                "language": "English",
                "description": "Test voice design",
                "single_text": "Hello design",
                "single_instruct": "Speak in a test voice",
                "batch_texts": ["Design text 1"],
                "batch_languages": ["English"],
                "batch_instructs": [""]
            },
            "Jonathan_Design": {
                "profile_type": "voice_design",
                "language": "English",
                "description": "Test voice design 2",
                "single_text": "Hi design",
                "single_instruct": "Speak in another test voice",
                "batch_texts": ["Design text 2"],
                "batch_languages": ["English"],
                "batch_instructs": [""]
            },
            "Clone_Voice": {
                "profile_type": "voice_clone",
                "voice_sample_file": "./input/test.wav",
                "sample_transcript": "Test transcript",
                "single_text": "Hello clone",
                "batch_texts": ["Clone text 1"]
            }
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_custom_voices_file(self, data):
        """Helper to create custom voices file."""
        with open(self.custom_voices_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voices_by_type_custom_voice(self, mock_load, mock_exists):
        """Test loading custom_voice type profiles."""
        mock_exists.return_value = True
        mock_load.return_value = self.test_custom_voices
        
        result = load_custom_voices_by_type('custom_voice')
        
        mock_exists.assert_called_once_with("custom/custom_voices.json")
        mock_load.assert_called_once_with("custom/custom_voices.json")
        
        # Should return only custom_voice profiles without profile_type field
        self.assertEqual(len(result), 2)
        self.assertIn("Amanda", result)
        self.assertIn("Jonathan", result)
        self.assertNotIn("profile_type", result["Amanda"])
        self.assertNotIn("profile_type", result["Jonathan"])
        self.assertEqual(result["Amanda"]["speaker"], "Sohee")
        self.assertEqual(result["Jonathan"]["speaker"], "Ryan")
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voices_by_type_voice_design(self, mock_load, mock_exists):
        """Test loading voice_design type profiles."""
        mock_exists.return_value = True
        mock_load.return_value = self.test_custom_voices
        
        result = load_custom_voices_by_type('voice_design')
        
        # Should return only voice_design profiles without profile_type field
        self.assertEqual(len(result), 2)
        self.assertIn("Amanda_Design", result)
        self.assertIn("Jonathan_Design", result)
        self.assertNotIn("profile_type", result["Amanda_Design"])
        self.assertNotIn("profile_type", result["Jonathan_Design"])
        self.assertIn("single_instruct", result["Amanda_Design"])
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voices_by_type_voice_clone(self, mock_load, mock_exists):
        """Test loading voice_clone type profiles."""
        mock_exists.return_value = True
        mock_load.return_value = self.test_custom_voices
        
        result = load_custom_voices_by_type('voice_clone')
        
        # Should return only voice_clone profiles without profile_type field
        self.assertEqual(len(result), 1)
        self.assertIn("Clone_Voice", result)
        self.assertNotIn("profile_type", result["Clone_Voice"])
        self.assertIn("voice_sample_file", result["Clone_Voice"])
    
    @patch('src.utils.config_loader.os.path.exists')
    def test_load_custom_voices_by_type_file_not_exists(self, mock_exists):
        """Test loading when custom voices file doesn't exist."""
        mock_exists.return_value = False
        
        result = load_custom_voices_by_type('custom_voice')
        
        self.assertEqual(result, {})
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voices_by_type_no_matching_profiles(self, mock_load, mock_exists):
        """Test loading when no profiles match the requested type."""
        mock_exists.return_value = True
        # Only custom_voice profiles, but requesting voice_design_clone
        mock_load.return_value = {
            "Amanda": {
                "profile_type": "custom_voice",
                "speaker": "Sohee"
            }
        }
        
        result = load_custom_voices_by_type('voice_design_clone')
        
        self.assertEqual(result, {})
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voices_by_type_removes_profile_type_field(self, mock_load, mock_exists):
        """Test that profile_type field is removed from returned profiles."""
        mock_exists.return_value = True
        mock_load.return_value = {
            "TestVoice": {
                "profile_type": "custom_voice",
                "speaker": "Ryan",
                "language": "English",
                "other_field": "value"
            }
        }
        
        result = load_custom_voices_by_type('custom_voice')
        
        self.assertNotIn("profile_type", result["TestVoice"])
        self.assertIn("speaker", result["TestVoice"])
        self.assertIn("language", result["TestVoice"])
        self.assertIn("other_field", result["TestVoice"])
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voices_by_type_handles_missing_profile_type(self, mock_load, mock_exists):
        """Test that profiles without profile_type are ignored."""
        mock_exists.return_value = True
        mock_load.return_value = {
            "ValidVoice": {
                "profile_type": "custom_voice",
                "speaker": "Ryan"
            },
            "InvalidVoice": {
                "speaker": "Sohee"
                # Missing profile_type
            }
        }
        
        result = load_custom_voices_by_type('custom_voice')
        
        self.assertIn("ValidVoice", result)
        self.assertNotIn("InvalidVoice", result)
    
    @patch('src.utils.config_loader.os.path.exists')
    @patch('src.utils.config_loader.load_json_config')
    @patch('src.utils.config_loader.print_error')
    def test_load_custom_voices_by_type_handles_exception(self, mock_error, mock_load, mock_exists):
        """Test that exceptions are handled gracefully."""
        mock_exists.return_value = True
        mock_load.side_effect = Exception("Test error")
        
        result = load_custom_voices_by_type('custom_voice')
        
        self.assertEqual(result, {})
        mock_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
