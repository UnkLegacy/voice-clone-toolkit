"""
Unit tests for utils/config_loader.py

Tests configuration loading and profile processing utilities.
"""

import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock

from src.utils.config_loader import (
    load_json_config,
    process_text_fields,
    create_default_voice_clone_profiles,
    create_default_custom_voice_profiles,
    create_default_voice_design_profiles,
    create_default_voice_design_clone_profiles,
    load_voice_clone_profiles,
    load_custom_voice_profiles,
    load_voice_design_profiles,
    load_voice_design_clone_profiles,
    load_conversation_scripts,
    validate_profile_structure,
    get_profile_choices,
    get_default_profile
)


class TestConfigLoaderUtilities(unittest.TestCase):
    """Test configuration loading utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "test_config.json")
        self.test_profiles = {
            "Profile1": {
                "description": "Test profile 1",
                "single_text": "Hello world",
                "batch_texts": ["Text 1", "Text 2"]
            },
            "Profile2": {
                "description": "Test profile 2", 
                "single_text": "file://test.txt",
                "batch_texts": ["Direct text"]
            }
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config_file(self, config_data):
        """Helper to create a test config file."""
        with open(self.test_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def test_load_json_config_existing_file(self):
        """Test load_json_config with existing valid file."""
        self._create_test_config_file(self.test_profiles)
        
        result = load_json_config(self.test_config_path)
        
        self.assertEqual(result, self.test_profiles)
    
    def test_load_json_config_nonexistent_file_no_default(self):
        """Test load_json_config with non-existent file, no default creation."""
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_json_config("nonexistent.json")
            
        self.assertEqual(result, {})
    
    def test_load_json_config_nonexistent_file_with_default(self):
        """Test load_json_config creating default config."""
        def default_factory():
            return {"default": "config"}
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_json_config(
                self.test_config_path,
                create_default=True,
                default_factory=default_factory,
                error_message_prefix="test config"
            )
        
        self.assertEqual(result, {"default": "config"})
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.test_config_path))
        
        # Verify file contents
        with open(self.test_config_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        self.assertEqual(file_data, {"default": "config"})
    
    def test_load_json_config_invalid_json(self):
        """Test load_json_config with invalid JSON."""
        # Create invalid JSON file
        with open(self.test_config_path, 'w') as f:
            f.write("invalid json content {")
        
        with patch('sys.stdout'):  # Suppress progress messages
            with patch('sys.exit') as mock_exit:
                load_json_config(self.test_config_path)
                mock_exit.assert_called_once_with(1)
    
    def test_process_text_fields(self):
        """Test process_text_fields processes file paths."""
        # Create a test text file
        test_text_file = os.path.join(self.temp_dir, "test.txt")
        test_content = "Content from file"
        with open(test_text_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        profiles = {
            "Profile1": {
                "single_text": test_text_file,
                "batch_texts": ["direct text", test_text_file]
            }
        }
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = process_text_fields(profiles, ["single_text", "batch_texts"])
        
        # Check that file path was loaded
        self.assertEqual(result["Profile1"]["single_text"], test_content)
        self.assertEqual(result["Profile1"]["batch_texts"][0], "direct text")
        self.assertEqual(result["Profile1"]["batch_texts"][1], test_content)
    
    def test_create_default_voice_clone_profiles(self):
        """Test create_default_voice_clone_profiles structure."""
        result = create_default_voice_clone_profiles()
        
        self.assertIn("Example", result)
        profile = result["Example"]
        
        required_fields = ["voice_sample_file", "sample_transcript", "single_text", "batch_texts"]
        for field in required_fields:
            self.assertIn(field, profile)
        
        self.assertIsInstance(profile["batch_texts"], list)
    
    def test_create_default_custom_voice_profiles(self):
        """Test create_default_custom_voice_profiles structure."""
        result = create_default_custom_voice_profiles()
        
        self.assertIn("Ryan", result)
        profile = result["Ryan"]
        
        required_fields = ["speaker", "language", "single_text", "batch_texts"]
        for field in required_fields:
            self.assertIn(field, profile)
        
        self.assertEqual(profile["speaker"], "Ryan")
        self.assertEqual(profile["language"], "English")
    
    def test_create_default_voice_design_profiles(self):
        """Test create_default_voice_design_profiles structure."""
        result = create_default_voice_design_profiles()
        
        self.assertIn("Example", result)
        profile = result["Example"]
        
        required_fields = ["instruct", "language", "single_text", "batch_texts"]
        for field in required_fields:
            self.assertIn(field, profile)
    
    def test_create_default_voice_design_clone_profiles(self):
        """Test create_default_voice_design_clone_profiles structure."""
        result = create_default_voice_design_clone_profiles()
        
        self.assertIn("Example", result)
        profile = result["Example"]
        
        required_fields = ["instruct", "language", "single_text", "batch_texts"]
        for field in required_fields:
            self.assertIn(field, profile)
    
    @patch('src.utils.config_loader.load_json_config')
    @patch('src.utils.config_loader.process_text_fields')
    def test_load_voice_clone_profiles(self, mock_process, mock_load):
        """Test load_voice_clone_profiles integration."""
        mock_load.return_value = self.test_profiles
        mock_process.return_value = self.test_profiles
        
        result = load_voice_clone_profiles("test_path.json")
        
        mock_load.assert_called_once()
        mock_process.assert_called_once_with(
            self.test_profiles, 
            ['single_text', 'batch_texts', 'sample_transcript']
        )
        self.assertEqual(result, self.test_profiles)
    
    @patch('src.utils.config_loader.load_json_config')
    def test_load_custom_voice_profiles(self, mock_load):
        """Test load_custom_voice_profiles."""
        mock_load.return_value = self.test_profiles
        
        result = load_custom_voice_profiles("test_path.json")
        
        mock_load.assert_called_once()
        self.assertEqual(result, self.test_profiles)
    
    @patch('src.utils.config_loader.load_json_config')
    def test_load_conversation_scripts(self, mock_load):
        """Test load_conversation_scripts."""
        mock_load.return_value = self.test_profiles
        
        result = load_conversation_scripts("test_path.json")
        
        mock_load.assert_called_once()
        self.assertEqual(result, self.test_profiles)
    
    def test_validate_profile_structure_valid(self):
        """Test validate_profile_structure with valid profile."""
        profile = {"field1": "value1", "field2": "value2", "field3": "value3"}
        required_fields = ["field1", "field2"]
        
        result = validate_profile_structure(profile, required_fields)
        
        self.assertTrue(result)
    
    def test_validate_profile_structure_missing_fields(self):
        """Test validate_profile_structure with missing fields."""
        profile = {"field1": "value1"}
        required_fields = ["field1", "field2", "field3"]
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_profile_structure(profile, required_fields, "TestProfile")
        
        self.assertFalse(result)
    
    def test_get_profile_choices(self):
        """Test get_profile_choices returns profile names."""
        result = get_profile_choices(self.test_profiles)
        
        self.assertEqual(set(result), {"Profile1", "Profile2"})
    
    def test_get_profile_choices_empty(self):
        """Test get_profile_choices with empty profiles."""
        result = get_profile_choices({})
        
        self.assertEqual(result, [])
    
    def test_get_default_profile_with_preferred(self):
        """Test get_default_profile with preferred default that exists."""
        result = get_default_profile(self.test_profiles, "Profile2")
        
        self.assertEqual(result, "Profile2")
    
    def test_get_default_profile_preferred_nonexistent(self):
        """Test get_default_profile with non-existent preferred default."""
        result = get_default_profile(self.test_profiles, "NonexistentProfile")
        
        # Should return first available profile
        self.assertIn(result, ["Profile1", "Profile2"])
    
    def test_get_default_profile_no_preferred(self):
        """Test get_default_profile without preferred default."""
        result = get_default_profile(self.test_profiles)
        
        # Should return first available profile
        self.assertIn(result, ["Profile1", "Profile2"])
    
    def test_get_default_profile_empty_profiles(self):
        """Test get_default_profile with empty profiles."""
        result = get_default_profile({})
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()