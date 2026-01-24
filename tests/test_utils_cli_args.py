"""
Unit tests for utils/cli_args.py

Tests command-line argument parsing utilities.
"""

import unittest
import argparse
from unittest.mock import patch

from src.utils.cli_args import (
    create_base_parser,
    add_audio_format_args,
    add_generation_control_args,
    add_playback_args,
    add_profile_listing_args,
    add_voice_selection_args,
    add_multi_voice_selection_args,
    add_common_args,
    validate_generation_args,
    get_generation_modes,
    create_standard_parser
)


class TestCLIArgsUtilities(unittest.TestCase):
    """Test CLI argument parsing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_profiles = {
            "Profile1": {"description": "Test profile 1"},
            "Profile2": {"description": "Test profile 2"}
        }
        self.default_profile = "Profile1"
    
    def test_create_base_parser(self):
        """Test create_base_parser creates parser with correct description."""
        description = "Test script description"
        parser = create_base_parser(description)
        
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(parser.description, description)
    
    def test_create_base_parser_with_profiles_and_examples(self):
        """Test create_base_parser with profiles and examples."""
        description = "Test script"
        examples = ["Example 1", "Example 2"]
        
        parser = create_base_parser(
            description, 
            available_profiles=self.test_profiles,
            examples=examples
        )
        
        self.assertIsInstance(parser, argparse.ArgumentParser)
        # Check that epilog contains profile names
        self.assertIn("Profile1", parser.epilog)
        self.assertIn("Profile2", parser.epilog)
    
    def test_add_audio_format_args(self):
        """Test add_audio_format_args adds correct arguments."""
        parser = argparse.ArgumentParser()
        add_audio_format_args(parser)
        
        # Parse test arguments
        args = parser.parse_args(['--output-format', 'mp3', '--bitrate', '320k'])
        
        self.assertEqual(args.output_format, 'mp3')
        self.assertEqual(args.bitrate, '320k')
    
    def test_add_audio_format_args_defaults(self):
        """Test add_audio_format_args default values."""
        parser = argparse.ArgumentParser()
        add_audio_format_args(parser)
        
        args = parser.parse_args([])
        
        self.assertEqual(args.output_format, 'wav')
        self.assertEqual(args.bitrate, '192k')
    
    def test_add_generation_control_args(self):
        """Test add_generation_control_args adds control arguments."""
        parser = argparse.ArgumentParser()
        add_generation_control_args(parser, default_batch_runs=3)
        
        args = parser.parse_args(['--no-single', '--batch-runs', '5'])
        
        self.assertTrue(args.no_single)
        self.assertEqual(args.batch_runs, 5)
        self.assertFalse(args.no_batch)  # Default value
    
    def test_add_playback_args(self):
        """Test add_playback_args adds playback control."""
        parser = argparse.ArgumentParser()
        add_playback_args(parser)
        
        args = parser.parse_args(['--no-play'])
        
        self.assertTrue(args.no_play)
    
    def test_add_profile_listing_args(self):
        """Test add_profile_listing_args adds listing argument."""
        parser = argparse.ArgumentParser()
        add_profile_listing_args(parser, profile_type="test profiles")
        
        args = parser.parse_args(['--list-voices'])
        
        self.assertTrue(args.list_voices)
    
    def test_add_voice_selection_args(self):
        """Test add_voice_selection_args adds selection argument."""
        parser = argparse.ArgumentParser()
        add_voice_selection_args(
            parser, self.test_profiles, self.default_profile,
            arg_name="profile", arg_short="p"
        )
        
        args = parser.parse_args(['--profile', 'Profile2'])
        
        self.assertEqual(args.profile, 'Profile2')
    
    def test_add_multi_voice_selection_args(self):
        """Test add_multi_voice_selection_args adds multi-selection."""
        parser = argparse.ArgumentParser()
        add_multi_voice_selection_args(parser, self.test_profiles)
        
        args = parser.parse_args(['--voices', 'Profile1', 'Profile2'])
        
        self.assertEqual(args.voices, ['Profile1', 'Profile2'])
    
    def test_add_common_args(self):
        """Test add_common_args adds all common arguments."""
        parser = argparse.ArgumentParser()
        add_common_args(parser, default_batch_runs=2)
        
        args = parser.parse_args([
            '--output-format', 'mp3',
            '--bitrate', '256k',
            '--no-single',
            '--batch-runs', '4',
            '--no-play',
            '--list-voices'
        ])
        
        # Check audio format args
        self.assertEqual(args.output_format, 'mp3')
        self.assertEqual(args.bitrate, '256k')
        
        # Check generation control args
        self.assertTrue(args.no_single)
        self.assertEqual(args.batch_runs, 4)
        
        # Check playback args
        self.assertTrue(args.no_play)
        
        # Check listing args
        self.assertTrue(args.list_voices)
    
    def test_validate_generation_args_no_conflicts(self):
        """Test validate_generation_args with no conflicts."""
        # Create a mock args object
        class MockArgs:
            no_single = False
            only_single = False
            no_batch = False
            only_batch = False
        
        args = MockArgs()
        
        # Should not raise an error
        validate_generation_args(args)
    
    def test_validate_generation_args_conflict_no_single_only_single(self):
        """Test validate_generation_args detects no_single/only_single conflict."""
        class MockArgs:
            no_single = True
            only_single = True
            no_batch = False
            only_batch = False
        
        args = MockArgs()
        
        with self.assertRaises(argparse.ArgumentError):
            validate_generation_args(args)
    
    def test_validate_generation_args_conflict_only_single_only_batch(self):
        """Test validate_generation_args detects only_single/only_batch conflict."""
        class MockArgs:
            no_single = False
            only_single = True
            no_batch = False
            only_batch = True
        
        args = MockArgs()
        
        with self.assertRaises(argparse.ArgumentError):
            validate_generation_args(args)
    
    def test_get_generation_modes_defaults(self):
        """Test get_generation_modes with default settings."""
        class MockArgs:
            only_single = False
            only_batch = False
            no_single = False
            no_batch = False
        
        args = MockArgs()
        run_single, run_batch = get_generation_modes(args)
        
        self.assertTrue(run_single)
        self.assertTrue(run_batch)
    
    def test_get_generation_modes_only_single(self):
        """Test get_generation_modes with only_single flag."""
        class MockArgs:
            only_single = True
            only_batch = False
            no_single = False
            no_batch = False
        
        args = MockArgs()
        run_single, run_batch = get_generation_modes(args)
        
        self.assertTrue(run_single)
        self.assertFalse(run_batch)
    
    def test_get_generation_modes_only_batch(self):
        """Test get_generation_modes with only_batch flag."""
        class MockArgs:
            only_single = False
            only_batch = True
            no_single = False
            no_batch = False
        
        args = MockArgs()
        run_single, run_batch = get_generation_modes(args)
        
        self.assertFalse(run_single)
        self.assertTrue(run_batch)
    
    def test_get_generation_modes_no_single(self):
        """Test get_generation_modes with no_single flag."""
        class MockArgs:
            only_single = False
            only_batch = False
            no_single = True
            no_batch = False
        
        args = MockArgs()
        run_single, run_batch = get_generation_modes(args)
        
        self.assertFalse(run_single)
        self.assertTrue(run_batch)
    
    def test_create_standard_parser(self):
        """Test create_standard_parser creates complete parser."""
        description = "Test standard parser"
        script_name = "test_script.py"
        
        parser = create_standard_parser(
            description=description,
            script_name=script_name,
            profiles=self.test_profiles,
            default_profile=self.default_profile,
            default_batch_runs=2
        )
        
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Test parsing complete argument set
        args = parser.parse_args([
            '--voice', 'Profile2',
            '--output-format', 'mp3',
            '--batch-runs', '3',
            '--no-play'
        ])
        
        self.assertEqual(args.voice, 'Profile2')
        self.assertEqual(args.output_format, 'mp3')
        self.assertEqual(args.batch_runs, 3)
        self.assertTrue(args.no_play)


if __name__ == '__main__':
    unittest.main()