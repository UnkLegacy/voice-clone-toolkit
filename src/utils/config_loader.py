"""
Configuration loading utilities.

This module provides JSON configuration loading functionality with error handling,
default config creation, and text field processing for all voice generation scripts.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Callable

from .progress import print_progress, print_error
from .file_utils import load_text_from_file_or_string


def load_json_config(config_path: str, 
                    create_default: bool = False,
                    default_factory: Optional[Callable[[], Dict[str, Any]]] = None,
                    error_message_prefix: str = "config") -> Dict[str, Any]:
    """
    Load a JSON configuration file with error handling.
    
    Args:
        config_path: Path to the JSON config file
        create_default: Whether to create a default config if file doesn't exist
        default_factory: Function that returns default config structure
        error_message_prefix: Prefix for error messages (e.g., "voice profiles", "config")
        
    Returns:
        Dictionary of configuration data
    """
    if not os.path.exists(config_path):
        if create_default and default_factory:
            print_error(f"{error_message_prefix} not found: {config_path}")
            print_progress("Creating default config file...")
            
            # Create default config
            default_config = default_factory()
            
            # Ensure config directory exists
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            print_progress(f"Created default config at: {config_path}")
            print_progress(f"Please edit this file to add your {error_message_prefix}.")
            return default_config
        else:
            print_error(f"{error_message_prefix} not found: {config_path}")
            return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Error parsing JSON {error_message_prefix}: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error loading {error_message_prefix}: {e}")
        sys.exit(1)


def process_text_fields(profiles: Dict[str, Any], 
                       text_fields: List[str]) -> Dict[str, Any]:
    """
    Process text fields in profiles to load from files if specified.
    
    Args:
        profiles: Dictionary of profiles
        text_fields: List of field names that can contain file paths or text
        
    Returns:
        Updated profiles dictionary with processed text fields
    """
    for profile_name, profile in profiles.items():
        for field in text_fields:
            if field in profile:
                profile[field] = load_text_from_file_or_string(profile[field])
    
    return profiles


def create_default_voice_clone_profiles() -> Dict[str, Any]:
    """Create default voice clone profiles structure."""
    return {
        "Example": {
            "voice_sample_file": "./input/example.wav",
            "sample_transcript": "This is an example reference text.",
            "single_text": "This is a single generation example.",
            "batch_texts": [
                "This is the first batch text.",
                "This is the second batch text."
            ]
        }
    }


def create_default_custom_voice_profiles() -> Dict[str, Any]:
    """Create default custom voice profiles structure."""
    return {
        "Ryan": {
            "speaker": "Ryan",
            "language": "English",
            "description": "Dynamic male voice with strong rhythmic drive",
            "single_text": "Hello everyone, this is a test of the custom voice system.",
            "single_instruct": "",
            "batch_texts": [
                "This is the first example.",
                "This is the second example."
            ],
            "batch_languages": ["English", "English"],
            "batch_instructs": ["", "Happy"]
        }
    }


def create_default_voice_design_profiles() -> Dict[str, Any]:
    """Create default voice design profiles structure."""
    return {
        "Example": {
            "instruct": "A clear, friendly voice",
            "language": "English",
            "single_text": "Hello everyone, this is a test of voice design.",
            "single_instruct": "",
            "batch_texts": [
                "This is the first example.",
                "This is the second example."
            ],
            "batch_languages": ["English", "English"],
            "batch_instructs": ["", "Happy"]
        }
    }


def create_default_voice_design_clone_profiles() -> Dict[str, Any]:
    """Create default voice design + clone profiles structure."""
    return {
        "Example": {
            "instruct": "A clear, friendly voice",
            "language": "English",
            "single_text": "Hello everyone, this is a test of voice design cloning.",
            "single_instruct": "",
            "batch_texts": [
                "This is the first example.",
                "This is the second example."
            ],
            "batch_languages": ["English", "English"],
            "batch_instructs": ["", "Happy"]
        }
    }


def load_voice_clone_profiles(config_path: str) -> Dict[str, Any]:
    """
    Load voice clone profiles with default creation and text processing.
    
    Args:
        config_path: Path to the voice clone profiles config file
        
    Returns:
        Dictionary of voice clone profiles
    """
    profiles = load_json_config(
        config_path=config_path,
        create_default=True,
        default_factory=create_default_voice_clone_profiles,
        error_message_prefix="Voice profiles config"
    )
    
    # Process text fields that can be loaded from files
    text_fields = ['single_text', 'batch_texts', 'sample_transcript']
    return process_text_fields(profiles, text_fields)


def load_custom_voice_profiles(config_path: str) -> Dict[str, Any]:
    """
    Load custom voice profiles.
    
    Args:
        config_path: Path to the custom voice profiles config file
        
    Returns:
        Dictionary of custom voice profiles
    """
    return load_json_config(
        config_path=config_path,
        create_default=True,
        default_factory=create_default_custom_voice_profiles,
        error_message_prefix="Custom voice profiles config"
    )


def load_voice_design_profiles(config_path: str) -> Dict[str, Any]:
    """
    Load voice design profiles.
    
    Args:
        config_path: Path to the voice design profiles config file
        
    Returns:
        Dictionary of voice design profiles
    """
    return load_json_config(
        config_path=config_path,
        create_default=True,
        default_factory=create_default_voice_design_profiles,
        error_message_prefix="Voice design profiles config"
    )


def load_voice_design_clone_profiles(config_path: str) -> Dict[str, Any]:
    """
    Load voice design + clone profiles.
    
    Args:
        config_path: Path to the voice design clone profiles config file
        
    Returns:
        Dictionary of voice design clone profiles
    """
    return load_json_config(
        config_path=config_path,
        create_default=True,
        default_factory=create_default_voice_design_clone_profiles,
        error_message_prefix="Voice design clone profiles config"
    )


def load_conversation_scripts(config_path: str) -> Dict[str, Any]:
    """
    Load conversation scripts configuration.
    
    Args:
        config_path: Path to the conversation scripts config file
        
    Returns:
        Dictionary of conversation scripts
    """
    return load_json_config(
        config_path=config_path,
        create_default=False,
        error_message_prefix="Conversation scripts config"
    )


def load_custom_voices_by_type(profile_type: str) -> Dict[str, Any]:
    """
    Load custom voices from custom/custom_voices.json filtered by profile type.
    
    Args:
        profile_type: The profile type to filter by ('voice_clone', 'custom_voice', 
                     'voice_design', 'voice_design_clone')
        
    Returns:
        Dictionary of custom voice profiles matching the specified type
    """
    CUSTOM_VOICES_CONFIG = "custom/custom_voices.json"
    if not os.path.exists(CUSTOM_VOICES_CONFIG):
        return {}
    
    try:
        custom_voices = load_json_config(CUSTOM_VOICES_CONFIG)
        # Filter by profile_type and remove the profile_type field from the profile data
        filtered_profiles = {}
        for name, profile in custom_voices.items():
            if profile.get('profile_type') == profile_type:
                # Create a copy without the profile_type field (it's metadata, not part of the profile structure)
                profile_copy = {k: v for k, v in profile.items() if k != 'profile_type'}
                filtered_profiles[name] = profile_copy
        return filtered_profiles
    except Exception as e:
        print_error(f"Error loading custom voices: {e}")
        return {}


def validate_profile_structure(profile: Dict[str, Any], 
                             required_fields: List[str],
                             profile_name: str = "Profile") -> bool:
    """
    Validate that a profile has all required fields.
    
    Args:
        profile: Profile dictionary to validate
        required_fields: List of required field names
        profile_name: Name of profile for error messages
        
    Returns:
        True if valid, False otherwise
    """
    missing_fields = [field for field in required_fields if field not in profile]
    
    if missing_fields:
        print_error(f"{profile_name} is missing required fields: {missing_fields}")
        return False
    
    return True


def get_profile_choices(profiles: Dict[str, Any]) -> List[str]:
    """
    Get a list of available profile names for argument choices.
    
    Args:
        profiles: Dictionary of profiles
        
    Returns:
        List of profile names
    """
    return list(profiles.keys())


def get_default_profile(profiles: Dict[str, Any], 
                       preferred_default: Optional[str] = None) -> Optional[str]:
    """
    Get the default profile name from available profiles.
    
    Args:
        profiles: Dictionary of profiles
        preferred_default: Preferred default profile name
        
    Returns:
        Default profile name, or None if no profiles available
    """
    if not profiles:
        return None
    
    # Use preferred default if it exists
    if preferred_default and preferred_default in profiles:
        return preferred_default
    
    # Otherwise use first available profile
    return next(iter(profiles.keys()))