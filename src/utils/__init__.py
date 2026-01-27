"""
Shared utilities for Voice Clone Toolkit

This package contains common utilities used across all voice generation scripts,
including audio processing, configuration loading, model management, and more.
"""

# Import commonly used utilities for easy access
from .progress import (
    print_progress, print_error, print_warning, 
    handle_fatal_error, handle_processing_error, 
    handle_mp3_conversion_error, handle_audio_playback_error,
    with_error_handling
)
from .audio_utils import (
    save_audio, play_audio, save_wav, ensure_output_dir, get_audio_info,
    normalize_audio, adjust_volume
)
from .file_utils import (
    load_text_from_file_or_string, ensure_directory_exists, validate_file_exists,
    get_safe_filename, get_unique_filepath, read_text_file, write_text_file,
    get_file_info, find_files, copy_file, get_relative_path, get_file_extension
)
from .config_loader import (
    load_json_config, load_voice_clone_profiles, load_custom_voice_profiles,
    load_voice_design_profiles, load_voice_design_clone_profiles, load_conversation_scripts,
    load_custom_voices_by_type,
    get_profile_choices, get_default_profile, validate_profile_structure
)
from .model_utils import (
    load_model_with_device, get_device, load_voice_clone_model,
    load_custom_voice_model, load_voice_design_model,
    get_model_memory_usage, clear_gpu_cache, get_device_info, validate_model_path,
    fix_pad_token_id
)
from .cli_args import (
    create_standard_parser, create_base_parser, add_common_args, get_generation_modes,
    add_audio_format_args, add_generation_control_args, add_playback_args,
    add_profile_listing_args, add_voice_selection_args, add_multi_voice_selection_args,
    validate_generation_args
)

__all__ = [
    # Progress and logging
    'print_progress', 'print_error', 'print_warning',
    'handle_fatal_error', 'handle_processing_error', 
    'handle_mp3_conversion_error', 'handle_audio_playback_error',
    'with_error_handling',
    
    # Audio utilities
    'save_audio', 'play_audio', 'save_wav', 'ensure_output_dir', 'get_audio_info',
    'normalize_audio', 'adjust_volume',
    
    # File utilities
    'load_text_from_file_or_string', 'ensure_directory_exists', 'validate_file_exists',
    'get_safe_filename', 'get_unique_filepath', 'read_text_file', 'write_text_file',
    'get_file_info', 'find_files', 'copy_file', 'get_relative_path', 'get_file_extension',
    
    # Configuration loading
    'load_json_config', 'load_voice_clone_profiles', 'load_custom_voice_profiles',
    'load_voice_design_profiles', 'load_voice_design_clone_profiles', 'load_conversation_scripts',
    'load_custom_voices_by_type',
    'get_profile_choices', 'get_default_profile', 'validate_profile_structure',
    
    # Model utilities
    'load_model_with_device', 'get_device', 'load_voice_clone_model',
    'load_custom_voice_model', 'load_voice_design_model',
    'get_model_memory_usage', 'clear_gpu_cache', 'get_device_info', 'validate_model_path',
    'fix_pad_token_id',
    
    # CLI utilities
    'create_standard_parser', 'create_base_parser', 'add_common_args', 'get_generation_modes',
    'add_audio_format_args', 'add_generation_control_args', 'add_playback_args',
    'add_profile_listing_args', 'add_voice_selection_args', 'add_multi_voice_selection_args',
    'validate_generation_args',
]