"""
File operations and text processing utilities.

This module provides file handling utilities for loading text from files,
processing file paths, and other file operations used across voice generation scripts.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

from .progress import print_progress


def load_text_from_file_or_string(value: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Load text from a file if the value is a file path, otherwise return the value as-is.
    Supports both single strings and lists of strings.
    
    Args:
        value: Either a string (text or file path) or a list of strings
        
    Returns:
        The text content (either from file or original value)
    """
    if isinstance(value, list):
        # Process each item in the list recursively
        return [load_text_from_file_or_string(item) for item in value]
    
    if isinstance(value, str):
        # Check if it looks like a file path and exists
        if os.path.exists(value) and os.path.isfile(value):
            try:
                with open(value, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                print_progress(f"Loaded text from file: {value}")
                return content
            except Exception as e:
                print_progress(f"Warning: Could not read file '{value}': {e}")
                print_progress(f"Using value as literal text instead.")
                return value
        else:
            # Not a file path or doesn't exist, treat as literal text
            return value
    
    return value


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Convert a string into a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum length for the filename
        
    Returns:
        Safe filename string
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    safe_filename = filename
    
    for char in invalid_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Remove control characters
    safe_filename = ''.join(char for char in safe_filename if ord(char) >= 32)
    
    # Trim whitespace and dots from ends
    safe_filename = safe_filename.strip('. ')
    
    # Ensure it's not too long
    if len(safe_filename) > max_length:
        name, ext = os.path.splitext(safe_filename)
        max_name_length = max_length - len(ext)
        safe_filename = name[:max_name_length] + ext
    
    # Ensure it's not empty
    if not safe_filename:
        safe_filename = "unnamed"
    
    return safe_filename


def get_unique_filepath(base_path: Union[str, Path], 
                       extension: str = "",
                       max_attempts: int = 1000) -> Path:
    """
    Get a unique file path by adding numbers if the file already exists.
    
    Args:
        base_path: Base file path (without extension)
        extension: File extension (with or without dot)
        max_attempts: Maximum number of attempts to find a unique name
        
    Returns:
        Unique Path object
        
    Raises:
        RuntimeError: If unable to find a unique filename within max_attempts
    """
    # Ensure extension starts with dot
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    base_path = Path(base_path)
    
    # Try the original path first
    full_path = base_path.with_suffix(extension)
    if not full_path.exists():
        return full_path
    
    # Try numbered variations
    for i in range(1, max_attempts + 1):
        numbered_path = base_path.with_name(f"{base_path.stem}_{i}{extension}")
        if not numbered_path.exists():
            return numbered_path
    
    raise RuntimeError(f"Could not find unique filename after {max_attempts} attempts")


def read_text_file(filepath: Union[str, Path], 
                   encoding: str = 'utf-8',
                   strip_whitespace: bool = True) -> Optional[str]:
    """
    Read a text file with error handling.
    
    Args:
        filepath: Path to the text file
        encoding: Text encoding to use
        strip_whitespace: Whether to strip leading/trailing whitespace
        
    Returns:
        File content as string, or None if error
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
        
        if strip_whitespace:
            content = content.strip()
        
        return content
        
    except Exception as e:
        print_progress(f"Error reading file '{filepath}': {e}")
        return None


def write_text_file(filepath: Union[str, Path], 
                   content: str,
                   encoding: str = 'utf-8',
                   create_dirs: bool = True) -> bool:
    """
    Write content to a text file with error handling.
    
    Args:
        filepath: Path to the text file
        content: Content to write
        encoding: Text encoding to use
        create_dirs: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print_progress(f"Error writing file '{filepath}': {e}")
        return False


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(filepath)
    
    if not path.exists():
        return {"error": "File not found", "path": str(path)}
    
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
    except Exception as e:
        return {"error": f"Could not get file info: {e}", "path": str(path)}


def find_files(directory: Union[str, Path], 
               pattern: str = "*",
               recursive: bool = False) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: File pattern to match (e.g., "*.txt", "audio.*")
        recursive: Whether to search subdirectories
        
    Returns:
        List of Path objects for matching files
    """
    path = Path(directory)
    
    if not path.exists() or not path.is_dir():
        print_progress(f"Directory not found: {directory}")
        return []
    
    try:
        if recursive:
            return list(path.rglob(pattern))
        else:
            return list(path.glob(pattern))
    except Exception as e:
        print_progress(f"Error searching directory '{directory}': {e}")
        return []


def copy_file(source: Union[str, Path], 
              destination: Union[str, Path],
              create_dirs: bool = True) -> bool:
    """
    Copy a file with error handling.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        True if successful, False otherwise
    """
    import shutil
    
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            print_progress(f"Source file not found: {source}")
            return False
        
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        return True
        
    except Exception as e:
        print_progress(f"Error copying file '{source}' to '{destination}': {e}")
        return False


def get_relative_path(path: Union[str, Path], 
                     base: Union[str, Path]) -> str:
    """
    Get a relative path from base directory.
    
    Args:
        path: Target path
        base: Base directory
        
    Returns:
        Relative path as string
    """
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        # If not relative, return absolute path
        return str(Path(path).resolve())


def validate_file_exists(filepath: Union[str, Path], 
                        file_type: str = "file") -> bool:
    """
    Validate that a file exists and print error if not.
    
    Args:
        filepath: Path to validate
        file_type: Description of file type for error messages
        
    Returns:
        True if file exists, False otherwise
    """
    path = Path(filepath)
    
    if not path.exists():
        print_progress(f"Error: {file_type} not found: {filepath}")
        return False
    
    if not path.is_file():
        print_progress(f"Error: Path is not a file: {filepath}")
        return False
    
    return True


def get_file_extension(filepath: Union[str, Path], 
                      with_dot: bool = True) -> str:
    """
    Get file extension from a path.
    
    Args:
        filepath: Path to get extension from
        with_dot: Whether to include the dot in the extension
        
    Returns:
        File extension string
    """
    path = Path(filepath)
    extension = path.suffix
    
    if not with_dot and extension.startswith('.'):
        extension = extension[1:]
    
    return extension