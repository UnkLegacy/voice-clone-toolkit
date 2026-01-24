"""
Unit tests for utils/file_utils.py

Tests file operations and text processing utilities.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from src.utils.file_utils import (
    load_text_from_file_or_string,
    ensure_directory_exists,
    get_safe_filename,
    get_unique_filepath,
    read_text_file,
    write_text_file,
    get_file_info,
    find_files,
    copy_file,
    get_relative_path,
    validate_file_exists,
    get_file_extension
)


class TestFileUtilities(unittest.TestCase):
    """Test file operations utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        self.test_content = "This is test content."
        
        # Create test file
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_text_from_file_or_string_existing_file(self):
        """Test load_text_from_file_or_string with existing file."""
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_text_from_file_or_string(self.test_file)
        
        self.assertEqual(result, self.test_content)
    
    def test_load_text_from_file_or_string_nonexistent_file(self):
        """Test load_text_from_file_or_string with non-existent file path."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.txt")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_text_from_file_or_string(nonexistent_path)
        
        # Should return the path as literal text
        self.assertEqual(result, nonexistent_path)
    
    def test_load_text_from_file_or_string_literal_text(self):
        """Test load_text_from_file_or_string with literal text."""
        literal_text = "This is literal text, not a file path."
        
        result = load_text_from_file_or_string(literal_text)
        
        self.assertEqual(result, literal_text)
    
    def test_load_text_from_file_or_string_list(self):
        """Test load_text_from_file_or_string with list input."""
        test_list = [self.test_file, "literal text", "another literal"]
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_text_from_file_or_string(test_list)
        
        expected = [self.test_content, "literal text", "another literal"]
        self.assertEqual(result, expected)
    
    def test_load_text_from_file_or_string_file_error(self):
        """Test load_text_from_file_or_string with file read error."""
        # Create file and then make it unreadable
        error_file = os.path.join(self.temp_dir, "error.txt")
        with open(error_file, 'w') as f:
            f.write("content")
        
        # Mock open to raise an exception
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('sys.stdout'):  # Suppress progress messages
                result = load_text_from_file_or_string(error_file)
        
        # Should return the path as literal text when read fails
        self.assertEqual(result, error_file)
    
    def test_ensure_directory_exists_new_directory(self):
        """Test ensure_directory_exists creates new directory."""
        new_dir = os.path.join(self.temp_dir, "new_directory")
        self.assertFalse(os.path.exists(new_dir))
        
        result = ensure_directory_exists(new_dir)
        
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))
        self.assertEqual(result, Path(new_dir))
    
    def test_ensure_directory_exists_existing_directory(self):
        """Test ensure_directory_exists with existing directory."""
        result = ensure_directory_exists(self.temp_dir)
        
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(result, Path(self.temp_dir))
    
    def test_get_safe_filename_invalid_chars(self):
        """Test get_safe_filename removes invalid characters."""
        unsafe_name = 'test<>:"/\\|?*file.txt'
        
        result = get_safe_filename(unsafe_name)
        
        # Count the actual invalid characters: < > : " / \ | ? * = 9 characters
        self.assertEqual(result, "test_________file.txt")
    
    def test_get_safe_filename_control_chars(self):
        """Test get_safe_filename removes control characters."""
        name_with_control = "test\x00\x01\x1ffile.txt"
        
        result = get_safe_filename(name_with_control)
        
        self.assertEqual(result, "testfile.txt")
    
    def test_get_safe_filename_too_long(self):
        """Test get_safe_filename truncates long names."""
        long_name = "a" * 300 + ".txt"
        
        result = get_safe_filename(long_name, max_length=50)
        
        self.assertEqual(len(result), 50)
        self.assertTrue(result.endswith(".txt"))
    
    def test_get_safe_filename_empty_result(self):
        """Test get_safe_filename with name that becomes empty."""
        result = get_safe_filename("...")
        
        self.assertEqual(result, "unnamed")
    
    def test_get_unique_filepath_no_collision(self):
        """Test get_unique_filepath when no file exists."""
        base_path = os.path.join(self.temp_dir, "unique_file")
        
        result = get_unique_filepath(base_path, "txt")
        
        expected = Path(os.path.join(self.temp_dir, "unique_file.txt"))
        self.assertEqual(result, expected)
    
    def test_get_unique_filepath_with_collision(self):
        """Test get_unique_filepath when file already exists."""
        base_path = os.path.join(self.temp_dir, "existing")
        
        # Create existing file
        existing_file = base_path + ".txt"
        with open(existing_file, 'w') as f:
            f.write("existing")
        
        result = get_unique_filepath(base_path, "txt")
        
        expected = Path(os.path.join(self.temp_dir, "existing_1.txt"))
        self.assertEqual(result, expected)
    
    def test_get_unique_filepath_max_attempts(self):
        """Test get_unique_filepath reaches max attempts."""
        base_path = os.path.join(self.temp_dir, "test")
        
        # Create many existing files
        for i in range(5):
            suffix = "" if i == 0 else f"_{i}"
            filepath = f"{base_path}{suffix}.txt"
            with open(filepath, 'w') as f:
                f.write("test")
        
        with self.assertRaises(RuntimeError):
            get_unique_filepath(base_path, "txt", max_attempts=3)
    
    def test_read_text_file_success(self):
        """Test read_text_file with successful read."""
        result = read_text_file(self.test_file)
        
        self.assertEqual(result, self.test_content)
    
    def test_read_text_file_nonexistent(self):
        """Test read_text_file with non-existent file."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = read_text_file(nonexistent)
        
        self.assertIsNone(result)
    
    def test_read_text_file_no_strip(self):
        """Test read_text_file without stripping whitespace."""
        whitespace_content = "  content with whitespace  \n"
        whitespace_file = os.path.join(self.temp_dir, "whitespace.txt")
        
        with open(whitespace_file, 'w') as f:
            f.write(whitespace_content)
        
        result = read_text_file(whitespace_file, strip_whitespace=False)
        
        self.assertEqual(result, whitespace_content)
    
    def test_write_text_file_success(self):
        """Test write_text_file with successful write."""
        output_file = os.path.join(self.temp_dir, "output.txt")
        content = "Test output content"
        
        result = write_text_file(output_file, content)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, 'r') as f:
            written_content = f.read()
        self.assertEqual(written_content, content)
    
    def test_write_text_file_create_dirs(self):
        """Test write_text_file creates parent directories."""
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "file.txt")
        content = "Test content"
        
        result = write_text_file(nested_path, content, create_dirs=True)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(nested_path))
    
    def test_write_text_file_no_create_dirs(self):
        """Test write_text_file without creating directories."""
        nested_path = os.path.join(self.temp_dir, "nonexistent", "file.txt")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = write_text_file(nested_path, "content", create_dirs=False)
        
        self.assertFalse(result)
    
    def test_get_file_info_existing_file(self):
        """Test get_file_info with existing file."""
        result = get_file_info(self.test_file)
        
        self.assertNotIn("error", result)
        self.assertEqual(result["name"], "test.txt")
        self.assertEqual(result["stem"], "test")
        self.assertEqual(result["suffix"], ".txt")
        self.assertTrue(result["is_file"])
        self.assertFalse(result["is_dir"])
        self.assertGreater(result["size_bytes"], 0)
    
    def test_get_file_info_nonexistent_file(self):
        """Test get_file_info with non-existent file."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")
        
        result = get_file_info(nonexistent)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "File not found")
    
    def test_find_files_basic_pattern(self):
        """Test find_files with basic pattern."""
        # Create additional test files
        txt_file = os.path.join(self.temp_dir, "another.txt")
        py_file = os.path.join(self.temp_dir, "script.py")
        
        with open(txt_file, 'w') as f:
            f.write("txt content")
        with open(py_file, 'w') as f:
            f.write("python content")
        
        result = find_files(self.temp_dir, "*.txt")
        
        txt_files = [f for f in result if f.suffix == '.txt']
        self.assertEqual(len(txt_files), 2)  # test.txt and another.txt
    
    def test_find_files_recursive(self):
        """Test find_files with recursive search."""
        # Create nested directory with files
        nested_dir = os.path.join(self.temp_dir, "nested")
        os.makedirs(nested_dir)
        nested_file = os.path.join(nested_dir, "nested.txt")
        
        with open(nested_file, 'w') as f:
            f.write("nested content")
        
        result = find_files(self.temp_dir, "*.txt", recursive=True)
        
        txt_files = [f for f in result if f.suffix == '.txt']
        self.assertGreaterEqual(len(txt_files), 2)  # At least test.txt and nested.txt
    
    def test_find_files_nonexistent_directory(self):
        """Test find_files with non-existent directory."""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = find_files(nonexistent_dir, "*.txt")
        
        self.assertEqual(result, [])
    
    def test_copy_file_success(self):
        """Test copy_file with successful copy."""
        dest_file = os.path.join(self.temp_dir, "copy.txt")
        
        result = copy_file(self.test_file, dest_file)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(dest_file))
        
        with open(dest_file, 'r') as f:
            copied_content = f.read()
        self.assertEqual(copied_content, self.test_content)
    
    def test_copy_file_source_not_found(self):
        """Test copy_file with non-existent source."""
        nonexistent_source = os.path.join(self.temp_dir, "nonexistent.txt")
        dest_file = os.path.join(self.temp_dir, "dest.txt")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = copy_file(nonexistent_source, dest_file)
        
        self.assertFalse(result)
    
    def test_copy_file_create_dirs(self):
        """Test copy_file creates destination directories."""
        nested_dest = os.path.join(self.temp_dir, "nested", "copy.txt")
        
        result = copy_file(self.test_file, nested_dest, create_dirs=True)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(nested_dest))
    
    def test_get_relative_path_success(self):
        """Test get_relative_path with relative path."""
        base_dir = self.temp_dir
        target_path = self.test_file
        
        result = get_relative_path(target_path, base_dir)
        
        self.assertEqual(result, "test.txt")
    
    def test_get_relative_path_not_relative(self):
        """Test get_relative_path with non-relative path."""
        base_dir = os.path.join(self.temp_dir, "other")
        target_path = self.test_file
        
        result = get_relative_path(target_path, base_dir)
        
        # Should return absolute path when not relative
        self.assertTrue(os.path.isabs(result))
    
    def test_validate_file_exists_success(self):
        """Test validate_file_exists with existing file."""
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_file_exists(self.test_file, "test file")
        
        self.assertTrue(result)
    
    def test_validate_file_exists_not_found(self):
        """Test validate_file_exists with non-existent file."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_file_exists(nonexistent, "test file")
        
        self.assertFalse(result)
    
    def test_validate_file_exists_not_file(self):
        """Test validate_file_exists with directory instead of file."""
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_file_exists(self.temp_dir, "test file")
        
        self.assertFalse(result)
    
    def test_get_file_extension_with_dot(self):
        """Test get_file_extension includes dot by default."""
        result = get_file_extension("test.txt")
        
        self.assertEqual(result, ".txt")
    
    def test_get_file_extension_without_dot(self):
        """Test get_file_extension without dot."""
        result = get_file_extension("test.txt", with_dot=False)
        
        self.assertEqual(result, "txt")
    
    def test_get_file_extension_no_extension(self):
        """Test get_file_extension with file having no extension."""
        result = get_file_extension("README")
        
        self.assertEqual(result, "")
    
    def test_get_file_extension_multiple_dots(self):
        """Test get_file_extension with multiple dots."""
        result = get_file_extension("archive.tar.gz")
        
        self.assertEqual(result, ".gz")


if __name__ == '__main__':
    unittest.main()