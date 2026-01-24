"""
Code quality tests to catch structural and duplication issues.

These tests verify that the codebase follows good practices and
doesn't have problematic duplication or structural issues.
"""

import unittest
import ast
import re
from pathlib import Path
from collections import defaultdict


class TestCodeQuality(unittest.TestCase):
    """Test code quality and structural issues."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.src_dir = Path(__file__).parent.parent / "src"
        self.main_scripts = [
            "clone_voice.py",
            "clone_voice_conversation.py", 
            "custom_voice.py",
            "voice_design.py",
            "voice_design_clone.py",
            "convert_audio_format.py"
        ]
        self.utility_modules = [
            "utils/progress.py",
            "utils/audio_utils.py",
            "utils/cli_args.py", 
            "utils/config_loader.py",
            "utils/model_utils.py",
            "utils/file_utils.py"
        ]
        self.all_files = self.main_scripts + self.utility_modules
    
    def test_no_duplicate_function_definitions(self):
        """Test that function names aren't duplicated across main scripts."""
        function_definitions = defaultdict(list)
        
        for file_path in self.main_scripts:
            full_path = self.src_dir / file_path
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private functions (starting with _)
                        if not node.name.startswith('_'):
                            function_definitions[node.name].append(file_path)
                            
            except Exception as e:
                self.fail(f"Error analyzing {file_path}: {e}")
        
        # Check for functions that appear in multiple main scripts
        duplicates = {name: files for name, files in function_definitions.items() 
                     if len(files) > 1}
        
        # Some functions are expected to be duplicated
        allowed_duplicates = {'main', 'parse_args'}  # These are expected in each script
        
        problematic_duplicates = {name: files for name, files in duplicates.items() 
                                if name not in allowed_duplicates}
        
        if problematic_duplicates:
            self.fail(f"Found duplicate function definitions that should be in utils: {problematic_duplicates}")
    
    def test_utility_functions_not_redefined(self):
        """Test that main scripts don't redefine utility functions."""
        # Get all utility function names
        utility_functions = set()
        
        for util_file in self.utility_modules:
            full_path = self.src_dir / util_file
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # Skip private functions
                            utility_functions.add(node.name)
                            
            except Exception:
                continue
        
        # Check main scripts for redefinitions
        for script in self.main_scripts:
            full_path = self.src_dir / script
            
            with self.subTest(script=script):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name in utility_functions:
                                # Check if it's importing the utility version
                                imports_utility = (
                                    f'from .utils.' in content or 
                                    f'from utils.' in content
                                )
                                
                                if imports_utility:
                                    self.fail(f"{script} redefines utility function '{node.name}' "
                                             f"but also imports from utils. This suggests the local "
                                             f"definition should be removed.")
                                    
                except Exception as e:
                    self.fail(f"Error analyzing {script}: {e}")
    
    def test_consistent_import_style(self):
        """Test that imports follow consistent style."""
        for file_path in self.all_files:
            full_path = self.src_dir / file_path
            
            with self.subTest(file=file_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Check for mixed relative/absolute imports to utils
                    has_relative_utils = any('from .utils.' in line for line in lines)
                    has_absolute_utils = any(re.match(r'^\s*from utils\.', line) for line in lines)
                    
                    # This is actually expected in our hybrid import system
                    # but we should make sure they're in the try/except pattern
                    if has_relative_utils and has_absolute_utils:
                        # This should only happen in main scripts with hybrid imports
                        if file_path in self.main_scripts:
                            # Check that they're in try/except blocks
                            content = ''.join(lines)
                            if 'except ImportError:' not in content:
                                self.fail(f"{file_path} has mixed import styles but no ImportError handling")
                        else:
                            # Utility modules shouldn't have mixed imports
                            self.fail(f"Utility module {file_path} has mixed import styles")
                            
                except Exception as e:
                    self.fail(f"Error analyzing {file_path}: {e}")
    
    def test_no_large_code_duplication(self):
        """Test for significant code duplication between files."""
        file_contents = {}
        
        # Read all files
        for file_path in self.main_scripts:
            full_path = self.src_dir / file_path
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    # Get non-empty, non-comment lines
                    lines = [
                        line.strip() 
                        for line in f.readlines() 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    file_contents[file_path] = lines
            except Exception:
                continue
        
        # Look for duplicated code blocks (sequences of 5+ lines)
        MIN_DUPLICATE_LINES = 5
        
        for file1, lines1 in file_contents.items():
            for file2, lines2 in file_contents.items():
                if file1 >= file2:  # Avoid duplicate comparisons
                    continue
                
                with self.subTest(file1=file1, file2=file2):
                    # Find common subsequences
                    for i in range(len(lines1) - MIN_DUPLICATE_LINES):
                        block1 = lines1[i:i + MIN_DUPLICATE_LINES]
                        
                        for j in range(len(lines2) - MIN_DUPLICATE_LINES):
                            block2 = lines2[j:j + MIN_DUPLICATE_LINES]
                            
                            if block1 == block2:
                                # Found duplicate block - check if it's acceptable
                                block_str = '\n'.join(block1)
                                
                                # Skip common patterns that are OK to duplicate
                                acceptable_patterns = [
                                    'import ',
                                    'from ',
                                    'def main():',
                                    'if __name__ == "__main__":',
                                    'try:',
                                    'except Exception as e:',
                                    'print_progress(',
                                ]
                                
                                is_acceptable = any(pattern in block_str for pattern in acceptable_patterns)
                                
                                if not is_acceptable:
                                    self.fail(f"Found significant code duplication between "
                                             f"{file1}:{i+1} and {file2}:{j+1}:\n{block_str}")
    
    def test_proper_utility_usage(self):
        """Test that main scripts use utility functions appropriately."""
        # Functions that should come from utils
        utility_functions = {
            'print_progress': 'utils.progress',
            'print_error': 'utils.progress', 
            'save_audio': 'utils.audio_utils',
            'play_audio': 'utils.audio_utils',
            'load_voice_clone_profiles': 'utils.config_loader',
            'create_base_parser': 'utils.cli_args',
        }
        
        for script in self.main_scripts:
            full_path = self.src_dir / script
            
            with self.subTest(script=script):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for func_name, expected_module in utility_functions.items():
                        if f'{func_name}(' in content:
                            # Check that it's imported from the right place
                            import_pattern = f'from .{expected_module} import'
                            fallback_pattern = f'from {expected_module} import'
                            
                            if not (import_pattern in content or fallback_pattern in content):
                                # Maybe it's imported as part of a larger import
                                if func_name not in content or f'def {func_name}(' in content:
                                    self.fail(f"{script} uses {func_name} but doesn't import it from {expected_module}")
                                    
                except Exception as e:
                    self.fail(f"Error analyzing {script}: {e}")


if __name__ == '__main__':
    unittest.main()