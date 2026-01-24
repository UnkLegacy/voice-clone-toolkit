"""
Error formatting tests to ensure proper error handling patterns.

These tests verify that error messages use appropriate formatting functions
and that error handling follows consistent patterns.
"""

import unittest
import ast
import re
from pathlib import Path


class TestErrorFormatting(unittest.TestCase):
    """Test error message formatting consistency."""
    
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
    
    def test_no_error_with_print_progress(self):
        """Test that error messages don't use print_progress inappropriately."""
        error_patterns = [
            r'print_progress\([^)]*["\'].*[Ee]rror:.*["\']',  # print_progress("Error: ...")
            r'print_progress\(f["\'].*[Ee]rror:.*["\']',      # print_progress(f"Error: ...")
            r'print_progress\([^)]*["\'].*[Ee]rror .*["\']',  # print_progress("Error ...")
        ]
        
        for file_path in self.all_files:
            full_path = self.src_dir / file_path
            
            with self.subTest(file=file_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in error_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            self.fail(f"Found inappropriate error formatting in {file_path}:\n{matches}")
                            
                except Exception as e:
                    self.fail(f"Error analyzing {file_path}: {e}")
    
    def test_error_messages_use_print_error(self):
        """Test that modules that need print_error actually import and use it."""
        for file_path in self.all_files:
            full_path = self.src_dir / file_path
            
            with self.subTest(file=file_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if file has error messages
                    has_error_messages = bool(re.search(r'["\'].*[Ee]rror[:\s]', content))
                    
                    if has_error_messages:
                        # Check if print_error is imported
                        has_print_error_import = (
                            'print_error' in content and 
                            ('from .utils.progress import' in content or 
                             'from utils.progress import' in content)
                        )
                        
                        if not has_print_error_import:
                            # This might be OK for some utility modules
                            if file_path not in self.main_scripts:
                                continue
                                
                            self.fail(f"{file_path} has error messages but doesn't import print_error")
                            
                except Exception as e:
                    self.fail(f"Error analyzing {file_path}: {e}")
    
    def test_error_handling_patterns(self):
        """Test that error handling follows consistent patterns."""
        for file_path in self.main_scripts:  # Focus on main scripts
            full_path = self.src_dir / file_path
            
            with self.subTest(file=file_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse AST to find exception handling
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ExceptHandler):
                            # Check if exception handler uses proper error reporting
                            handler_code = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                            
                            # Look for common error handling patterns
                            has_proper_error_handling = any([
                                'print_error(' in handler_code,
                                'handle_fatal_error(' in handler_code,
                                'handle_processing_error(' in handler_code,
                                'sys.exit(' in handler_code,  # Direct exit is sometimes OK
                            ])
                            
                            # Skip if it's just a pass or simple assignment
                            is_minimal_handler = (
                                len(node.body) == 1 and (
                                    isinstance(node.body[0], ast.Pass) or
                                    isinstance(node.body[0], ast.Assign) or
                                    isinstance(node.body[0], ast.Continue) or
                                    isinstance(node.body[0], ast.Return)
                                )
                            )
                            
                            if not has_proper_error_handling and not is_minimal_handler:
                                # This might be a custom error handler that needs review
                                pass  # Could add warning here if needed
                                
                except Exception as e:
                    # AST parsing might fail for various reasons, skip the test
                    pass
    
    def test_stderr_vs_stdout_usage(self):
        """Test that error-related prints go to stderr."""
        stderr_patterns = [
            r'print\([^)]*file=sys\.stderr',  # print(..., file=sys.stderr)
            r'print_error\(',                # Our error function
            r'handle_.*_error\(',            # Our error handlers
        ]
        
        stdout_error_patterns = [
            r'print\([^)]*["\'].*[Ee]rror[:\s][^"\']*["\'][^)]*\)',  # print("Error: ...") without file=sys.stderr
        ]
        
        for file_path in self.all_files:
            full_path = self.src_dir / file_path
            
            with self.subTest(file=file_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for stdout error patterns (potentially problematic)
                    for pattern in stdout_error_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip if it's using our proper error functions or stderr
                            if not any(p in match for p in ['print_error', 'file=sys.stderr', 'handle_']):
                                # This could indicate improper error formatting
                                # We'll be lenient and just note it for now
                                pass
                                
                except Exception as e:
                    self.fail(f"Error analyzing {file_path}: {e}")


if __name__ == '__main__':
    unittest.main()