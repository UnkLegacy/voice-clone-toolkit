"""
Import integrity tests to catch import-related issues.

These tests verify that all modules can be imported correctly and that
there are no missing imports or import-related errors.
"""

import unittest
import subprocess
import sys
import ast
import importlib.util
from pathlib import Path


class TestImportIntegrity(unittest.TestCase):
    """Test import integrity across all modules."""
    
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
    
    def test_main_scripts_importable(self):
        """Test that all main scripts can be imported without errors."""
        for script in self.main_scripts:
            script_path = self.src_dir / script
            module_name = script[:-3]  # Remove .py extension
            
            with self.subTest(script=script):
                # Test importability - add src directory to path for fallback imports
                original_path = sys.path[:]
                if str(self.src_dir) not in sys.path:
                    sys.path.insert(0, str(self.src_dir))
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, script_path)
                    self.assertIsNotNone(spec, f"Could not create spec for {script}")
                    
                    module = importlib.util.module_from_spec(spec)
                    
                    # This should not raise ImportError
                    try:
                        spec.loader.exec_module(module)
                    except ImportError as e:
                        self.fail(f"ImportError in {script}: {e}")
                    except Exception as e:
                        # Other exceptions might be expected (like missing config files)
                        # but ImportError specifically indicates missing imports
                        pass
                finally:
                    # Restore original sys.path
                    sys.path[:] = original_path
    
    def test_utility_modules_importable(self):
        """Test that all utility modules can be imported without errors."""
        for util_module in self.utility_modules:
            util_path = self.src_dir / util_module
            module_name = util_module.replace("/", ".").replace(".py", "")
            
            with self.subTest(module=util_module):
                spec = importlib.util.spec_from_file_location(module_name, util_path)
                self.assertIsNotNone(spec, f"Could not create spec for {util_module}")
                
                module = importlib.util.module_from_spec(spec)
                
                try:
                    spec.loader.exec_module(module)
                except ImportError as e:
                    self.fail(f"ImportError in {util_module}: {e}")
    
    def test_scripts_help_command(self):
        """Test that scripts can be executed directly with --help."""
        for script in self.main_scripts:
            script_path = self.src_dir / script
            
            with self.subTest(script=script):
                try:
                    # Run script with --help to test import integrity
                    result = subprocess.run(
                        [sys.executable, str(script_path), "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=self.src_dir.parent  # Run from project root
                    )
                    
                    # Should exit successfully (code 0) when showing help
                    if result.returncode != 0:
                        self.fail(f"{script} failed with --help:\nStderr: {result.stderr}\nStdout: {result.stdout}")
                        
                except subprocess.TimeoutExpired:
                    self.fail(f"{script} timed out with --help command")
                except Exception as e:
                    self.fail(f"Error running {script} --help: {e}")
    
    def test_no_duplicate_imports(self):
        """Test that modules don't have duplicate imports."""
        for script in self.main_scripts + self.utility_modules:
            script_path = self.src_dir / script
            
            with self.subTest(script=script):
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse the AST to find imports
                    tree = ast.parse(content)
                    imports = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(f"import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for alias in node.names:
                                imports.append(f"from {module} import {alias.name}")
                    
                    # Check for duplicates
                    import_counts = {}
                    for imp in imports:
                        import_counts[imp] = import_counts.get(imp, 0) + 1
                    
                    # Filter out expected hybrid import patterns (exactly 2 occurrences)
                    # These are intentional try/except fallback imports
                    expected_hybrid_patterns = [
                        "from .import_helper import get_utils",
                        "from import_helper import get_utils"
                    ]
                    
                    duplicates = {}
                    for imp, count in import_counts.items():
                        if count > 1:
                            # Check if this is a hybrid import pattern with exactly 2 occurrences
                            is_hybrid = any(pattern in imp for pattern in expected_hybrid_patterns)
                            if is_hybrid and count == 2:
                                # This is expected - try/except pattern
                                continue
                            else:
                                duplicates[imp] = count
                    
                    if duplicates:
                        self.fail(f"Duplicate imports found in {script}: {duplicates}")
                        
                except Exception as e:
                    self.fail(f"Error analyzing imports in {script}: {e}")
    
    def test_undefined_names_in_main_scripts(self):
        """Test for common undefined name patterns that indicate missing imports."""
        undefined_patterns = {
            'torch': ['torch.device', 'torch.cuda', 'torch.bfloat16'],
            'wave': ['wave.open'],
            'playsound': ['playsound('],
            'pygame': ['pygame.mixer', 'pygame.init'],
            'pydub': ['AudioSegment'],
            'json': ['json.load', 'json.dump'],
            'argparse': ['ArgumentParser', 'argparse.ArgumentParser'],
        }
        
        for script in self.main_scripts:
            script_path = self.src_dir / script
            
            with self.subTest(script=script):
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse imports to see what's available
                    tree = ast.parse(content)
                    imported_modules = set()
                    imported_names = set()
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imported_modules.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imported_modules.add(node.module.split('.')[0])
                            for alias in node.names:
                                imported_names.add(alias.name)
                    
                    # Check for usage of undefined patterns
                    for module, patterns in undefined_patterns.items():
                        if module not in imported_modules:
                            for pattern in patterns:
                                if pattern in content and not any(name in pattern for name in imported_names):
                                    # Special handling for utility imports
                                    if 'utils.' not in content or f'from utils.{module}' not in content:
                                        self.fail(f"Found usage of '{pattern}' but '{module}' not imported in {script}")
                        
                except Exception as e:
                    self.fail(f"Error analyzing undefined names in {script}: {e}")


if __name__ == '__main__':
    unittest.main()