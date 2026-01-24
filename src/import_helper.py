"""
Import helper for Voice Clone Toolkit.

This module provides a single utility function to handle the 
relative/absolute import logic without duplication across scripts.
"""

import sys
from pathlib import Path


def get_utils():
    """
    Import utility modules with fallback for direct execution.
    
    This function handles the hybrid import pattern needed to support
    both package execution (python -m src.script) and direct execution
    (python src/script.py) without duplicate imports.
    
    Returns:
        utils module: The loaded utils module
    """
    try:
        # Try relative import first (when run as package)
        from . import utils
        return utils
    except ImportError:
        # Fallback to absolute import (when run directly)
        sys.path.insert(0, str(Path(__file__).parent))
        import utils
        return utils