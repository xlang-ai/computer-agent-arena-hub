import os
import sys
from pathlib import Path

def setup_import_paths():
    """Setup import paths for both standalone and submodule usage"""
    current_dir = Path(__file__).parent
    
    # Add the agents directory to path for both scenarios
    if current_dir not in sys.path:
        sys.path.append(str(current_dir))
    
    # Check if we're in backend submodule
    backend_dir = current_dir.parent
    if backend_dir.name == 'backend':
        if str(backend_dir) not in sys.path:
            sys.path.append(str(backend_dir))