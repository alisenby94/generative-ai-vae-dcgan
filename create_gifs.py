#!/usr/bin/env python3
"""
Convenience wrapper for creating training GIFs
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tools.create_training_gifs import main

if __name__ == '__main__':
    main()
