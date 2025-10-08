#!/usr/bin/env python3
"""
Preprocessing script for MovieLens dataset.
Standalone script that can be run from command line.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import main

if __name__ == "__main__":
    main()

