"""Executable script to verify arbitrary claims from the terminal."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default run with sample claim if no arguments provided
        sys.argv.extend(["verify", "Albert Einstein was a German physicist."])
    main()
