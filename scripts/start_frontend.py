"""Convenience runner script to launch the FEVER-X Streamlit UI."""

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    print("Launching FEVER-X Streamlit Web Dashboard on http://localhost:8501 ...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "frontend" / "app.py"),
    ]
    subprocess.run(cmd)
