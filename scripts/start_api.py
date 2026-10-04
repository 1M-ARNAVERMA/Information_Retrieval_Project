"""Convenience runner script to launch the FEVER-X FastAPI backend."""

from pathlib import Path
import sys
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    print("Starting FEVER-X FastAPI REST Microservice on http://localhost:8000 ...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
