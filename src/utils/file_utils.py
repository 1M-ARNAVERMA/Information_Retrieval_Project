"""File and directory utility functions for FEVER-X."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating parents if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """Yields dictionaries line by line from a JSONL file.

    Args:
        file_path: Path to the .jsonl file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num} of {file_path}: {e}")


def write_jsonl(file_path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> int:
    """Writes an iterable of dictionaries to a JSONL file.

    Args:
        file_path: Target path for the .jsonl file.
        records: Iterable of serializable dictionaries.

    Returns:
        Total number of lines written.
    """
    path = Path(file_path)
    ensure_dir(path.parent)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Serializes an object using pickle.

    Args:
        obj: Python object to persist.
        file_path: Output file path.
    """
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Deserializes an object from a pickle file.

    Args:
        file_path: Path to serialized file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file does not exist: {file_path}")
    with open(path, "rb") as f:
        return pickle.load(f)
