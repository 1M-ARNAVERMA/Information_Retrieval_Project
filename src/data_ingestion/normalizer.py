"""Text and Wikipedia title normalization routines for FEVER data."""

import re
import urllib.parse
from typing import Optional


FEVER_TOKEN_REPLACEMENTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
}


def normalize_fever_text(text: str) -> str:
    """Replaces FEVER-specific escaped brackets and formatting artifacts.

    Examples:
        'Albert Einstein -LRB- 1879 -RRB-' -> 'Albert Einstein ( 1879 )'
    """
    if not text:
        return ""

    for token, replacement in FEVER_TOKEN_REPLACEMENTS.items():
        text = text.replace(token, replacement)

    # Clean double spaces or excess whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_wiki_title(title: str) -> str:
    """Normalizes a Wikipedia document title.

    Replaces underscores with spaces, decodes URL percent-encoding,
    and replaces FEVER bracket tokens.

    Examples:
        'Albert_Einstein' -> 'Albert Einstein'
        'Paris_-LRB-disambiguation-RRB-' -> 'Paris (disambiguation)'
    """
    if not title:
        return ""

    # Decode URL quoting if present
    title = urllib.parse.unquote(title)

    # Replace FEVER bracket tokens
    for token, replacement in FEVER_TOKEN_REPLACEMENTS.items():
        title = title.replace(token, replacement)

    # Convert underscores to spaces
    title = title.replace("_", " ")

    return re.sub(r"\s+", " ", title).strip()
