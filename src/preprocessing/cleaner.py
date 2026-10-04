"""Text cleaning and normalization module for FEVER-X."""

import html
import re
import unicodedata
from typing import Optional


# Common contractions expansion mapping
CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expands English contractions into their constituent words."""
    def _replace(match):
        match_str = match.group(0).lower()
        expanded = CONTRACTIONS.get(match_str, match.group(0))
        # Match original casing for first letter
        if match.group(0)[0].isupper():
            return expanded.capitalize()
        return expanded

    return _CONTRACTION_PATTERN.sub(_replace, text)


def clean_text(
    text: Optional[str],
    expand_contract: bool = False,
    strip_punctuation: bool = False,
    to_lower: bool = False,
) -> str:
    """Sanitizes text by unescaping HTML, normalizing unicode, and handling whitespace.

    Args:
        text: Raw input text.
        expand_contract: Whether to expand contractions ('did not' instead of "didn't").
        strip_punctuation: Whether to remove all non-alphanumeric punctuation.
        to_lower: Whether to cast characters to lowercase.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)

    # Normalize unicode to standard form (NFKC removes full-width, ligature anomalies)
    text = unicodedata.normalize("NFKC", text)

    # Expand contractions if requested
    if expand_contract:
        text = expand_contractions(text)

    # Remove or retain punctuation
    if strip_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)

    # Lowercase if requested
    if to_lower:
        text = text.lower()

    # Collapse excess whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
