"""Tokenization utilities for words and sentences."""

import re
from typing import List


# Regex pattern to match words, numbers, and contractions
WORD_TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:'[A-Za-z]+)?\b|[^\w\s]")
WORD_ONLY_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:'[A-Za-z]+)?\b")


class Tokenizer:
    """Configurable word and sentence tokenizer."""

    def __init__(self, strip_punctuation: bool = False) -> None:
        self.strip_punctuation = strip_punctuation
        self._pattern = WORD_ONLY_PATTERN if strip_punctuation else WORD_TOKEN_PATTERN

    def tokenize(self, text: str) -> List[str]:
        """Splits a string into a list of word/punctuation tokens.

        Args:
            text: Input string.

        Returns:
            List of tokens.
        """
        if not text:
            return []
        return self._pattern.findall(text)

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Splits paragraph into individual sentences based on punctuation."""
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
