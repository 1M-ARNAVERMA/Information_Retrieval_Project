"""Stemming module wrapping PorterStemmer with fallback."""

from typing import List

try:
    from nltk.stem import PorterStemmer as NltkPorterStemmer
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


class Stemmer:
    """Configurable stemmer for reducing words to their root stems."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._stemmer = NltkPorterStemmer() if _NLTK_AVAILABLE else None

    def stem_word(self, word: str) -> str:
        """Stems a single token. Returns lowercased original if disabled."""
        if not self.enabled or not word:
            return word

        if self._stemmer:
            return self._stemmer.stem(word.lower())

        # Simple heuristic fallback if NLTK stemmer is unavailable
        w = word.lower()
        for suffix in ("ing", "ies", "ed", "ly", "es", "s"):
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                return w[: -len(suffix)]
        return w

    def stem(self, tokens: List[str]) -> List[str]:
        """Stems an iterable of tokens."""
        if not self.enabled:
            return tokens
        return [self.stem_word(t) for t in tokens]
