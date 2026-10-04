"""Stopword management module with negation preservation for fact verification."""

from typing import Iterable, List, Optional, Set

# Negation words critical for fact verification that should NOT be removed by default
NEGATION_WORDS: Set[str] = {
    "no",
    "not",
    "nor",
    "neither",
    "never",
    "none",
    "nobody",
    "nowhere",
    "nothing",
    "cannot",
    "without",
    "hardly",
    "scarcely",
    "barely",
}

# Standard English stopwords (comprehensive base set)
BASE_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}


class StopwordFilter:
    """Configurable stopword filter with negation protection."""

    def __init__(
        self,
        preserve_negations: bool = True,
        extra_stopwords: Optional[Iterable[str]] = None,
        exclude_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes the stopword filter.

        Args:
            preserve_negations: If True, retains negation terms (e.g. 'not', 'never').
            extra_stopwords: Custom words to add to the stopword set.
            exclude_stopwords: Words to remove from the stopword set.
        """
        self.preserve_negations = preserve_negations
        self._stopwords = set(BASE_STOPWORDS)

        if preserve_negations:
            self._stopwords.difference_update(NEGATION_WORDS)

        if extra_stopwords:
            self._stopwords.update(w.lower() for w in extra_stopwords)

        if exclude_stopwords:
            self._stopwords.difference_update(w.lower() for w in exclude_stopwords)

    @property
    def stopwords(self) -> Set[str]:
        """Returns the active stopword set."""
        return self._stopwords

    def is_stopword(self, token: str) -> bool:
        """Checks if a token is a stopword."""
        return token.lower() in self._stopwords

    def filter(self, tokens: List[str]) -> List[str]:
        """Filters stopwords out of a token list while preserving word order."""
        return [t for t in tokens if not self.is_stopword(t)]
