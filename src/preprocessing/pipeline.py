"""Unified text preprocessing pipeline supporting LEXICAL, SEMANTIC, and RAW modes."""

from typing import Dict, List, Optional, Union

from src.common.types import PreprocessMode
from src.preprocessing.cleaner import clean_text
from src.preprocessing.stemmer import Stemmer
from src.preprocessing.stopwords import StopwordFilter
from src.preprocessing.tokenizer import Tokenizer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("text_preprocessor")


class TextPreprocessor:
    """Preprocesses text for either lexical (TF-IDF/BM25) or neural semantic pipelines."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        preserve_negations: bool = True,
    ) -> None:
        """Initializes components using config or defaults."""
        if config is None:
            try:
                full_cfg = load_config()
                self.config = full_cfg.get("preprocessing", {})
            except Exception:
                self.config = {}
        else:
            self.config = config

        # Lexical configuration flags
        lex_cfg = self.config.get("lexical", {})
        self.lex_lower = lex_cfg.get("lowercase", True)
        self.lex_strip_punct = lex_cfg.get("remove_punctuation", True)
        self.lex_remove_stopwords = lex_cfg.get("remove_stopwords", True)
        self.lex_stemming = lex_cfg.get("stemming", True)

        # Components
        self.tokenizer = Tokenizer(strip_punctuation=False)
        self.stopword_filter = StopwordFilter(preserve_negations=preserve_negations)
        self.stemmer = Stemmer(enabled=self.lex_stemming)

    def preprocess(
        self,
        text: Optional[str],
        mode: Union[str, PreprocessMode] = PreprocessMode.LEXICAL,
    ) -> str:
        """Transforms text into a cleaned string representation based on the selected mode.

        Args:
            text: Input claim or evidence sentence.
            mode: PreprocessMode.LEXICAL | PreprocessMode.SEMANTIC | PreprocessMode.RAW

        Returns:
            Processed string.
        """
        if not text:
            return ""

        if isinstance(mode, str):
            mode = PreprocessMode(mode.lower())

        if mode == PreprocessMode.RAW:
            return clean_text(text, expand_contract=False, strip_punctuation=False, to_lower=False)

        if mode == PreprocessMode.SEMANTIC:
            # Preserve capitalization, contractions, and punctuation for Transformer models
            return clean_text(text, expand_contract=False, strip_punctuation=False, to_lower=False)

        # Lexical Mode (for TF-IDF, BM25)
        # 1. Expand contractions so 'didn't' -> 'did not'
        cleaned = clean_text(
            text,
            expand_contract=True,
            strip_punctuation=self.lex_strip_punct,
            to_lower=self.lex_lower,
        )

        # 2. Tokenize
        tokens = self.tokenizer.tokenize(cleaned)

        # 3. Filter stopwords (negations preserved)
        if self.lex_remove_stopwords:
            tokens = self.stopword_filter.filter(tokens)

        # 4. Stem
        if self.lex_stemming:
            tokens = self.stemmer.stem(tokens)

        return " ".join(tokens)

    def tokenize(
        self,
        text: Optional[str],
        mode: Union[str, PreprocessMode] = PreprocessMode.LEXICAL,
    ) -> List[str]:
        """Returns a list of processed tokens for indexing or BM25 corpus building.

        Args:
            text: Input string.
            mode: PreprocessMode

        Returns:
            List of string tokens.
        """
        processed_str = self.preprocess(text, mode=mode)
        if not processed_str:
            return []
        return processed_str.split()


# Singleton default preprocessor instance for convenience
_default_preprocessor: Optional[TextPreprocessor] = None


def get_preprocessor() -> TextPreprocessor:
    """Retrieves or initializes the global default TextPreprocessor."""
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()
    return _default_preprocessor


def preprocess(
    text: Optional[str],
    mode: Union[str, PreprocessMode] = PreprocessMode.LEXICAL,
) -> str:
    """Functional convenience wrapper for text preprocessing."""
    return get_preprocessor().preprocess(text, mode=mode)
