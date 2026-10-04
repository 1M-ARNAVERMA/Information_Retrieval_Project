"""Unit tests for Phase 3: Text Preprocessing module."""

import pytest

from src.common.types import PreprocessMode
from src.preprocessing.cleaner import clean_text, expand_contractions
from src.preprocessing.pipeline import TextPreprocessor, preprocess
from src.preprocessing.stemmer import Stemmer
from src.preprocessing.stopwords import StopwordFilter
from src.preprocessing.tokenizer import Tokenizer


def test_cleaner_empty_and_spaces():
    """Validates cleaner handles None, empty string, and excess spaces."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   hello   world   \t \n") == "hello world"


def test_cleaner_html_and_unicode():
    """Validates unescaping HTML and unicode NFKC normalization."""
    raw = "Einstein &amp; Bohr &lt;physics&gt;"
    assert clean_text(raw) == "Einstein & Bohr <physics>"


def test_expand_contractions():
    """Validates contraction expansion."""
    raw = "He didn't know that they haven't arrived."
    expanded = expand_contractions(raw)
    assert "did not" in expanded
    assert "have not" in expanded


def test_tokenizer_words_and_punctuation():
    """Validates tokenization of words and punctuation marks."""
    tok = Tokenizer(strip_punctuation=False)
    tokens = tok.tokenize("Hello, World! 123.")
    assert "Hello" in tokens
    assert "," in tokens
    assert "World" in tokens
    assert "!" in tokens

    tok_no_punct = Tokenizer(strip_punctuation=True)
    tokens_no_punct = tok_no_punct.tokenize("Hello, World! 123.")
    assert "," not in tokens_no_punct
    assert "!" not in tokens_no_punct
    assert "Hello" in tokens_no_punct


def test_tokenizer_split_sentences():
    """Validates splitting paragraphs into sentences."""
    para = "First sentence. Second sentence! Third sentence? Yes."
    sents = Tokenizer.split_sentences(para)
    assert len(sents) == 4
    assert sents[0] == "First sentence."
    assert sents[1] == "Second sentence!"


def test_stopword_filter_and_negations():
    """Validates stopword filtering with negation preservation."""
    sw = StopwordFilter(preserve_negations=True)
    tokens = ["this", "is", "not", "a", "good", "claim"]
    filtered = sw.filter(tokens)

    # 'this', 'is', 'a' should be removed; 'not' and 'good' and 'claim' retained
    assert "not" in filtered
    assert "good" in filtered
    assert "claim" in filtered
    assert "is" not in filtered
    assert "this" not in filtered

    # When preserve_negations=False, 'not' should be removed
    sw_no_neg = StopwordFilter(preserve_negations=False)
    filtered_no_neg = sw_no_neg.filter(tokens)
    assert "not" not in filtered_no_neg


def test_stemmer():
    """Validates stemming functionality."""
    stemmer = Stemmer(enabled=True)
    assert stemmer.stem_word("developing") in ("develop", "developi")
    assert stemmer.stem_word("relativity") in ("relativ", "relativity", "rel")
    assert stemmer.stem_word("universities") == "univers"

    stems = stemmer.stem(["physicists", "developed", "theories"])
    assert len(stems) == 3
    assert stems[0] in ("physicist", "physic")


def test_pipeline_modes():
    """Validates the 3 preprocessing modes: LEXICAL, SEMANTIC, RAW."""
    processor = TextPreprocessor()
    text = "Albert Einstein didn't develop this in France!"

    # 1. SEMANTIC mode (Preserves casing, punctuation, contractions for Transformers)
    sem = processor.preprocess(text, mode=PreprocessMode.SEMANTIC)
    assert sem == "Albert Einstein didn't develop this in France!"

    # 2. RAW mode (Whitespace cleanup only)
    raw = processor.preprocess("  " + text + "  \n", mode=PreprocessMode.RAW)
    assert raw == text

    # 3. LEXICAL mode (Lowercased, contractions expanded, stopwords removed, stemmed)
    lex = processor.preprocess(text, mode=PreprocessMode.LEXICAL)
    # Casing should be gone
    assert "Albert" not in lex
    assert "albert" in lex or "einstein" in lex
    # Negation "not" must be preserved for fact verification
    assert "not" in lex
    # Exclamation mark should be stripped
    assert "!" not in lex

    # Test tokenized output
    tokens = processor.tokenize(text, mode=PreprocessMode.LEXICAL)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_convenience_function():
    """Validates the module-level preprocess() function."""
    res = preprocess("Testing module preprocess helper", mode="lexical")
    assert isinstance(res, str)
    assert len(res) > 0
