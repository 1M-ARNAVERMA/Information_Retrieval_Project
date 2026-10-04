"""Custom exceptions for the FEVER-X project."""


class FeverXException(Exception):
    """Base exception for all FEVER-X errors."""
    pass


class ConfigurationError(FeverXException):
    """Raised when configuration files or environment variables are missing/invalid."""
    pass


class DataIngestionError(FeverXException):
    """Raised when dataset loading, parsing, or normalization fails."""
    pass


class IndexNotFoundError(FeverXException):
    """Raised when an index (TF-IDF, BM25, FAISS) is expected but not found on disk."""
    pass


class ModelLoadError(FeverXException):
    """Raised when a transformer model, spaCy pipeline, or checkpoint fails to load."""
    pass


class InferenceError(FeverXException):
    """Raised when model inference or pipeline execution encounters a runtime error."""
    pass


class PipelineError(FeverXException):
    """Raised when end-to-end pipeline execution encounters an invalid state or failure."""
    pass

