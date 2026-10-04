"""Centralized model loader and singleton cache for FEVER-X.

Prevents redundant model reloads across modules and queries, ensuring high performance.
"""

import os
from typing import Any, Dict, Optional, Tuple

# Suppress TensorFlow / Keras 3 incompatibility with HuggingFace transformers
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch

from src.common.exceptions import ModelLoadError
from src.utils.config_loader import load_model_config
from src.utils.logger import get_logger

logger = get_logger("model_loader")

# Global singleton model cache
_MODEL_CACHE: Dict[str, Any] = {}


def resolve_device(configured_device: Optional[str] = None) -> str:
    """Resolves target execution device ('cuda' if available and requested, else 'cpu')."""
    if configured_device:
        target = configured_device.lower()
        if target.startswith("cuda") and torch.cuda.is_available():
            return target
        return "cpu"

    # Default to config
    try:
        cfg = load_model_config()
        configured = cfg.get("runtime", {}).get("device", "cpu").lower()
        if configured.startswith("cuda") and torch.cuda.is_available():
            return configured
    except Exception:
        pass

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_sentence_transformer(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Loads and caches a SentenceTransformer model.

    Args:
        model_name: HuggingFace model checkpoint or path.
        device: 'cpu' or 'cuda'.

    Returns:
        SentenceTransformer instance.
    """
    if not model_name:
        cfg = load_model_config()
        model_name = cfg["models"]["sentence_transformer"]

    dev = resolve_device(device)
    cache_key = f"st_{model_name}_{dev}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    logger.info("Loading SentenceTransformer '%s' on %s...", model_name, dev)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=dev)
        _MODEL_CACHE[cache_key] = model
        logger.info("SentenceTransformer loaded successfully.")
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load SentenceTransformer '{model_name}': {e}") from e


def get_cross_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Loads and caches a CrossEncoder model for neural reranking."""
    if not model_name:
        cfg = load_model_config()
        model_name = cfg["models"]["cross_encoder"]

    dev = resolve_device(device)
    cache_key = f"ce_{model_name}_{dev}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    logger.info("Loading CrossEncoder '%s' on %s...", model_name, dev)
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device=dev)
        _MODEL_CACHE[cache_key] = model
        logger.info("CrossEncoder loaded successfully.")
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load CrossEncoder '{model_name}': {e}") from e


def get_spacy_model(model_name: Optional[str] = None) -> Any:
    """Loads and caches a spaCy NLP pipeline."""
    if not model_name:
        cfg = load_model_config()
        model_name = cfg.get("models", {}).get("spacy_model", "en_core_web_sm")

    cache_key = f"spacy_{model_name}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    logger.info("Loading spaCy pipeline '%s'...", model_name)
    try:
        import spacy

        try:
            nlp = spacy.load(model_name)
        except OSError:
            logger.warning("spaCy model '%s' not installed; falling back to blank:en", model_name)
            nlp = spacy.blank("en")

        _MODEL_CACHE[cache_key] = nlp
        logger.info("spaCy pipeline loaded successfully.")
        return nlp
    except Exception as e:
        raise ModelLoadError(f"Failed to load spaCy model '{model_name}': {e}") from e


def clear_model_cache() -> None:
    """Clears all cached models to reclaim memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared.")
