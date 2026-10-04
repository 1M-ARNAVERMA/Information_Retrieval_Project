"""ColBERT / Late-Interaction Token-Level Retrieval module.

Implements Contextualized Late Interaction (MaxSim) with graceful CPU fallback.
Designed to be completely modular so that missing GPU or ColBERT binaries
never prevents the rest of the verification system from running.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.common.exceptions import IndexNotFoundError
from src.common.types import Document, PreprocessMode, RetrievalResult
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.retrieval.base import BaseRetriever
from src.utils.config_loader import load_config, load_model_config
from src.utils.file_utils import ensure_dir, load_pickle, save_pickle
from src.utils.logger import get_logger
from src.utils.model_loader import resolve_device

logger = get_logger("colbert_retriever")


def compute_maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Computes the ColBERT MaxSim operator between query and document token embeddings.

    Formula:
        Score(Q, D) = sum_{i in |Q|} max_{j in |D|} (E_{Q,i} . E_{D,j})

    Args:
        query_tokens: Array of shape (|Q|, dim), L2-normalized.
        doc_tokens: Array of shape (|D|, dim), L2-normalized.

    Returns:
        MaxSim scalar interaction score.
    """
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0

    # Dot product similarity matrix: shape (|Q|, |D|)
    sim_matrix = np.dot(query_tokens, doc_tokens.T)

    # Max over document tokens for each query token, then sum
    max_per_query_token = np.max(sim_matrix, axis=1)
    return float(np.sum(max_per_query_token))


class ColbertRetriever(BaseRetriever):
    """ColBERT token-level late interaction retriever with MaxSim operator."""

    def __init__(
        self,
        checkpoint: Optional[str] = "fallback",
        device: Optional[str] = None,
        embedding_dim: int = 128,
        default_top_k: int = 50,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.checkpoint = checkpoint or "fallback"
        self.device = resolve_device(device)
        self.embedding_dim = embedding_dim
        self.default_top_k = default_top_k
        self.preprocessor = preprocessor or get_preprocessor()

        self._encoder = None
        self._tokenizer = None
        # Document representations: List of token embedding arrays of shape (|D_i|, dim)
        self.doc_embeddings: List[np.ndarray] = []
        self.documents: List[Document] = []

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if ColBERT or HuggingFace transformers is usable."""
        try:
            import transformers
            return True
        except ImportError:
            return False

    def _init_encoder(self) -> None:
        """Initializes token encoder if not already instantiated."""
        if self._encoder is not None:
            return

        # If fallback requested or explicitly offline, use fast deterministic token hasher
        if self.checkpoint == "fallback":
            self._encoder = None
            self._tokenizer = None
            return

        if not self.checkpoint:
            try:
                cfg = load_model_config()
                self.checkpoint = cfg.get("models", {}).get(
                    "colbert_checkpoint", "fallback"
                )
            except Exception:
                self.checkpoint = "fallback"

        if self.checkpoint == "fallback":
            self._encoder = None
            self._tokenizer = None
            return

        logger.info("Initializing ColBERT encoder on %s...", self.device)
        try:
            from transformers import AutoModel, AutoTokenizer

            # Attempt local files first to avoid network stall
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
                self._encoder = AutoModel.from_pretrained(self.checkpoint, local_files_only=True).to(self.device)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
                self._encoder = AutoModel.from_pretrained(self.checkpoint).to(self.device)

            self._encoder.eval()
            logger.info("ColBERT late-interaction encoder ready.")
        except Exception as e:
            logger.warning("Could not initialize transformer for ColBERT (%s); using token hasher fallback.", e)
            self._encoder = None
            self._tokenizer = None

    def _encode_tokens(self, text: str) -> np.ndarray:
        """Generates L2-normalized token embeddings for text."""
        self._init_encoder()

        if self._encoder is not None and self._tokenizer is not None:
            try:
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._encoder(**inputs)
                    # Last hidden states: (1, seq_len, hidden_dim)
                    hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

                # Project to embedding_dim if needed, then normalize
                if hidden.shape[1] > self.embedding_dim:
                    hidden = hidden[:, : self.embedding_dim]

                norms = np.linalg.norm(hidden, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                return hidden / norms
            except Exception as e:
                logger.warning("Transformer token encoding failed: %s; falling back to hash", e)

        # Fallback pseudo-embedding generator (deterministic token hashing for CPU testing)
        words = text.split()
        if not words:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        arr = np.zeros((len(words), self.embedding_dim), dtype=np.float32)
        for i, word in enumerate(words):
            h = hash(word.lower())
            np.random.seed(abs(h) % (2**31))
            vec = np.random.randn(self.embedding_dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            arr[i] = vec / (norm if norm > 0 else 1.0)
        return arr

    def index(self, documents: List[Document]) -> None:
        """Encodes all documents into multi-vector token arrays.

        Args:
            documents: List of Document models.
        """
        if not documents:
            logger.warning("Empty document list provided for ColBERT indexing.")
            return

        logger.info("Indexing %d documents with ColBERT multi-vectors...", len(documents))
        self.documents = list(documents)
        self.doc_embeddings = []

        for doc in self.documents:
            # Semantic mode retains punctuation/casing for token context
            clean_text = self.preprocessor.preprocess(doc.text, mode=PreprocessMode.SEMANTIC)
            token_embs = self._encode_tokens(clean_text)
            self.doc_embeddings.append(token_embs)

        logger.info("ColBERT indexing complete. %d document token matrices built.", len(self.doc_embeddings))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves top_k scoring documents via token-level MaxSim late interaction.

        Args:
            query: Claim or search query string.
            top_k: Max candidates to return.

        Returns:
            List of RetrievalResult models sorted by MaxSim score descending.
        """
        if not self.documents or not self.doc_embeddings:
            logger.warning("ColBERT queried before index was built or loaded.")
            return []

        k = top_k or self.default_top_k
        clean_q = self.preprocessor.preprocess(query, mode=PreprocessMode.SEMANTIC)
        if not clean_q.strip():
            return []

        # Encode query into token vectors: shape (|Q|, dim)
        query_embs = self._encode_tokens(clean_q)

        scores: List[Tuple[float, int]] = []
        for idx, doc_embs in enumerate(self.doc_embeddings):
            sim = compute_maxsim(query_embs, doc_embs)
            scores.append((sim, idx))

        # Sort by MaxSim score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:k]

        results: List[RetrievalResult] = []
        for score, idx in top_scores:
            doc = self.documents[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    sentence_id=doc.sentence_id,
                    text=doc.text,
                    score=round(score, 6),
                    retriever="colbert",
                    metadata={"query_processed": clean_q, "maxsim_score": round(score, 6)},
                )
            )

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persists ColBERT document multi-vectors and catalog to disk."""
        dir_path = ensure_dir(directory)
        save_pickle(self.doc_embeddings, dir_path / "colbert_embeddings.pkl")
        save_pickle(self.documents, dir_path / "colbert_documents.pkl")
        logger.info("ColBERT index saved to %s", dir_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Loads ColBERT document multi-vectors and catalog from disk."""
        dir_path = Path(directory)
        emb_file = dir_path / "colbert_embeddings.pkl"
        doc_file = dir_path / "colbert_documents.pkl"

        if not (emb_file.exists() and doc_file.exists()):
            raise IndexNotFoundError(f"ColBERT index files missing in {directory}")

        self.doc_embeddings = load_pickle(emb_file)
        self.documents = load_pickle(doc_file)
        logger.info("ColBERT index loaded from %s (%d documents)", dir_path, len(self.documents))
