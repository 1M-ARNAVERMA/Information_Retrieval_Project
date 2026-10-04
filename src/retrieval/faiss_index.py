"""FAISS vector indexing module with metadata tracking and persistence."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.common.exceptions import IndexNotFoundError
from src.utils.file_utils import ensure_dir, load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger("faiss_index")

# Attempt faiss import with fallback
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("FAISS library not found; operating with vectorized NumPy fallback.")


class FaissVectorIndex:
    """FAISS-based dense vector index utilizing Inner Product / Cosine Similarity."""

    def __init__(self, dimension: int = 384, metric: str = "ip") -> None:
        """Initializes dense index.

        Args:
            dimension: Embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2).
            metric: 'ip' (Inner Product / Cosine) or 'l2' (Euclidean).
        """
        self.dimension = dimension
        self.metric = metric.lower()
        self.metadata: List[Dict[str, Any]] = []

        if _FAISS_AVAILABLE:
            if self.metric == "ip":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None
            self._numpy_vectors: Optional[np.ndarray] = None

    @property
    def ntotal(self) -> int:
        """Total vectors currently indexed."""
        if _FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        if self._numpy_vectors is not None:
            return len(self._numpy_vectors)
        return 0

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Adds normalized vector embeddings and corresponding document metadata.

        Args:
            embeddings: Float32 array of shape (N, dimension).
            metadata: List of dicts matching vectors, containing doc_id, sentence_id, text.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings count and metadata length must match.")

        emb_matrix = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Normalize if using inner product for cosine similarity
        if self.metric == "ip":
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            emb_matrix = emb_matrix / norms

        if _FAISS_AVAILABLE and self.index is not None:
            self.index.add(emb_matrix)
        else:
            if self._numpy_vectors is None:
                self._numpy_vectors = emb_matrix
            else:
                self._numpy_vectors = np.vstack([self._numpy_vectors, emb_matrix])

        self.metadata.extend(metadata)
        logger.info("Added %d vectors to index. Total: %d", len(metadata), self.ntotal)

    def search(
        self, query_vector: np.ndarray, top_k: int = 50
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Queries index for top_k nearest neighbors.

        Args:
            query_vector: 1D or 2D array of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of (metadata_dict, score) sorted by score descending.
        """
        if self.ntotal == 0:
            return []

        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        # Normalize query for cosine similarity
        if self.metric == "ip":
            q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
            q_norm[q_norm == 0] = 1e-12
            q_vec = q_vec / q_norm

        k = min(top_k, self.ntotal)

        if _FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(q_vec, k)
            dist_row = distances[0]
            idx_row = indices[0]
        else:
            # NumPy cosine similarity fallback
            scores = np.dot(self._numpy_vectors, q_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:k]
            idx_row = top_indices
            dist_row = scores[top_indices]

        results = []
        for dist, idx in zip(dist_row, idx_row):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(dist)))

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Serializes FAISS binary index and metadata."""
        dir_path = ensure_dir(directory)
        meta_file = dir_path / "faiss_metadata.pkl"
        save_pickle(self.metadata, meta_file)

        if _FAISS_AVAILABLE and self.index is not None:
            bin_file = dir_path / "faiss_index.bin"
            faiss.write_index(self.index, str(bin_file))
        else:
            npy_file = dir_path / "numpy_vectors.npy"
            np.save(str(npy_file), self._numpy_vectors)

        logger.info("Saved vector index (%d vectors) to %s", self.ntotal, dir_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Loads FAISS binary index and metadata."""
        dir_path = Path(directory)
        meta_file = dir_path / "faiss_metadata.pkl"

        if not meta_file.exists():
            raise IndexNotFoundError(f"FAISS metadata missing in {directory}")

        self.metadata = load_pickle(meta_file)

        bin_file = dir_path / "faiss_index.bin"
        npy_file = dir_path / "numpy_vectors.npy"

        if _FAISS_AVAILABLE and bin_file.exists():
            self.index = faiss.read_index(str(bin_file))
            self.dimension = self.index.d
        elif npy_file.exists():
            self._numpy_vectors = np.load(str(npy_file))
            self.dimension = self._numpy_vectors.shape[1]
        else:
            raise IndexNotFoundError(f"No FAISS index binary or NumPy vector file in {directory}")

        logger.info("Loaded vector index (%d vectors) from %s", self.ntotal, dir_path)
