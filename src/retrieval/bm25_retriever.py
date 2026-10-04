"""BM25 (Okapi) lexical retrieval module using rank-bm25."""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from rank_bm25 import BM25Okapi

from src.common.exceptions import IndexNotFoundError
from src.common.types import Document, PreprocessMode, RetrievalResult
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.retrieval.base import BaseRetriever
from src.utils.file_utils import ensure_dir, load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger("bm25_retriever")


class BM25Retriever(BaseRetriever):
    """Lexical retriever utilizing Okapi BM25 ranking algorithm."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        preprocessor: Optional[TextPreprocessor] = None,
        default_top_k: int = 50,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.preprocessor = preprocessor or get_preprocessor()
        self.default_top_k = default_top_k

        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []

    def index(self, documents: List[Document]) -> None:
        """Builds BM25 inverted index from tokenized documents.

        Args:
            documents: List of Document instances.
        """
        if not documents:
            logger.warning("Empty document list provided for BM25 indexing.")
            return

        logger.info("Indexing %d documents into BM25 (Okapi)...", len(documents))
        self.documents = list(documents)

        # Tokenize all documents in LEXICAL mode
        tokenized_corpus = [
            self.preprocessor.tokenize(doc.text, mode=PreprocessMode.LEXICAL)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index successfully constructed.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves top_k scoring candidates using BM25 score.

        Args:
            query: Claim or query string.
            top_k: Max candidates to return.

        Returns:
            List of RetrievalResult instances sorted by score descending.
        """
        if self.bm25 is None or not self.documents:
            logger.warning("BM25 retriever queried before index was built or loaded.")
            return []

        k = top_k or self.default_top_k
        query_tokens = self.preprocessor.tokenize(query, mode=PreprocessMode.LEXICAL)
        if not query_tokens:
            return []

        # Get scores across all indexed documents
        scores = self.bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        # Sort top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        results: List[RetrievalResult] = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    sentence_id=doc.sentence_id,
                    text=doc.text,
                    score=round(score, 6),
                    retriever="bm25",
                    metadata={"query_tokens": query_tokens},
                )
            )

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Saves BM25 index and documents to directory."""
        dir_path = ensure_dir(directory)
        save_pickle(self.bm25, dir_path / "bm25_index.pkl")
        save_pickle(self.documents, dir_path / "bm25_documents.pkl")
        logger.info("BM25 index saved to %s", dir_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Loads BM25 index and documents from directory."""
        dir_path = Path(directory)
        idx_path = dir_path / "bm25_index.pkl"
        doc_path = dir_path / "bm25_documents.pkl"

        if not (idx_path.exists() and doc_path.exists()):
            raise IndexNotFoundError(f"BM25 index files missing in {directory}")

        self.bm25 = load_pickle(idx_path)
        self.documents = load_pickle(doc_path)
        logger.info("BM25 index loaded from %s (%d documents)", dir_path, len(self.documents))
