"""TF-IDF lexical retrieval module using Scikit-Learn."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.exceptions import IndexNotFoundError
from src.common.types import Document, PreprocessMode, RetrievalResult
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.retrieval.base import BaseRetriever
from src.utils.config_loader import load_config
from src.utils.file_utils import ensure_dir, load_pickle, save_pickle
from src.utils.logger import get_logger

logger = get_logger("tfidf_retriever")


class TfidfRetriever(BaseRetriever):
    """Lexical retriever utilizing TF-IDF vectorization with cosine similarity."""

    def __init__(
        self,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        preprocessor: Optional[TextPreprocessor] = None,
        default_top_k: int = 50,
    ) -> None:
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.preprocessor = preprocessor or get_preprocessor()
        self.default_top_k = default_top_k

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None
        self.documents: List[Document] = []

    def index(self, documents: List[Document]) -> None:
        """Builds TF-IDF vocabulary and document term matrix from documents.

        Args:
            documents: List of Document instances.
        """
        if not documents:
            logger.warning("Empty document list provided for TF-IDF indexing.")
            return

        logger.info("Indexing %d documents into TF-IDF vectorizer...", len(documents))
        self.documents = list(documents)

        # Preprocess all document texts in LEXICAL mode
        corpus_texts = [
            self.preprocessor.preprocess(doc.text, mode=PreprocessMode.LEXICAL)
            for doc in self.documents
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"\S+",  # Already tokenized and joined by spaces
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        logger.info("TF-IDF matrix built with shape: %s", self.tfidf_matrix.shape)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves top_k scoring candidates for a query.

        Args:
            query: Claim or search text.
            top_k: Max candidates to return. Defaults to config or 50.

        Returns:
            List of RetrievalResult instances.
        """
        if self.vectorizer is None or self.tfidf_matrix is None or not self.documents:
            logger.warning("TF-IDF retriever queried before index was built or loaded.")
            return []

        k = top_k or self.default_top_k
        clean_q = self.preprocessor.preprocess(query, mode=PreprocessMode.LEXICAL)
        if not clean_q.strip():
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([clean_q])

        # Dot product with normalized TF-IDF vectors gives cosine similarity
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        if len(scores) == 0:
            return []

        # Find top indices
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
                    retriever="tfidf",
                    metadata={"query_processed": clean_q},
                )
            )

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Saves vectorizer, matrix, and documents list to directory."""
        dir_path = ensure_dir(directory)
        save_pickle(self.vectorizer, dir_path / "tfidf_vectorizer.pkl")
        save_pickle(self.tfidf_matrix, dir_path / "tfidf_matrix.pkl")
        save_pickle(self.documents, dir_path / "tfidf_documents.pkl")
        logger.info("TF-IDF index saved to %s", dir_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Loads vectorizer, matrix, and documents list from directory."""
        dir_path = Path(directory)
        vec_path = dir_path / "tfidf_vectorizer.pkl"
        mat_path = dir_path / "tfidf_matrix.pkl"
        doc_path = dir_path / "tfidf_documents.pkl"

        if not (vec_path.exists() and mat_path.exists() and doc_path.exists()):
            raise IndexNotFoundError(f"TF-IDF index files missing in {directory}")

        self.vectorizer = load_pickle(vec_path)
        self.tfidf_matrix = load_pickle(mat_path)
        self.documents = load_pickle(doc_path)
        logger.info("TF-IDF index loaded from %s (%d documents)", dir_path, len(self.documents))
