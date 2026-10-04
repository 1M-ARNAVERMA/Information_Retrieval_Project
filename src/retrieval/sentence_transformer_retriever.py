"""Dense semantic retriever using Sentence Transformers and FAISS."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.common.types import Document, PreprocessMode, RetrievalResult
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.retrieval.base import BaseRetriever
from src.retrieval.faiss_index import FaissVectorIndex
from src.utils.config_loader import load_config, load_model_config
from src.utils.logger import get_logger
from src.utils.model_loader import get_sentence_transformer

logger = get_logger("sentence_transformer_retriever")


class SentenceTransformerRetriever(BaseRetriever):
    """Dense vector retriever using pretrained bi-encoder embeddings and FAISS."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        default_top_k: int = 50,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.default_top_k = default_top_k
        self.preprocessor = preprocessor or get_preprocessor()

        self._model = None
        self.index_wrapper: Optional[FaissVectorIndex] = None

    @property
    def model(self) -> Any:
        """Lazy-loaded SentenceTransformer model via centralized cache."""
        if self._model is None:
            self._model = get_sentence_transformer(
                model_name=self.model_name,
                device=self.device,
            )
        return self._model

    def index(self, documents: List[Document]) -> None:
        """Encodes documents into dense embeddings and indexes them with FAISS.

        Args:
            documents: List of Document models.
        """
        if not documents:
            logger.warning("Empty document list provided for dense indexing.")
            return

        logger.info("Encoding %d documents with SentenceTransformer...", len(documents))

        # Semantic mode preserves punctuation and capitalization for neural embedders
        texts = [
            self.preprocessor.preprocess(doc.text, mode=PreprocessMode.SEMANTIC)
            for doc in documents
        ]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        dim = embeddings.shape[1]
        self.index_wrapper = FaissVectorIndex(dimension=dim, metric="ip")

        # Prepare metadata
        metadata = [
            {
                "doc_id": doc.doc_id,
                "sentence_id": doc.sentence_id,
                "text": doc.text,
            }
            for doc in documents
        ]

        self.index_wrapper.add(embeddings, metadata)
        logger.info("Dense index built with %d documents (dim=%d).", len(documents), dim)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves nearest semantic neighbors for query.

        Args:
            query: Claim or search query string.
            top_k: Max results to return.

        Returns:
            List of RetrievalResult models sorted by cosine similarity.
        """
        if self.index_wrapper is None or self.index_wrapper.ntotal == 0:
            logger.warning("SentenceTransformer retriever queried before index was built or loaded.")
            return []

        k = top_k or self.default_top_k
        clean_q = self.preprocessor.preprocess(query, mode=PreprocessMode.SEMANTIC)
        if not clean_q.strip():
            return []

        query_emb = self.model.encode(
            clean_q,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        matches = self.index_wrapper.search(query_emb, top_k=k)

        results: List[RetrievalResult] = []
        for meta, score in matches:
            results.append(
                RetrievalResult(
                    doc_id=meta["doc_id"],
                    sentence_id=meta["sentence_id"],
                    text=meta["text"],
                    score=round(score, 6),
                    retriever="sentence_transformer",
                    metadata={"query_processed": clean_q},
                )
            )

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Saves dense vector index and metadata to disk."""
        if self.index_wrapper is None:
            raise ValueError("No dense index to save.")
        self.index_wrapper.save(directory)

    def load(self, directory: Union[str, Path]) -> None:
        """Loads dense vector index and metadata from disk."""
        self.index_wrapper = FaissVectorIndex()
        self.index_wrapper.load(directory)
