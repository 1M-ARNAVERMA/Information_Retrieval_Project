"""Abstract base interface for all FEVER-X retrievers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from src.common.types import Document, RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class defining the contract for retrieval models."""

    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """Indexes an iterable of Document objects into memory or an on-disk structure.

        Args:
            documents: List of Document instances to index.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves top_k scoring candidates for a query string.

        Args:
            query: User claim or search query string.
            top_k: Number of highest-ranking documents to return.

        Returns:
            List of RetrievalResult models sorted by score descending.
        """
        pass

    @abstractmethod
    def save(self, directory: Union[str, Path]) -> None:
        """Persists the retrieval index and document metadata to disk.

        Args:
            directory: Target directory path.
        """
        pass

    @abstractmethod
    def load(self, directory: Union[str, Path]) -> None:
        """Loads the retrieval index and document metadata from disk.

        Args:
            directory: Directory path containing serialized index files.
        """
        pass
