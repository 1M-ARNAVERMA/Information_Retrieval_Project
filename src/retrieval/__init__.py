"""Information Retrieval package for FEVER-X."""

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.colbert_retriever import ColbertRetriever, compute_maxsim
from src.retrieval.entity_retriever import EntityAwareRetriever
from src.retrieval.faiss_index import FaissVectorIndex
from src.retrieval.sentence_transformer_retriever import SentenceTransformerRetriever
from src.retrieval.sqlite_bm25_retriever import SqliteBM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever

__all__ = [
    "BaseRetriever",
    "TfidfRetriever",
    "BM25Retriever",
    "SqliteBM25Retriever",
    "FaissVectorIndex",
    "SentenceTransformerRetriever",
    "EntityAwareRetriever",
    "ColbertRetriever",
    "compute_maxsim",
]
