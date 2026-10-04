"""FEVER data ingestion, normalization, and parsing package."""

from src.data_ingestion.normalizer import normalize_fever_text, normalize_wiki_title
from src.data_ingestion.fever_loader import parse_fever_record, stream_claims, load_claims
from src.data_ingestion.wiki_loader import parse_wiki_page, stream_documents, load_documents

__all__ = [
    "normalize_fever_text",
    "normalize_wiki_title",
    "parse_fever_record",
    "stream_claims",
    "load_claims",
    "parse_wiki_page",
    "stream_documents",
    "load_documents",
]
