"""Script to build and persist retrieval indexes offline."""

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_ingestion.wiki_loader import load_documents
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.sentence_transformer_retriever import SentenceTransformerRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.utils.config_loader import load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("build_indexes")


def main():
    parser = argparse.ArgumentParser(description="Build and persist retrieval indices offline.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sample/wiki_sample.jsonl",
        help="Path to Wikipedia JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/indexes",
        help="Directory to save serialized indices.",
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        default="all",
        help="Comma-separated retrievers to build: 'tfidf', 'bm25', 'sentence_transformer', or 'all'.",
    )
    args = parser.parse_args()

    data_file = Path(args.data_path)
    if not data_file.exists():
        logger.error("Data file does not exist: %s", data_file)
        sys.exit(1)

    out_dir = ensure_dir(args.output_dir)
    logger.info("Loading Wikipedia documents from %s...", data_file)
    docs = load_documents(data_file)
    logger.info("Loaded %d documents/sentences.", len(docs))

    retrievers_to_build = (
        ["tfidf", "bm25", "sentence_transformer"]
        if args.retrievers == "all"
        else [r.strip().lower() for r in args.retrievers.split(",")]
    )

    if "tfidf" in retrievers_to_build:
        logger.info("Building TF-IDF Index...")
        tfidf = TfidfRetriever()
        tfidf.index(docs)
        tfidf.save(out_dir)

    if "bm25" in retrievers_to_build:
        logger.info("Building BM25 Index...")
        bm25 = BM25Retriever()
        bm25.index(docs)
        bm25.save(out_dir)

    if "sentence_transformer" in retrievers_to_build:
        logger.info("Building SentenceTransformer (FAISS) Index...")
        st_retriever = SentenceTransformerRetriever()
        st_retriever.index(docs)
        st_retriever.save(out_dir)

    logger.info("Indexing complete! All indices stored in %s", out_dir)


if __name__ == "__main__":
    main()
