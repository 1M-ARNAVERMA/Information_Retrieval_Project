"""FEVER Wikipedia corpus loader and sentence extractor."""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

from src.common.exceptions import DataIngestionError
from src.common.types import Document
from src.data_ingestion.normalizer import normalize_fever_text, normalize_wiki_title
from src.utils.file_utils import read_jsonl
from src.utils.logger import get_logger

logger = get_logger("wiki_loader")


def parse_wiki_page(raw: Dict[str, Union[str, int]]) -> List[Document]:
    """Parses a FEVER Wikipedia page JSON record into sentence-level Document models.

    Args:
        raw: Dict from wiki-pages-*.jsonl containing 'id', 'text', and/or 'lines'.

    Returns:
        List of Document instances representing distinct sentences.
    """
    raw_id = str(raw.get("id", "")).strip()
    if not raw_id:
        return []

    doc_id = normalize_wiki_title(raw_id)
    documents: List[Document] = []

    lines_str = raw.get("lines")
    if lines_str and isinstance(lines_str, str):
        # Format in official FEVER dumps:
        # <sentence_id>\t<sentence_text>\t<url_id>\t<line_id>\n...
        for line in lines_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    sent_id = int(parts[0])
                    sent_text = normalize_fever_text(parts[1])
                    if sent_text:
                        documents.append(
                            Document(
                                doc_id=doc_id,
                                sentence_id=sent_id,
                                text=sent_text,
                            )
                        )
                except ValueError:
                    continue
    else:
        # Fallback if lines is not provided but text is available
        text_content = raw.get("text", "")
        if isinstance(text_content, str):
            for idx, sent in enumerate(text_content.split("\n")):
                cleaned = normalize_fever_text(sent)
                if cleaned:
                    documents.append(
                        Document(
                            doc_id=doc_id,
                            sentence_id=idx,
                            text=cleaned,
                        )
                    )

    return documents


def stream_documents(
    file_path: Union[str, Path],
    limit_docs: Optional[int] = None,
    limit_sentences: Optional[int] = None,
) -> Generator[Document, None, None]:
    """Streams individual sentence-level Document instances from a Wikipedia JSONL dump.

    Args:
        file_path: Path to wiki-pages JSONL file.
        limit_docs: Optional maximum number of articles to read.
        limit_sentences: Optional maximum number of total sentences to yield.

    Yields:
        Document objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"Wikipedia dump file not found: {path}")

    doc_count = 0
    sent_count = 0

    for raw_page in read_jsonl(path):
        docs = parse_wiki_page(raw_page)
        for doc in docs:
            yield doc
            sent_count += 1
            if limit_sentences is not None and sent_count >= limit_sentences:
                return

        doc_count += 1
        if limit_docs is not None and doc_count >= limit_docs:
            break


def load_documents(
    file_path: Union[str, Path],
    limit_docs: Optional[int] = None,
    limit_sentences: Optional[int] = None,
) -> List[Document]:
    """Loads sentence-level Documents from a Wikipedia JSONL dump into memory.

    Args:
        file_path: Path to wiki-pages JSONL file.
        limit_docs: Optional maximum articles to read.
        limit_sentences: Optional maximum sentences to load.

    Returns:
        List of Document instances.
    """
    return list(
        stream_documents(
            file_path=file_path,
            limit_docs=limit_docs,
            limit_sentences=limit_sentences,
        )
    )
