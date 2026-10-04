"""High-performance disk-backed BM25 retriever utilizing SQLite FTS5 for multi-million document corpora."""

from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Union

from src.common.exceptions import IndexNotFoundError
from src.common.types import Document, RetrievalResult
from src.retrieval.base import BaseRetriever
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("sqlite_bm25_retriever")


class SqliteBM25Retriever(BaseRetriever):
    """Full-scale, disk-backed BM25 retriever using SQLite FTS5.

    Scales to millions of Wikipedia articles without high RAM consumption.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/indexes/wiki_bm25.db",
        default_top_k: int = 50,
    ) -> None:
        self.db_path = Path(db_path)
        self.default_top_k = default_top_k
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Returns or creates a reusable thread-safe SQLite connection."""
        if self._conn is None:
            if not self.db_path.exists():
                raise IndexNotFoundError(f"SQLite FTS5 database not found at: {self.db_path}")
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA query_only = ON;")
        return self._conn

    @property
    def bm25(self) -> Any:
        """Compatibility property for health check."""
        return self._get_connection() if self.is_available() else None

    def is_available(self) -> bool:
        """Checks if the SQLite BM25 database exists on disk."""
        return self.db_path.exists() and self.db_path.stat().st_size > 1024

    def sanitize_query(self, query: str) -> str:
        """Sanitizes user claim/query for safe and effective FTS5 MATCH syntax."""
        from src.preprocessing.stopwords import BASE_STOPWORDS

        # Extract alphanumeric words and basic phrases
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return ""
        
        # Filter high-frequency stopwords to avoid skewing BM25 scores on short documents
        content_tokens = [t for t in tokens if t.lower() not in BASE_STOPWORDS]
        active_tokens = content_tokens if content_tokens else tokens

        # Combine tokens with OR for high recall across the multi-million sentence corpus
        return " OR ".join(active_tokens)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves top_k scoring sentences using native BM25 ranking.

        Args:
            query: Claim or search query string.
            top_k: Number of candidate sentences to return.

        Returns:
            List of RetrievalResult models sorted by relevance descending.
        """
        k = top_k or self.default_top_k
        if not query or not query.strip():
            return []

        if not self.is_available():
            logger.warning("SQLite BM25 database not found at %s.", self.db_path)
            return []

        fts_query = self.sanitize_query(query)
        if not fts_query:
            return []

        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # In FTS5, bm25(wiki_fts, weight_title, weight_text) returns negative score.
            # Weighting title matches 5.0x higher strongly prioritizes relevant Wikipedia articles.
            sql = """
                SELECT doc_id, sentence_id, text, bm25(wiki_fts, 5.0, 1.0) AS rank_score
                FROM wiki_fts
                WHERE wiki_fts MATCH ?
                ORDER BY rank_score ASC
                LIMIT ?;
            """
            cur.execute(sql, (fts_query, k))
            rows = cur.fetchall()

            results: List[RetrievalResult] = []
            for doc_id, sent_id, text, raw_rank in rows:
                # Normalize raw negative BM25 rank into positive score
                # raw_rank is typically between -0.1 and -30.0
                score = round(abs(float(raw_rank)), 4)

                results.append(
                    RetrievalResult(
                        doc_id=doc_id,
                        sentence_id=int(sent_id),
                        text=text,
                        score=score,
                        retriever="bm25",
                        metadata={"raw_bm25_rank": raw_rank},
                    )
                )

            return results
        except Exception as exc:
            logger.error("SQLite BM25 query error for '%s': %s", query, exc)
            return []

    def index(self, documents: List[Document]) -> None:
        """Indexes an in-memory document list into the SQLite FTS5 table."""
        ensure_dir(self.db_path.parent)
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
                doc_id UNINDEXED,
                sentence_id UNINDEXED,
                title,
                text,
                tokenize = 'porter unicode61'
            );
            """
        )

        batch = [
            (doc.doc_id, doc.sentence_id, doc.doc_id.replace("_", " "), doc.text)
            for doc in documents
        ]
        cur.executemany(
            "INSERT INTO wiki_fts (doc_id, sentence_id, title, text) VALUES (?, ?, ?, ?);",
            batch,
        )
        conn.commit()
        conn.close()
        logger.info("Indexed %d documents into %s", len(documents), self.db_path)

    def save(self, directory: Union[str, Path]) -> None:
        """Persists SQLite database (already stored directly on disk)."""
        pass

    def load(self, directory: Union[str, Path]) -> None:
        """Connects to the database path in the specified directory."""
        target = Path(directory)
        if target.is_dir():
            db_candidate = target / "wiki_bm25.db"
            if db_candidate.exists():
                self.db_path = db_candidate
        elif target.is_file():
            self.db_path = target
