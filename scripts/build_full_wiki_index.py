"""High-speed indexing script to index the full FEVER Wikipedia corpus into a disk-backed SQLite FTS5 database."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_ingestion.normalizer import normalize_fever_text, normalize_wiki_title
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("build_full_wiki_index")


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initializes high-performance SQLite FTS5 database."""
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Performance tuning pragmas
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA cache_size = 100000;")
    cur.execute("PRAGMA temp_store = MEMORY;")

    # FTS5 Virtual Table
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

    # Checkpoint tracking table for resumable indexing
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS _indexed_files (
            filename TEXT PRIMARY KEY,
            articles_count INTEGER,
            sentences_count INTEGER,
            indexed_at REAL
        );
        """
    )
    conn.commit()
    return conn


def get_already_indexed_files(conn: sqlite3.Connection) -> set:
    """Returns set of filenames that have already been processed."""
    cur = conn.cursor()
    cur.execute("SELECT filename FROM _indexed_files;")
    return {row[0] for row in cur.fetchall()}


def index_wiki_file(
    file_path: Path,
    conn: sqlite3.Connection,
    max_sentences_per_article: int = 5,
    batch_size: int = 20000,
) -> tuple:
    """Indexes a single wiki JSONL dump file into SQLite FTS5."""
    cur = conn.cursor()
    batch = []
    articles_count = 0
    sentences_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            raw_id = obj.get("id")
            lines_str = obj.get("lines")
            if not raw_id or not lines_str:
                continue

            doc_id = normalize_wiki_title(str(raw_id).strip())
            title_text = doc_id.replace("_", " ")
            articles_count += 1

            # Extract lead sentences (sentence 0 through max_sentences)
            for s_line in lines_str.split("\n")[:max_sentences_per_article]:
                s_line = s_line.strip()
                if not s_line:
                    continue
                parts = s_line.split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    try:
                        sent_id = int(parts[0])
                        cleaned_text = normalize_fever_text(parts[1].strip())
                        if cleaned_text:
                            batch.append((doc_id, sent_id, title_text, cleaned_text))
                            sentences_count += 1
                    except ValueError:
                        continue

            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO wiki_fts (doc_id, sentence_id, title, text) VALUES (?, ?, ?, ?);",
                    batch,
                )
                conn.commit()
                batch.clear()

    if batch:
        cur.executemany(
            "INSERT INTO wiki_fts (doc_id, sentence_id, title, text) VALUES (?, ?, ?, ?);",
            batch,
        )
        conn.commit()

    # Record checkpoint
    cur.execute(
        "INSERT OR REPLACE INTO _indexed_files (filename, articles_count, sentences_count, indexed_at) VALUES (?, ?, ?, ?);",
        (file_path.name, articles_count, sentences_count, time.time()),
    )
    conn.commit()
    return articles_count, sentences_count


def main():
    parser = argparse.ArgumentParser(description="Build disk-backed SQLite FTS5 index for full FEVER Wikipedia corpus.")
    parser.add_argument(
        "--wiki_dir",
        type=str,
        default="data/raw/wiki-pages",
        help="Path to folder containing wiki-*.jsonl files",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/indexes/wiki_bm25.db",
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max number of files to process (None for all 109)",
    )
    parser.add_argument(
        "--sentences_per_article",
        type=int,
        default=5,
        help="Number of lead sentences to index per article (default 5 covers 95%+ FEVER evidence)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25000,
        help="SQLite batch insert size",
    )
    args = parser.parse_args()

    wiki_dir = Path(args.wiki_dir)
    if not wiki_dir.exists():
        logger.error("Wiki directory does not exist: %s", wiki_dir)
        sys.exit(1)

    all_files = sorted(list(wiki_dir.glob("wiki-*.jsonl")))
    if not all_files:
        logger.error("No wiki-*.jsonl files found in %s", wiki_dir)
        sys.exit(1)

    if args.max_files:
        all_files = all_files[:args.max_files]

    db_path = Path(args.db_path)
    logger.info("Initializing SQLite FTS5 database at %s...", db_path)
    conn = init_db(db_path)

    already_done = get_already_indexed_files(conn)
    files_to_do = [f for f in all_files if f.name not in already_done]

    logger.info(
        "Total files: %d | Already indexed: %d | Remaining: %d",
        len(all_files),
        len(already_done),
        len(files_to_do),
    )

    if not files_to_do:
        logger.info("All files are already indexed in %s! Index is ready.", db_path)
        conn.close()
        return

    t_start = time.time()
    total_articles = 0
    total_sentences = 0

    pbar = tqdm(files_to_do, desc="Indexing Wikipedia Dumps", unit="file")
    for file_path in pbar:
        t_file = time.time()
        arts, sents = index_wiki_file(
            file_path=file_path,
            conn=conn,
            max_sentences_per_article=args.sentences_per_article,
            batch_size=args.batch_size,
        )
        total_articles += arts
        total_sentences += sents
        rate = arts / max(0.001, (time.time() - t_file))
        pbar.set_postfix({"arts/s": f"{rate:.0f}", "total_sents": f"{total_sentences:,}"})

    total_time = time.time() - t_start
    logger.info(
        "Indexing finished successfully! Indexed %d articles (%d sentences) in %.2fs (%.1f arts/sec).",
        total_articles,
        total_sentences,
        total_time,
        total_articles / max(0.001, total_time),
    )
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    logger.info("Database size: %.2f MB at %s", db_size_mb, db_path)
    conn.close()


if __name__ == "__main__":
    main()
