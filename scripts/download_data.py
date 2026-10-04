"""Script to download FEVER dataset or prepare local sample data."""

import argparse
from pathlib import Path
import sys
import urllib.request

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("download_data")

FEVER_OFFICIAL_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "paper_dev": "https://fever.ai/download/fever/paper_dev.jsonl",
    "paper_test": "https://fever.ai/download/fever/paper_test.jsonl",
}


def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file from url to dest_path with progress logging."""
    logger.info("Downloading %s to %s...", url, dest_path)
    ensure_dir(dest_path.parent)

    try:
        urllib.request.urlretrieve(url, str(dest_path))
        logger.info("Successfully downloaded %s (%d bytes).", dest_path.name, dest_path.stat().st_size)
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        raise


def setup_sample_data(sample_dir: Path) -> None:
    """Checks and confirms sample datasets exist."""
    fever_sample = sample_dir / "fever_sample.jsonl"
    wiki_sample = sample_dir / "wiki_sample.jsonl"

    if fever_sample.exists() and wiki_sample.exists():
        logger.info("Sample datasets are ready in %s", sample_dir)
        logger.info(" - %s (%d bytes)", fever_sample.name, fever_sample.stat().st_size)
        logger.info(" - %s (%d bytes)", wiki_sample.name, wiki_sample.stat().st_size)
    else:
        logger.warning("Sample datasets missing in %s. Check data/sample/", sample_dir)


def main():
    parser = argparse.ArgumentParser(description="FEVER dataset download & sample preparation tool")
    parser.add_argument("--mode", choices=["sample", "official"], default="sample", help="Mode: 'sample' or 'official'")
    parser.add_argument("--split", choices=["all", "train", "paper_dev", "paper_test"], default="paper_dev", help="Split to download if mode=official")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    sample_dir = Path(cfg["paths"]["sample_data_dir"])

    if args.mode == "sample":
        setup_sample_data(sample_dir)
    elif args.mode == "official":
        logger.info("Preparing to download official FEVER data to %s...", raw_dir)
        splits = FEVER_OFFICIAL_URLS.keys() if args.split == "all" else [args.split]
        for s in splits:
            dest = raw_dir / f"{s}.jsonl"
            download_file(FEVER_OFFICIAL_URLS[s], dest)


if __name__ == "__main__":
    main()
