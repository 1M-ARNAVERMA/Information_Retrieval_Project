"""CLI script to run full evaluation and 9-stage ablation study."""

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_ingestion.fever_loader import load_claims
from src.data_ingestion.wiki_loader import load_documents
from src.evaluation.ablation import AblationStudy
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("run_evaluation")


def main():
    parser = argparse.ArgumentParser(description="FEVER-X Evaluation & Ablation Benchmark Runner")
    parser.add_argument(
        "--claims_path",
        type=str,
        default="data/sample/fever_sample.jsonl",
        help="Path to FEVER claims JSONL file.",
    )
    parser.add_argument(
        "--wiki_path",
        type=str,
        default="data/sample/wiki_sample.jsonl",
        help="Path to Wikipedia JSONL corpus file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/evaluation",
        help="Output directory for CSV and JSON benchmark results.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff rank position for P@K, Recall@K, MAP@K, nDCG@K.",
    )
    args = parser.parse_args()

    claims_file = Path(args.claims_path)
    wiki_file = Path(args.wiki_path)
    out_dir = ensure_dir(args.output_dir)

    if not claims_file.exists() or not wiki_file.exists():
        logger.error("Dataset files missing. Verify %s and %s", claims_file, wiki_file)
        sys.exit(1)

    logger.info("Loading claims from %s...", claims_file)
    claims = load_claims(claims_file)
    logger.info("Loaded %d claims.", len(claims))

    logger.info("Loading documents from %s...", wiki_file)
    documents = load_documents(wiki_file)
    logger.info("Loaded %d Wikipedia documents.", len(documents))

    study = AblationStudy(documents=documents, k=args.k)
    df_results = study.run_ablation(claims)

    study.save_results(df_results, out_dir)
    print("\n" + "=" * 80)
    print("FEVER-X ABLATION BENCHMARK RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
