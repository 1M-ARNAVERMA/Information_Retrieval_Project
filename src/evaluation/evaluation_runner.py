"""Evaluation runner benchmarking retrievers against FEVER ground-truth evidence."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Claim, RetrievalResult
from src.evaluation.map import mean_average_precision
from src.evaluation.ndcg import mean_ndcg_at_k
from src.evaluation.precision_recall import precision_at_k, recall_at_k
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("evaluation_runner")


def extract_ground_truth(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Extracts ground truth evidence keys (e.g. 'Albert Einstein:0') per claim ID."""
    ground_truth: Dict[str, Set[str]] = {}
    for c in claims:
        qid = str(c.id or hash(c.claim))
        keys: Set[str] = set()
        for ev_set in c.evidence:
            for item in ev_set.items:
                keys.add(f"{item.doc_id}:{item.sentence_id}")
                # Also include article title alone for article-level retrieval benchmarks
                keys.add(item.doc_id)
        if keys:
            ground_truth[qid] = keys
    return ground_truth


class EvaluationRunner:
    """Evaluates retrieval rankings against ground truth and exports benchmark reports."""

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def evaluate_rankings(
        self,
        retrieved_results: Dict[str, List[str]],
        ground_truth: Dict[str, Set[str]],
    ) -> Dict[str, float]:
        """Calculates P@K, Recall@K, MAP@K, and nDCG@K.

        Args:
            retrieved_results: Mapping of qid -> list of retrieved identifiers.
            ground_truth: Mapping of qid -> set of ground truth identifiers.

        Returns:
            Dictionary with aggregated metrics.
        """
        p_scores = []
        r_scores = []

        for qid, relevant in ground_truth.items():
            retrieved = retrieved_results.get(qid, [])
            p_scores.append(precision_at_k(retrieved, relevant, k=self.k))
            r_scores.append(recall_at_k(retrieved, relevant, k=self.k))

        mean_p = float(sum(p_scores) / len(p_scores)) if p_scores else 0.0
        mean_r = float(sum(r_scores) / len(r_scores)) if r_scores else 0.0
        map_score = mean_average_precision(retrieved_results, ground_truth, k=self.k)
        ndcg_score = mean_ndcg_at_k(retrieved_results, ground_truth, k=self.k)

        return {
            f"Precision@{self.k}": round(mean_p, 4),
            f"Recall@{self.k}": round(mean_r, 4),
            f"MAP@{self.k}": round(map_score, 4),
            f"nDCG@{self.k}": round(ndcg_score, 4),
            "num_queries": len(ground_truth),
        }

    def save_report(
        self,
        metrics: Dict[str, Any],
        output_file: Path,
    ) -> None:
        """Saves evaluation results to JSON and CSV."""
        ensure_dir(output_file.parent)

        # 1. Save JSON
        json_path = output_file.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 2. Save CSV
        csv_path = output_file.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics.items():
                writer.writerow([k, v])

        logger.info("Evaluation report saved to %s and %s", json_path, csv_path)
