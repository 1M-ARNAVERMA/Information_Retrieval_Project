"""Evaluation and benchmark package for FEVER-X."""

from src.evaluation.ablation import AblationStudy
from src.evaluation.evaluation_runner import EvaluationRunner, extract_ground_truth
from src.evaluation.map import average_precision_at_k, mean_average_precision
from src.evaluation.ndcg import dcg_at_k, mean_ndcg_at_k, ndcg_at_k
from src.evaluation.precision_recall import precision_at_k, recall_at_k

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "average_precision_at_k",
    "mean_average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "mean_ndcg_at_k",
    "EvaluationRunner",
    "extract_ground_truth",
    "AblationStudy",
]
