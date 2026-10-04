"""Unit tests for Phase 14: Ablation study and benchmark suite."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_ingestion.fever_loader import load_claims
from src.data_ingestion.wiki_loader import load_documents
from src.evaluation.ablation import AblationStudy

SAMPLE_CLAIMS = Path("data/sample/fever_sample.jsonl")
SAMPLE_WIKI = Path("data/sample/wiki_sample.jsonl")


def test_ablation_study_execution(tmp_path):
    """Validates running the 9-stage ablation study on sample documents."""
    claims = load_claims(SAMPLE_CLAIMS, limit=3)
    docs = load_documents(SAMPLE_WIKI)

    study = AblationStudy(documents=docs, k=5)
    df = study.run_ablation(claims)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9  # Exactly 9 experimental configurations

    expected_configs = [
        "1. TF-IDF",
        "2. BM25",
        "3. Sentence Transformer",
        "4. TF-IDF + BM25 + Dense (Merge)",
        "5. + RRF Fusion",
        "6. + Entity-Aware Retrieval",
        "7. + ColBERT Late-Interaction",
        "8. + Cross-Encoder Reranking",
        "9. Full System (End-to-End)",
    ]
    assert list(df["Configuration"]) == expected_configs
    assert "Precision@5" in df.columns
    assert "Recall@5" in df.columns
    assert "MAP@5" in df.columns
    assert "nDCG@5" in df.columns

    # Test file persistence
    study.save_results(df, tmp_path)
    csv_file = tmp_path / "ablation_results.csv"
    json_file = tmp_path / "ablation_results.json"
    assert csv_file.exists()
    assert json_file.exists()
    assert csv_file.stat().st_size > 0
    assert json_file.stat().st_size > 0
