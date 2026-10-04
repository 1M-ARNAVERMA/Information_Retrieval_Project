"""Ablation study comparing all 9 retrieval and verification architectures."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from src.claim_analysis.decomposition import ClaimDecomposer
from src.claim_analysis.ner import extract_entities
from src.claim_analysis.query_expansion import QueryExpander
from src.common.types import Claim, Document
from src.evaluation.evaluation_runner import EvaluationRunner, extract_ground_truth
from src.ranking.cross_encoder import CrossEncoderReranker
from src.ranking.rrf import ReciprocalRankFusion
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.colbert_retriever import ColbertRetriever
from src.retrieval.entity_retriever import EntityAwareRetriever
from src.retrieval.sentence_transformer_retriever import SentenceTransformerRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("ablation_study")


class AblationStudy:
    """Executes systematic 9-stage ablation experiments across the FEVER-X pipeline."""

    def __init__(self, documents: List[Document], k: int = 10) -> None:
        self.documents = documents
        self.k = k
        self.runner = EvaluationRunner(k=k)

        # Initialize all modular components
        logger.info("Initializing retrievers for ablation study...")
        self.tfidf = TfidfRetriever()
        self.tfidf.index(documents)

        self.bm25 = BM25Retriever()
        self.bm25.index(documents)

        self.dense = SentenceTransformerRetriever()
        self.dense.index(documents)

        self.colbert = ColbertRetriever(checkpoint="fallback")
        self.colbert.index(documents)

        self.rrf = ReciprocalRankFusion(k=60)
        self.entity_booster = EntityAwareRetriever(boost_factor=1.25)
        self.reranker = CrossEncoderReranker(model_name="fallback", default_top_k=k)
        self.decomposer = ClaimDecomposer()
        self.expander = QueryExpander(use_llm=False)

    def _to_ids(self, results) -> List[str]:
        """Maps retrieval or reranked results to identifiers (doc_id:sentence_id and doc_id)."""
        ids = []
        for r in results:
            ids.append(f"{r.doc_id}:{r.sentence_id}")
            ids.append(r.doc_id)
        # Deduplicate preserving order
        return list(dict.fromkeys(ids))

    def run_ablation(self, claims: List[Claim]) -> pd.DataFrame:
        """Executes the 9 experimental configurations on actual claims.

        Configurations:
            1. TF-IDF
            2. BM25
            3. Sentence Transformer
            4. TF-IDF + BM25 + Dense (Simple Linear Merge)
            5. + RRF Fusion
            6. + Entity-Aware Retrieval
            7. + ColBERT Late-Interaction
            8. + Cross-Encoder Reranking
            9. Full System (Decomposition + Expansion + RRF + Entity + CrossEncoder)

        Returns:
            Pandas DataFrame containing true evaluated metrics for all 9 configurations.
        """
        ground_truth = extract_ground_truth(claims)
        if not ground_truth:
            logger.warning("No ground truth evidence found in claims for evaluation.")
            return pd.DataFrame()

        results_by_config: Dict[str, Dict[str, List[str]]] = {
            "1. TF-IDF": {},
            "2. BM25": {},
            "3. Sentence Transformer": {},
            "4. TF-IDF + BM25 + Dense (Merge)": {},
            "5. + RRF Fusion": {},
            "6. + Entity-Aware Retrieval": {},
            "7. + ColBERT Late-Interaction": {},
            "8. + Cross-Encoder Reranking": {},
            "9. Full System (End-to-End)": {},
        }

        logger.info("Running ablation benchmarks across %d claims...", len(claims))

        for c in claims:
            qid = str(c.id or hash(c.claim))
            query = c.claim
            ents = extract_entities(query)

            # 1. Individual retrievers
            tfidf_res = self.tfidf.retrieve(query, top_k=self.k * 2)
            bm25_res = self.bm25.retrieve(query, top_k=self.k * 2)
            dense_res = self.dense.retrieve(query, top_k=self.k * 2)
            colbert_res = self.colbert.retrieve(query, top_k=self.k * 2)

            results_by_config["1. TF-IDF"][qid] = self._to_ids(tfidf_res)
            results_by_config["2. BM25"][qid] = self._to_ids(bm25_res)
            results_by_config["3. Sentence Transformer"][qid] = self._to_ids(dense_res)

            # 4. Simple linear merge
            merged = tfidf_res + bm25_res + dense_res
            results_by_config["4. TF-IDF + BM25 + Dense (Merge)"][qid] = self._to_ids(merged)

            # 5. RRF Fusion
            rrf_res = self.rrf.fuse({
                "tfidf": tfidf_res,
                "bm25": bm25_res,
                "sentence_transformer": dense_res,
            }, top_k=self.k * 2)
            results_by_config["5. + RRF Fusion"][qid] = self._to_ids(rrf_res)

            # 6. Entity-Aware Retrieval
            entity_boosted = self.entity_booster.rerank(query, rrf_res, entities=ents, top_k=self.k * 2)
            results_by_config["6. + Entity-Aware Retrieval"][qid] = self._to_ids(entity_boosted)

            # 7. With ColBERT
            colbert_fused = self.rrf.fuse({
                "tfidf": tfidf_res,
                "bm25": bm25_res,
                "sentence_transformer": dense_res,
                "colbert": colbert_res,
            }, top_k=self.k * 2)
            results_by_config["7. + ColBERT Late-Interaction"][qid] = self._to_ids(colbert_fused)

            # 8. Cross-Encoder Reranking
            cross_reranked = self.reranker.rerank(query, entity_boosted, top_k=self.k)
            results_by_config["8. + Cross-Encoder Reranking"][qid] = self._to_ids(cross_reranked)

            # 9. Full System (Decomposition + Expansion + Entity RRF + Cross-Encoder)
            subclaims = self.decomposer.decompose(query)
            expanded_queries = self.expander.expand(query, entities=ents)

            multi_retrievals = []
            for eq in expanded_queries:
                multi_retrievals.extend(self.bm25.retrieve(eq, top_k=10))
                multi_retrievals.extend(self.dense.retrieve(eq, top_k=10))

            full_fused = self.rrf.fuse({
                "expanded": multi_retrievals,
                "entity_boosted": entity_boosted,
            }, top_k=self.k * 2)
            full_final = self.reranker.rerank(query, full_fused, top_k=self.k)
            results_by_config["9. Full System (End-to-End)"][qid] = self._to_ids(full_final)

        # Compute metric rows for each configuration
        ablation_rows = []
        for config_name, retrieved_dict in results_by_config.items():
            metrics = self.runner.evaluate_rankings(retrieved_dict, ground_truth)
            ablation_rows.append({
                "Configuration": config_name,
                f"Precision@{self.k}": metrics[f"Precision@{self.k}"],
                f"Recall@{self.k}": metrics[f"Recall@{self.k}"],
                f"MAP@{self.k}": metrics[f"MAP@{self.k}"],
                f"nDCG@{self.k}": metrics[f"nDCG@{self.k}"],
            })

        df = pd.DataFrame(ablation_rows)
        logger.info("Ablation study complete:\n%s", df.to_string(index=False))
        return df

    def save_results(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Saves ablation DataFrame to CSV and JSON."""
        ensure_dir(output_dir)
        csv_file = output_dir / "ablation_results.csv"
        json_file = output_dir / "ablation_results.json"

        df.to_csv(csv_file, index=False)
        df.to_json(json_file, orient="records", indent=2)
        logger.info("Saved ablation results to %s and %s", csv_file, json_file)
