"""End-to-end Fact Verification Pipeline integrating all components."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

from src.claim_analysis.claim_type import classify_claim_type
from src.claim_analysis.decomposition import ClaimDecomposer
from src.claim_analysis.ner import extract_entities
from src.claim_analysis.query_expansion import QueryExpander
from src.common.exceptions import PipelineError
from src.common.types import (
    Document,
    Entity,
    EvidenceGraphData,
    EvidenceScoreBreakdown,
    NLIPrediction,
    RerankedResult,
    RetrievalResult,
    SubClaim,
    VeracityLabel,
    VerificationResult,
)
from src.data_ingestion.wiki_loader import load_documents
from src.explainability.evidence_scoring import EvidenceScorer
from src.explainability.explanation_generator import ExplanationGenerator
from src.knowledge_graph.graph_builder import EvidenceGraphBuilder
from src.knowledge_graph.graph_reasoner import GraphReasoner
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.ranking.cross_encoder import CrossEncoderReranker
from src.ranking.mmr import maximal_marginal_relevance
from src.ranking.rrf import ReciprocalRankFusion
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.colbert_retriever import ColbertRetriever
from src.retrieval.entity_retriever import EntityAwareRetriever
from src.retrieval.sentence_transformer_retriever import SentenceTransformerRetriever
from src.retrieval.sqlite_bm25_retriever import SqliteBM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.utils.config_loader import load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger
from src.verification.confidence import ConfidenceEstimator
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.nli_verifier import NLIVerifier
from src.verification.verdict_mapper import VerdictMapper

logger = get_logger("fact_verification_pipeline")


class FactVerificationPipeline:
    """Orchestrator for multi-stage neural retrieval, ranking, and KG-based fact verification."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        indexes_dir: Optional[Union[str, Path]] = "data/indexes",
        sample_data_path: Optional[Union[str, Path]] = "data/sample/wiki_sample.jsonl",
        use_fallback_models: bool = True,
    ) -> None:
        self.config = load_config(config_path) if config_path else {}
        self.indexes_dir = Path(indexes_dir) if indexes_dir else Path("data/indexes")
        self.sample_data_path = Path(sample_data_path) if sample_data_path else Path("data/sample/wiki_sample.jsonl")
        self.use_fallback_models = use_fallback_models

        # Text preprocessing & claim analysis
        self.preprocessor = get_preprocessor()
        self.decomposer = ClaimDecomposer()
        self.expander = QueryExpander(use_llm=False)

        # Retrieval components
        self.tfidf_retriever = TfidfRetriever()
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = SentenceTransformerRetriever()
        self.colbert_retriever = ColbertRetriever(checkpoint="fallback" if use_fallback_models else "colbert-ir/colbertv2.0")

        # Fusion & ranking
        self.rrf = ReciprocalRankFusion(k=60)
        self.entity_booster = EntityAwareRetriever(boost_factor=1.25)
        self.reranker = CrossEncoderReranker(model_name="fallback" if use_fallback_models else "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.mmr_lambda = 0.7

        # Verification components
        self.nli_verifier = NLIVerifier(model_name="fallback" if use_fallback_models else "facebook/bart-large-mnli")
        self.contradiction_detector = ContradictionDetector()
        self.verdict_mapper = VerdictMapper(strategy="weighted_prob")
        self.confidence_estimator = ConfidenceEstimator()

        # Knowledge graph components
        self.graph_builder = EvidenceGraphBuilder()

        # Explainability components
        self.evidence_scorer = EvidenceScorer()
        self.explanation_generator = ExplanationGenerator(self.evidence_scorer)

        self.documents: List[Document] = []
        self._initialize_indices()

    def _initialize_indices(self) -> None:
        """Loads indexes from disk if available, or indexes the sample wiki data."""
        tfidf_file = self.indexes_dir / "tfidf_matrix.pkl"
        bm25_file = self.indexes_dir / "bm25_index.pkl"
        dense_file = self.indexes_dir / "faiss_metadata.pkl"

        if tfidf_file.exists() and bm25_file.exists() and dense_file.exists():
            try:
                logger.info("Loading prebuilt indices from %s...", self.indexes_dir)
                self.tfidf_retriever.load(self.indexes_dir)
                self.bm25_retriever.load(self.indexes_dir)
                self.dense_retriever.load(self.indexes_dir)
                self.documents = self.tfidf_retriever.documents

                # Check if full-scale SQLite BM25 database is available
                sqlite_db = self.indexes_dir / "wiki_bm25.db"
                if sqlite_db.exists() and sqlite_db.stat().st_size > 1024:
                    logger.info("Activating full-scale SQLite BM25 database at %s", sqlite_db)
                    self.bm25_retriever = SqliteBM25Retriever(db_path=sqlite_db)

                logger.info("Indices successfully loaded with %d documents.", len(self.documents))
                return
            except Exception as exc:
                logger.warning("Failed to load existing indices (%s). Rebuilding from data...", exc)

        if self.sample_data_path.exists():
            logger.info("Indexing sample documents from %s...", self.sample_data_path)
            self.documents = load_documents(self.sample_data_path)
            self.tfidf_retriever.index(self.documents)
            self.bm25_retriever.index(self.documents)
            self.dense_retriever.index(self.documents)
            self.colbert_retriever.index(self.documents)
        else:
            logger.warning("No index files or sample documents found. Pipeline initialized with empty corpus.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        use_entity_boost: bool = True,
        use_rrf: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieves and ranks candidate sentences for a query using multi-stage retrieval."""
        if not query or not query.strip():
            return []

        active_methods = methods or ["bm25", "tfidf", "sentence_transformer"]
        results_map: Dict[str, List[RetrievalResult]] = {}

        if "tfidf" in active_methods:
            results_map["tfidf"] = self.tfidf_retriever.retrieve(query, top_k=top_k * 2)
        if "bm25" in active_methods:
            results_map["bm25"] = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        if "sentence_transformer" in active_methods:
            results_map["sentence_transformer"] = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        if "colbert" in active_methods:
            results_map["colbert"] = self.colbert_retriever.retrieve(query, top_k=top_k * 2)

        if use_rrf and results_map:
            fused = self.rrf.fuse(results_map, top_k=top_k * 2)
        else:
            merged: List[RetrievalResult] = []
            for res_list in results_map.values():
                merged.extend(res_list)
            # Deduplicate by (doc_id, sentence_id)
            seen = set()
            fused = []
            for r in sorted(merged, key=lambda x: x.score, reverse=True):
                key = (r.doc_id, r.sentence_id)
                if key not in seen:
                    seen.add(key)
                    fused.append(r)

        if use_entity_boost:
            entities = extract_entities(query)
            fused = self.entity_booster.rerank(claim=query, candidates=fused, entities=entities)

        return fused[:top_k]

    def verify(
        self,
        claim: str,
        top_k: int = 5,
        include_graph: bool = True,
        aggregation_strategy: Optional[str] = "weighted_prob",
        use_decomposition: bool = True,
        use_query_expansion: bool = True,
        use_colbert: bool = False,
    ) -> VerificationResult:
        """Executes full end-to-end verification of an input claim."""
        start_time = time.time()
        clean_claim = claim.strip()
        if not clean_claim:
            raise PipelineError("Cannot verify empty claim.")

        # 1. Claim Analysis
        entities = extract_entities(clean_claim)
        claim_type = classify_claim_type(clean_claim)
        subclaims = self.decomposer.decompose(clean_claim) if use_decomposition else []
        queries = self.expander.expand(clean_claim) if use_query_expansion else [clean_claim]

        # 2. Retrieval
        if isinstance(self.bm25_retriever, SqliteBM25Retriever):
            retrieval_methods = ["bm25"]
        else:
            retrieval_methods = ["bm25", "sentence_transformer", "tfidf"]
        if use_colbert:
            retrieval_methods.append("colbert")

        candidate_map: Dict[tuple, RetrievalResult] = {}
        for q in queries:
            cands = self.retrieve(
                q,
                top_k=top_k * 3,
                methods=retrieval_methods,
                use_entity_boost=True,
                use_rrf=True,
            )
            for c in cands:
                k = (c.doc_id, c.sentence_id)
                if k not in candidate_map or c.score > candidate_map[k].score:
                    candidate_map[k] = c

        all_candidates = list(candidate_map.values())
        if not all_candidates:
            # Fallback to direct retrieval if queries yielded nothing
            all_candidates = self.retrieve(clean_claim, top_k=top_k * 2)

        # 3. Neural Cross-Encoder Reranking & MMR Diversification
        reranked = self.reranker.rerank(clean_claim, all_candidates, top_k=top_k * 2)
        diversified_evidence = maximal_marginal_relevance(
            reranked, top_k=top_k, lambda_param=self.mmr_lambda
        )

        # 4. NLI Inference & Verification
        predictions: List[NLIPrediction] = []
        for ev in diversified_evidence:
            e_p, c_p, n_p, label = self.nli_verifier.predict_pair(clean_claim, ev.text)
            predictions.append(
                NLIPrediction(
                    doc_id=ev.doc_id,
                    sentence_id=ev.sentence_id,
                    text=ev.text,
                    entailment_prob=round(e_p, 4),
                    contradiction_prob=round(c_p, 4),
                    neutral_prob=round(n_p, 4),
                    predicted_label=label,
                )
            )

        strat = aggregation_strategy or "weighted_prob"
        self.verdict_mapper.strategy = strat
        evidence_scores = [float(ev.rerank_score) for ev in diversified_evidence]
        verdict, prob_dist = self.verdict_mapper.map_verdict(
            predictions=predictions,
            weights=evidence_scores if evidence_scores else None,
        )
        confidence = self.confidence_estimator.estimate_confidence(
            verdict=verdict,
            prob_dist=prob_dist,
            evidence_scores=evidence_scores,
        )

        # 5. Knowledge Graph Construction & Reasoning
        graph_data = EvidenceGraphData()
        reasoning_traces: List[Dict[str, Any]] = []
        bridge_entities: List[Dict[str, Any]] = []
        if include_graph:
            ev_graph = self.graph_builder.build_graph(
                claim=clean_claim,
                entities=entities,
                evidence_list=diversified_evidence,
                nli_predictions=predictions,
                subclaims=subclaims,
            )
            graph_data = ev_graph.to_graph_data()
            reasoner = GraphReasoner(ev_graph)
            reasoning_traces = reasoner.get_multi_hop_traces()
            bridge_entities = reasoner.find_bridge_entities()

        # 6. Explainability Breakdown & Audit Trail
        score_breakdowns: List[EvidenceScoreBreakdown] = []
        for ev, pred in zip(diversified_evidence, predictions):
            bd = self.evidence_scorer.score_evidence(
                claim=clean_claim,
                evidence=ev,
                nli_prediction=pred,
                claim_entities=entities,
            )
            score_breakdowns.append(bd)

        explanation = self.explanation_generator.generate_explanation(
            claim=clean_claim,
            verdict=verdict,
            confidence=confidence,
            evidence_list=diversified_evidence,
            nli_predictions=predictions,
            entities=entities,
            score_breakdowns=score_breakdowns,
        )
        explanation["evidence_breakdowns"] = [b.model_dump() for b in score_breakdowns]
        explanation["claim_type"] = claim_type.value if hasattr(claim_type, "value") else str(claim_type)
        explanation["multi_hop_traces"] = reasoning_traces
        explanation["bridge_entities"] = bridge_entities

        duration = round((time.time() - start_time) * 1000, 2)
        metadata = {
            "execution_time_ms": duration,
            "queries_used": queries,
            "total_candidates_retrieved": len(all_candidates),
            "num_evidence_selected": len(diversified_evidence),
            "aggregation_strategy": strat,
        }

        return VerificationResult(
            claim=clean_claim,
            verdict=verdict,
            confidence=round(confidence, 4),
            top_evidence=diversified_evidence,
            entities=entities,
            subclaims=subclaims,
            nli_predictions=predictions,
            explanation=explanation,
            graph=graph_data,
            retrieval_metadata=metadata,
        )
