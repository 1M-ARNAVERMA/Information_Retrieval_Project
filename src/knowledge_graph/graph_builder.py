"""Constructs multi-hop evidence graphs linking claims, entities, and evidence sentences."""

from typing import List, Optional

from src.claim_analysis.ner import extract_entities
from src.common.types import (
    Entity,
    EvidenceGraphData,
    NLIPrediction,
    RerankedResult,
    SubClaim,
    VeracityLabel,
)
from src.knowledge_graph.evidence_graph import EvidenceGraph
from src.utils.logger import get_logger

logger = get_logger("graph_builder")


class EvidenceGraphBuilder:
    """Builds structured NetworkX evidence graphs from verification pipeline outputs."""

    def __init__(self) -> None:
        pass

    def build_graph(
        self,
        claim: str,
        entities: List[Entity],
        evidence_list: List[RerankedResult],
        nli_predictions: List[NLIPrediction],
        subclaims: Optional[List[SubClaim]] = None,
    ) -> EvidenceGraph:
        """Constructs an interpretable evidence graph.

        Args:
            claim: User claim string.
            entities: Entities extracted from claim.
            evidence_list: Top reranked evidence sentences.
            nli_predictions: Sentence-level NLI prediction outcomes.
            subclaims: Optional decomposed subclaims.

        Returns:
            Populated EvidenceGraph instance.
        """
        graph = EvidenceGraph()

        # 1. Add root CLAIM node
        claim_node_id = "claim:root"
        graph.add_node(claim_node_id, label="CLAIM", text=claim)

        # 2. Add Subclaim nodes if available
        if subclaims:
            for sc in subclaims:
                sc_id = f"subclaim:{sc.id}"
                graph.add_node(sc_id, label="SUBCLAIM", text=sc.text)
                graph.add_edge(claim_node_id, sc_id, relation="DECOMPOSED_INTO")

        # 3. Add Claim Entities and link (CLAIM) -[MENTIONS]-> (ENTITY)
        for ent in entities:
            ent_id = f"entity:{ent.text.strip().lower()}"
            if not graph.has_node(ent_id):
                graph.add_node(ent_id, label="ENTITY", text=ent.text, entity_type=ent.label)
            graph.add_edge(claim_node_id, ent_id, relation="MENTIONS", weight=1.0)

        # Map predictions by (doc_id, sentence_id) for rapid alignment
        pred_map = {(p.doc_id, p.sentence_id): p for p in nli_predictions}

        # 4. Add Evidence nodes and relations
        for ev in evidence_list:
            ev_id = f"evidence:{ev.doc_id}:{ev.sentence_id}"
            pred = pred_map.get((ev.doc_id, ev.sentence_id))

            verdict_label = pred.predicted_label.value if pred else "NEUTRAL"
            verdict_prob = (
                pred.entailment_prob
                if verdict_label == "SUPPORTS"
                else (pred.contradiction_prob if verdict_label == "REFUTES" else pred.neutral_prob)
            ) if pred else 0.5

            graph.add_node(
                ev_id,
                label="EVIDENCE",
                doc_id=ev.doc_id,
                sentence_id=ev.sentence_id,
                text=ev.text,
                verdict=verdict_label,
                rerank_score=ev.rerank_score,
            )

            # Link (EVIDENCE) -> (CLAIM) with veracity relation
            relation = verdict_label  # "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
            graph.add_edge(ev_id, claim_node_id, relation=relation, weight=verdict_prob)

            # Link evidence to article topic entity (resolves coreference for 'She', 'He', 'It')
            doc_ent_id = f"entity:{ev.doc_id.strip().lower()}"
            if not graph.has_node(doc_ent_id):
                graph.add_node(doc_ent_id, label="ENTITY", text=ev.doc_id, entity_type="TOPIC")
            graph.add_edge(ev_id, doc_ent_id, relation="MENTIONS", weight=1.0)

            # Extract entities inside evidence sentence
            ev_entities = extract_entities(ev.text)
            for ev_ent in ev_entities:
                ev_ent_id = f"entity:{ev_ent.text.strip().lower()}"
                if not graph.has_node(ev_ent_id):
                    graph.add_node(
                        ev_ent_id,
                        label="ENTITY",
                        text=ev_ent.text,
                        entity_type=ev_ent.label,
                    )
                # Link (EVIDENCE) -[MENTIONS]-> (ENTITY)
                graph.add_edge(ev_id, ev_ent_id, relation="MENTIONS", weight=1.0)

            # 5. Entity-to-Entity co-occurrence edges within evidence sentence
            if len(ev_entities) >= 2:
                for i in range(len(ev_entities)):
                    for j in range(i + 1, len(ev_entities)):
                        e1_id = f"entity:{ev_entities[i].text.strip().lower()}"
                        e2_id = f"entity:{ev_entities[j].text.strip().lower()}"
                        if e1_id != e2_id:
                            graph.add_edge(
                                e1_id,
                                e2_id,
                                relation="CO_OCCURS",
                                weight=ev.rerank_score,
                                source_evidence=ev_id,
                            )

        logger.info(
            "Evidence graph constructed with %d nodes and %d edges.",
            len(graph.get_nodes()),
            len(graph.get_edges()),
        )
        return graph
