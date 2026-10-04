"""Unit tests for Phase 12: Knowledge & Evidence Graph Reasoning."""

import pytest

from src.common.types import (
    Entity,
    EvidenceGraphData,
    NLIPrediction,
    RerankedResult,
    SubClaim,
    VeracityLabel,
)
from src.knowledge_graph.evidence_graph import EvidenceGraph
from src.knowledge_graph.graph_builder import EvidenceGraphBuilder
from src.knowledge_graph.graph_reasoner import GraphReasoner


def test_evidence_graph_basic():
    """Validates node/edge addition and Pydantic EvidenceGraphData serialization."""
    g = EvidenceGraph()
    g.add_node("claim:0", label="CLAIM", text="Einstein relativity.")
    g.add_node("ent:einstein", label="ENTITY", text="Albert Einstein")
    g.add_edge("claim:0", "ent:einstein", relation="MENTIONS", weight=1.0)

    assert g.has_node("claim:0")
    assert g.has_node("ent:einstein")
    assert not g.has_node("nonexistent")

    data = g.to_graph_data()
    assert isinstance(data, EvidenceGraphData)
    assert len(data.nodes) == 2
    assert len(data.edges) == 1
    assert data.edges[0].relation == "MENTIONS"

    # Test dictionary export
    d = g.to_dict()
    assert "nodes" in d
    assert "edges" in d


def test_graph_builder_and_reasoner():
    """Validates end-to-end evidence graph construction and multi-hop reasoning."""
    builder = EvidenceGraphBuilder()

    claim = "Marie Curie was born in Poland and won two Nobel Prizes."
    entities = [
        Entity(text="Marie Curie", label="PERSON"),
        Entity(text="Poland", label="GPE"),
    ]
    subclaims = [
        SubClaim(id=1, text="Marie Curie was born in Poland."),
        SubClaim(id=2, text="Marie Curie won two Nobel Prizes."),
    ]

    ev1 = RerankedResult(
        doc_id="Marie Curie",
        sentence_id=0,
        text="Marie Curie was a Polish physicist born in Poland.",
        retrieval_score=0.9,
        rerank_score=0.95,
        retriever_origin="bm25",
    )
    ev2 = RerankedResult(
        doc_id="Marie Curie",
        sentence_id=2,
        text="She won the Nobel Prize in Physics and Chemistry.",
        retrieval_score=0.85,
        rerank_score=0.92,
        retriever_origin="bm25",
    )

    preds = [
        NLIPrediction(
            doc_id="Marie Curie",
            sentence_id=0,
            text=ev1.text,
            entailment_prob=0.95,
            contradiction_prob=0.02,
            neutral_prob=0.03,
            predicted_label=VeracityLabel.SUPPORTS,
        ),
        NLIPrediction(
            doc_id="Marie Curie",
            sentence_id=2,
            text=ev2.text,
            entailment_prob=0.88,
            contradiction_prob=0.04,
            neutral_prob=0.08,
            predicted_label=VeracityLabel.SUPPORTS,
        ),
    ]

    graph = builder.build_graph(
        claim=claim,
        entities=entities,
        evidence_list=[ev1, ev2],
        nli_predictions=preds,
        subclaims=subclaims,
    )

    # Validate graph nodes
    nodes = graph.get_nodes()
    labels = {n.label for n in nodes}
    assert "CLAIM" in labels
    assert "SUBCLAIM" in labels
    assert "ENTITY" in labels
    assert "EVIDENCE" in labels

    # Validate reasoner
    reasoner = GraphReasoner(graph)

    # 1. Bridge entities (Marie Curie is mentioned in both evidence sentences)
    bridges = reasoner.find_bridge_entities()
    assert len(bridges) >= 1
    bridge_names = [b["bridge_entity"].lower() for b in bridges]
    assert any("marie curie" in name for name in bridge_names)

    # 2. Multi-hop path between entities
    paths = reasoner.find_paths_between_entities("Marie Curie", "Poland", max_hops=4)
    assert len(paths) > 0

    # 3. Multi-hop traces from claim to evidence
    traces = reasoner.get_multi_hop_traces()
    assert len(traces) > 0
