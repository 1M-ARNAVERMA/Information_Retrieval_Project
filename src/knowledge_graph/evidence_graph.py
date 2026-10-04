"""NetworkX-based Evidence Graph representation and serialization for FEVER-X."""

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from src.common.types import EvidenceGraphData, GraphEdge, GraphNode


class EvidenceGraph:
    """Directed graph representing entities, claims, evidence, and their semantic relationships."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, label: str, **attributes: Any) -> None:
        """Adds a node with category label and metadata attributes."""
        self.graph.add_node(str(node_id), label=label, **attributes)

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        **attributes: Any,
    ) -> None:
        """Adds a directed relation edge between two nodes."""
        self.graph.add_edge(
            str(source),
            str(target),
            relation=relation,
            weight=float(weight),
            **attributes,
        )

    def has_node(self, node_id: str) -> bool:
        """Checks if a node exists in the graph."""
        return self.graph.has_node(str(node_id))

    def get_nodes(self) -> List[GraphNode]:
        """Returns all graph nodes as typed GraphNode models."""
        nodes: List[GraphNode] = []
        for node_id, data in self.graph.nodes(data=True):
            label = data.get("label", "ENTITY")
            attrs = {k: v for k, v in data.items() if k != "label"}
            nodes.append(GraphNode(id=node_id, label=label, attributes=attrs))
        return nodes

    def get_edges(self) -> List[GraphEdge]:
        """Returns all graph edges as typed GraphEdge models."""
        edges: List[GraphEdge] = []
        for u, v, data in self.graph.edges(data=True):
            relation = data.get("relation", "CONNECTED_TO")
            weight = float(data.get("weight", 1.0))
            edges.append(GraphEdge(source=u, target=v, relation=relation, weight=weight))
        return edges

    def to_graph_data(self) -> EvidenceGraphData:
        """Serializes into standard EvidenceGraphData schema for frontend/API consumption."""
        return EvidenceGraphData(nodes=self.get_nodes(), edges=self.get_edges())

    def to_dict(self) -> Dict[str, Any]:
        """Converts graph into node-link JSON dictionary suitable for D3/Plotly."""
        return {
            "nodes": [
                {"id": n.id, "label": n.label, **n.attributes}
                for n in self.get_nodes()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation, "weight": e.weight}
                for e in self.get_edges()
            ],
        }
