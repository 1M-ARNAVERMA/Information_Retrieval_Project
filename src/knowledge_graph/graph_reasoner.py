"""Multi-hop reasoning, path finding, and bridge entity discovery over evidence graphs."""

from typing import Any, Dict, List, Optional, Set
import networkx as nx

from src.knowledge_graph.evidence_graph import EvidenceGraph
from src.utils.logger import get_logger

logger = get_logger("graph_reasoner")


class GraphReasoner:
    """Performs multi-hop path analysis and bridge discovery over an EvidenceGraph."""

    def __init__(self, evidence_graph: EvidenceGraph) -> None:
        self.evidence_graph = evidence_graph
        # Convert to undirected graph for flexible multi-hop path traversal
        self.undirected = evidence_graph.graph.to_undirected()

    def find_paths_between_entities(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 3,
    ) -> List[List[str]]:
        """Finds shortest multi-hop paths connecting two entities in the graph.

        Args:
            entity_a: Text name or node ID of first entity.
            entity_b: Text name or node ID of second entity.
            max_hops: Maximum path length allowed.

        Returns:
            List of paths (each path is a list of node IDs).
        """
        id_a = entity_a if self.undirected.has_node(entity_a) else f"entity:{entity_a.strip().lower()}"
        id_b = entity_b if self.undirected.has_node(entity_b) else f"entity:{entity_b.strip().lower()}"

        if not self.undirected.has_node(id_a) or not self.undirected.has_node(id_b):
            return []

        try:
            paths = list(
                nx.all_simple_paths(
                    self.undirected,
                    source=id_a,
                    target=id_b,
                    cutoff=max_hops,
                )
            )
            # Sort by shortest length first
            paths.sort(key=len)
            return paths[:5]
        except Exception as e:
            logger.debug("Path search between %s and %s failed: %s", id_a, id_b, e)
            return []

    def find_bridge_entities(self) -> List[Dict[str, Any]]:
        """Identifies intermediate 'bridge' entities that connect disjoint evidence items.

        A bridge entity is an entity connected to 2 or more distinct evidence sentences.
        """
        bridges: List[Dict[str, Any]] = []

        for node_id, data in self.evidence_graph.graph.nodes(data=True):
            if data.get("label") != "ENTITY":
                continue

            # Find neighboring evidence nodes
            neighbors = list(self.undirected.neighbors(node_id))
            ev_neighbors = [
                n for n in neighbors
                if self.evidence_graph.graph.nodes[n].get("label") == "EVIDENCE"
            ]

            if len(ev_neighbors) >= 2:
                bridges.append({
                    "bridge_entity": data.get("text", node_id),
                    "entity_type": data.get("entity_type", "ENTITY"),
                    "connected_evidence_count": len(ev_neighbors),
                    "connected_evidence_nodes": ev_neighbors,
                })

        # Sort bridges by number of connections
        bridges.sort(key=lambda x: x["connected_evidence_count"], reverse=True)
        return bridges

    def get_multi_hop_traces(self) -> List[Dict[str, Any]]:
        """Summarizes multi-hop reasoning chains connecting evidence sentences to claim entities."""
        traces = []
        claim_id = "claim:root"
        if not self.undirected.has_node(claim_id):
            return traces

        # Create a graph copy without direct claim-to-evidence veracity edges
        # to reveal the underlying multi-hop entity reasoning path
        entity_graph = self.undirected.copy()
        for node_id, data in self.evidence_graph.graph.nodes(data=True):
            if data.get("label") == "EVIDENCE" and entity_graph.has_edge(claim_id, node_id):
                entity_graph.remove_edge(claim_id, node_id)

        # Find entity paths from claim to all evidence nodes
        for node_id, data in self.evidence_graph.graph.nodes(data=True):
            if data.get("label") == "EVIDENCE":
                try:
                    if nx.has_path(entity_graph, claim_id, node_id):
                        path = nx.shortest_path(entity_graph, claim_id, node_id)
                        traces.append({
                            "evidence_id": node_id,
                            "hops": len(path) - 1,
                            "reasoning_path": path,
                            "verdict": data.get("verdict"),
                        })
                except Exception:
                    continue

        return traces
