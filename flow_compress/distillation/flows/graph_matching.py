"""
Learned dynamic graph matching algorithm for topology-agnostic knowledge transfer.
"""

from typing import Dict, List, Tuple, Optional, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_compress.distillation.flows.graph_representation import ComputationalGraph, GraphNode


class GraphMatchingNetwork(nn.Module):
    """
    Neural network for learning graph matching.
    """

    def __init__(
        self,
        signature_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        self.signature_dim = signature_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.node_proj = nn.Linear(signature_dim, hidden_dim)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-head attention for matching
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        # Matching score computation
        self.match_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        teacher_adj: Optional[torch.Tensor] = None,
        student_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matching scores between teacher and student nodes.
        """

        # Project to hidden dimension
        t_feat = self.node_proj(teacher_features)  # (T, hidden_dim)
        s_feat = self.node_proj(student_features)  # (S, hidden_dim)

        # Apply attention layers
        t_encoded = t_feat
        s_encoded = s_feat

        for attn, norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention on teacher
            t_encoded, _ = attn(t_encoded, t_encoded, t_encoded)
            t_encoded = norm(t_encoded + t_feat)

            # Self-attention on student
            s_encoded, _ = attn(s_encoded, s_encoded, s_encoded)
            s_encoded = norm(s_encoded + s_feat)

        # Compute pairwise matching scores
        T, H = t_encoded.shape
        S, _ = s_encoded.shape

        # Expand for pairwise comparison
        t_expanded = t_encoded.unsqueeze(1).expand(T, S, H)  # (T, S, H)
        s_expanded = s_encoded.unsqueeze(0).expand(T, S, H)  # (T, S, H)

        # Concatenate and compute matching score
        paired = torch.cat([t_expanded, s_expanded], dim=-1)  # (T, S, 2H)
        matching_scores = self.match_score(paired).squeeze(-1)  # (T, S)

        return matching_scores, None


class DynamicGraphMatcher:
    """
    Dynamic graph matching algorithm that learns to find functionally analogous sub-structures between teacher and student networks.
    """

    def __init__(
        self,
        signature_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        temperature: float = 1.0,
        use_structural_similarity: bool = True,
    ):
        self.matching_network = GraphMatchingNetwork(
            signature_dim=signature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.temperature = temperature
        self.use_structural_similarity = use_structural_similarity

        # Structural similarity weight
        self.structural_weight = nn.Parameter(torch.tensor(0.5))

    def compute_matching(
        self,
        teacher_graph: ComputationalGraph,
        student_graph: ComputationalGraph,
        teacher_activations: Optional[Dict[str, torch.Tensor]] = None,
        student_activations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, str]:
        """
        Compute dynamic matching between teacher and student graphs.
        """

        # Update signatures with activations if available
        if teacher_activations:
            teacher_graph.update_signatures(teacher_activations)
        if student_activations:
            student_graph.update_signatures(student_activations)

        # Get node features
        t_features = teacher_graph.get_node_features()  # (T, signature_dim)
        s_features = student_graph.get_node_features()  # (S, signature_dim)

        # Get adjacency matrices
        t_adj = teacher_graph.get_adjacency_matrix()
        s_adj = student_graph.get_adjacency_matrix()

        # Compute functional matching scores
        matching_scores, _ = self.matching_network(
            t_features, s_features, t_adj, s_adj
        )  # (T, S)

        # Add structural similarity if enabled
        if self.use_structural_similarity:
            structural_sim = self._compute_structural_similarity(
                teacher_graph, student_graph, t_adj, s_adj
            )
            matching_scores = matching_scores + self.structural_weight * structural_sim

        # Apply temperature scaling
        matching_scores = matching_scores / self.temperature

        # Compute hard matching (one-to-one or many-to-one)
        matching = self._compute_hard_matching(
            matching_scores,
            teacher_graph.node_order,
            student_graph.node_order,
        )

        return matching

    def _compute_structural_similarity(
        self,
        teacher_graph: ComputationalGraph,
        student_graph: ComputationalGraph,
        t_adj: torch.Tensor,
        s_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute structural similarity between nodes based on graph topology.
        """

        T = len(teacher_graph.node_order)
        S = len(student_graph.node_order)
        structural_sim = torch.zeros(T, S)

        # Compute node degrees
        t_degrees = t_adj.sum(dim=1)  # (T,)
        s_degrees = s_adj.sum(dim=1)  # (S,)

        # Normalize degrees
        t_deg_norm = t_degrees / (t_degrees.max() + 1e-8)
        s_deg_norm = s_degrees / (s_degrees.max() + 1e-8)

        # Compute similarity based on degree similarity
        for i in range(T):
            for j in range(S):
                deg_sim = 1.0 - abs(t_deg_norm[i] - s_deg_norm[j])
                structural_sim[i, j] = deg_sim

        return structural_sim

    def _compute_hard_matching(
        self,
        matching_scores: torch.Tensor,
        teacher_nodes: List[str],
        student_nodes: List[str],
    ) -> Dict[str, str]:
        """
        Compute hard matching from soft matching scores.
        """

        T, S = matching_scores.shape

        # Use greedy matching (many-to-one allowed)
        # Each teacher node matches to best student node
        matching = {}

        # For each teacher node, find best student match
        for i, t_node in enumerate(teacher_nodes):
            best_j = matching_scores[i].argmax().item()
            s_node = student_nodes[best_j]
            matching[t_node] = s_node

        return matching

    def get_matching_parameters(self) -> List[torch.nn.Parameter]:
        """Get parameters of the matching network for optimization."""

        return list(self.matching_network.parameters()) + [self.structural_weight]

    def update_temperature(self, new_temperature: float):
        """Update temperature for matching."""

        self.temperature = new_temperature


def compute_graph_matching(
    teacher_graph: ComputationalGraph,
    student_graph: ComputationalGraph,
    matcher: DynamicGraphMatcher,
    teacher_activations: Optional[Dict[str, torch.Tensor]] = None,
    student_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, str]:
    """
    Convenience function to compute graph matching.
    """

    return matcher.compute_matching(
        teacher_graph,
        student_graph,
        teacher_activations,
        student_activations,
    )
