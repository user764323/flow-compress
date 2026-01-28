"""
Computational graph representation for topology-agnostic knowledge transfer.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNode:
    """
    Represents a node in the computational graph.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        node_type: str,
        depth: int,
        in_features: Optional[Tuple] = None,
        out_features: Optional[Tuple] = None,
        connectivity_pattern: str = "sequential",
    ):
        self.name = name
        self.module = module
        self.node_type = node_type
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.connectivity_pattern = connectivity_pattern

        # Graph structure
        self.predecessors: List[str] = []
        self.successors: List[str] = []

        # Functional signature (for matching)
        self.functional_signature: Optional[torch.Tensor] = None

    def compute_functional_signature(
        self,
        activations: Optional[torch.Tensor] = None,
        embedding_dim: int = 128,
    ) -> torch.Tensor:
        """
        Computes a functional signature for this node.
        """

        if self.functional_signature is not None:
            return self.functional_signature

        # Build signature from multiple sources
        signature_parts = []

        # 1. Node type encoding
        type_encoding = self._encode_node_type(
            self.node_type, embedding_dim // 4)
        signature_parts.append(type_encoding)

        # 2. Connectivity pattern encoding
        conn_encoding = self._encode_connectivity(
            self.connectivity_pattern, embedding_dim // 4
        )
        signature_parts.append(conn_encoding)

        # 3. Depth encoding (normalized)
        depth_encoding = self._encode_depth(self.depth, embedding_dim // 4)
        signature_parts.append(depth_encoding)

        # 4. Feature statistics (if activations available)
        if activations is not None:
            feat_encoding = self._encode_feature_stats(
                activations, embedding_dim // 4)
            signature_parts.append(feat_encoding)
        else:
            # Use module parameters as proxy
            param_encoding = self._encode_module_params(embedding_dim // 4)
            signature_parts.append(param_encoding)

        # Concatenate and normalize
        signature = torch.cat(signature_parts, dim=0)
        signature = F.normalize(signature, p=2, dim=0)

        self.functional_signature = signature
        return signature

    def _encode_node_type(self, node_type: str, dim: int) -> torch.Tensor:
        """Encode node type as embedding."""

        # Simple hash-based encoding
        hash_val = hash(node_type) % (2**16)
        encoding = torch.zeros(dim)
        for i in range(dim):
            encoding[i] = np.sin(hash_val / (10000 ** (2 * i / dim)))
        return encoding

    def _encode_connectivity(self, pattern: str, dim: int) -> torch.Tensor:
        """Encode connectivity pattern."""

        patterns = {"sequential": 0, "residual": 1, "dense": 2, "attention": 3}
        idx = patterns.get(pattern, 0)
        encoding = torch.zeros(dim)
        encoding[idx % dim] = 1.0
        return encoding

    def _encode_depth(self, depth: int, dim: int) -> torch.Tensor:
        """Encode normalized depth."""

        normalized_depth = depth / max(1, depth + 1)  # Normalize to [0, 1]
        encoding = torch.zeros(dim)
        for i in range(dim):
            encoding[i] = np.sin(normalized_depth * np.pi * (i + 1))
        return encoding

    def _encode_feature_stats(
        self, activations: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Encode feature statistics from activations."""

        if activations.numel() == 0:
            return torch.zeros(dim)

        flat = activations.view(activations.size(0), -1)

        # Compute statistics
        mean_val = flat.mean().item()
        std_val = flat.std().item() + 1e-8
        max_val = flat.max().item()
        min_val = flat.min().item()

        # Encode statistics
        stats = torch.tensor([mean_val, std_val, max_val, min_val])
        encoding = torch.zeros(dim)
        for i in range(min(dim, len(stats))):
            encoding[i] = stats[i]
        # Fill rest with periodic encoding
        for i in range(len(stats), dim):
            encoding[i] = np.sin(mean_val * (i + 1))

        return encoding

    def _encode_module_params(self, dim: int) -> torch.Tensor:
        """Encode module parameters as proxy for functional role."""
        encoding = torch.zeros(dim)

        if hasattr(self.module, "weight") and self.module.weight is not None:
            weight = self.module.weight.data
            # Use weight statistics
            encoding[0] = weight.numel()
            encoding[1] = weight.mean().item()
            encoding[2] = weight.std().item() + 1e-8
            if len(encoding) > 3:
                encoding[3] = weight.norm().item()

        return encoding


class ComputationalGraph:
    """
    Represents a neural network as a computational graph.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.nodes: Dict[str, GraphNode] = {}
        self.node_order: List[str] = []

        self._build_graph()

    def _build_graph(self):
        """Build computational graph from model."""
        # Create nodes
        for idx, name in enumerate(self.layer_names):
            module = self._get_module_by_name(name)
            if module is None:
                continue

            node_type = self._classify_module(module)
            connectivity = self._infer_connectivity(name, module)

            node = GraphNode(
                name=name,
                module=module,
                node_type=node_type,
                depth=idx,
                connectivity_pattern=connectivity,
            )
            self.nodes[name] = node
            self.node_order.append(name)

        # Build edges (predecessors/successors)
        self._build_edges()

    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by its name."""

        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _classify_module(self, module: nn.Module) -> str:
        """Classify module type."""

        module_type = type(module).__name__
        return module_type

    def _infer_connectivity(self, name: str, module: nn.Module) -> str:
        """Infer connectivity pattern from module and context."""

        # Simple heuristics - can be enhanced
        if "residual" in name.lower() or "shortcut" in name.lower():
            return "residual"
        elif "attention" in name.lower() or "attn" in name.lower():
            return "attention"
        elif "dense" in name.lower():
            return "dense"
        else:
            return "sequential"

    def _build_edges(self):
        """Build edges between nodes based on execution order."""
        for i in range(len(self.node_order) - 1):
            curr_name = self.node_order[i]
            next_name = self.node_order[i + 1]

            if curr_name in self.nodes and next_name in self.nodes:
                self.nodes[curr_name].successors.append(next_name)
                self.nodes[next_name].predecessors.append(curr_name)

    def update_signatures(self, activations: Dict[str, torch.Tensor]):
        """
        Update functional signatures using activations.
        """

        for name, node in self.nodes.items():
            if name in activations:
                node.compute_functional_signature(activations[name])
            else:
                node.compute_functional_signature()

    def get_subgraph(
        self,
        node_names: List[str],
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> "ComputationalGraph":
        """
        Extract a subgraph containing specified nodes.
        """

        if include_neighbors:
            expanded_names = set(node_names)
            for _ in range(neighbor_depth):
                new_nodes = set()
                for name in expanded_names:
                    if name in self.nodes:
                        new_nodes.update(self.nodes[name].predecessors)
                        new_nodes.update(self.nodes[name].successors)
                expanded_names.update(new_nodes)
            node_names = list(expanded_names)

        # Create subgraph
        subgraph = ComputationalGraph.__new__(ComputationalGraph)
        subgraph.model = self.model
        subgraph.layer_names = [n for n in self.layer_names if n in node_names]
        subgraph.nodes = {n: self.nodes[n]
                          for n in node_names if n in self.nodes}
        subgraph.node_order = [n for n in self.node_order if n in node_names]

        return subgraph

    def get_node_features(self) -> torch.Tensor:
        """
        Get feature matrix of all nodes.
        """

        signatures = []
        for name in self.node_order:
            if name in self.nodes:
                sig = self.nodes[name].compute_functional_signature()
                signatures.append(sig)

        if not signatures:
            return torch.empty(0, 128)

        return torch.stack(signatures)

    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get adjacency matrix of the graph.
        """

        num_nodes = len(self.node_order)
        adj = torch.zeros(num_nodes, num_nodes)

        name_to_idx = {name: idx for idx, name in enumerate(self.node_order)}

        for name, node in self.nodes.items():
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            for succ in node.successors:
                if succ in name_to_idx:
                    succ_idx = name_to_idx[succ]
                    adj[idx, succ_idx] = 1.0

        return adj
