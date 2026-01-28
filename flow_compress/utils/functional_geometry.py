"""
Functional geometry analysis for teacher model.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionalGeometryAnalyzer:
    """
    Analyzes the functional geometry of a neural network model.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.layer_names = layer_names
        self.device = device
        self.model.eval()

        # Cache for computed properties
        self._functional_properties: Dict[str, Dict] = {}

    def compute_functional_manifold(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the functional manifold for each layer.
        """

        activations = {}
        hooks = []

        # Register hooks
        def make_hook(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                if isinstance(output, torch.Tensor):
                    activations[name].append(output.detach().cpu())

            return hook

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)

        # Collect activations
        sample_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                _ = self.model(x)
                sample_count += x.size(0)

        # Remove hooks
        for handle in hooks:
            handle.remove()

        # Compute manifold representations
        manifolds = {}
        for name, acts in activations.items():
            if not acts:
                continue

            # Stack activations
            stacked = torch.cat(acts, dim=0)  # (N, ...)

            # Flatten to feature vectors
            flat = stacked.view(stacked.size(0), -1)  # (N, F)

            # Compute principal components (functional directions)
            U, S, Vt = torch.svd(flat.T)  # Vt: (F, min(N, F))

            # Store top-k principal components
            k = min(50, Vt.size(1))
            manifolds[name] = {
                "principal_components": Vt[:k].T,  # (k, F)
                "singular_values": S[:k],  # (k,)
                "mean": flat.mean(dim=0),  # (F,)
                "covariance": torch.cov(flat.T),  # (F, F)
            }

        return manifolds

    def compute_functional_complexity(
        self,
        manifolds: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Computes functional complexity for each layer.
        """

        complexities = {}

        for name, manifold in manifolds.items():
            # Use effective rank (based on singular values)
            singular_values = manifold["singular_values"]

            # Normalize singular values
            normalized = singular_values / (singular_values.sum() + 1e-8)

            # Effective rank (entropy-based)
            entropy = -(normalized * torch.log(normalized + 1e-8)).sum()
            effective_rank = torch.exp(entropy)

            # Complexity is proportional to effective rank
            complexities[name] = effective_rank.item()

        return complexities

    def compute_functional_criticality(
        self,
        manifolds: Dict[str, Dict],
        downstream_layers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Computes functional criticality for each layer.
        """

        if downstream_layers is None:
            downstream_layers = self.layer_names

        criticalities = {}

        for name, manifold in manifolds.items():
            # Criticality based on variance explained
            singular_values = manifold["singular_values"]

            # Variance explained by top components
            total_variance = (singular_values**2).sum()
            top_k_variance = (singular_values[:10] ** 2).sum()
            variance_ratio = top_k_variance / (total_variance + 1e-8)

            # Higher variance concentration = higher criticality
            criticalities[name] = variance_ratio.item()

        return criticalities

    def compute_functional_alignment(
        self,
        teacher_manifolds: Dict[str, Dict],
        student_manifolds: Dict[str, Dict],
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Computes functional alignment between teacher and student.
        """

        if layer_mapping is None:
            # Use common layers
            common_layers = set(teacher_manifolds.keys()) & set(
                student_manifolds.keys()
            )
            layer_mapping = {layer: layer for layer in common_layers}

        if not layer_mapping:
            return 0.0

        alignments = []

        for t_layer, s_layer in layer_mapping.items():
            if t_layer not in teacher_manifolds or s_layer not in student_manifolds:
                continue

            t_manifold = teacher_manifolds[t_layer]
            s_manifold = student_manifolds[s_layer]

            # Compare principal components
            t_pc = t_manifold["principal_components"]  # (k, F)
            s_pc = s_manifold["principal_components"]  # (k, F)

            # Align dimensions (pad or truncate)
            min_k = min(t_pc.size(0), s_pc.size(0))
            t_pc = t_pc[:min_k]
            s_pc = s_pc[:min_k]

            # Compute cosine similarity between principal directions
            similarities = []
            for i in range(min_k):
                t_dir = t_pc[i]
                s_dir = s_pc[i]

                # Normalize
                t_dir = F.normalize(t_dir, p=2, dim=0)
                s_dir = F.normalize(s_dir, p=2, dim=0)

                # Cosine similarity
                sim = (t_dir * s_dir).sum().item()
                similarities.append(sim)

            # Average alignment
            alignment = np.mean(similarities)
            alignments.append(alignment)

        return np.mean(alignments) if alignments else 0.0

    def analyze(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 1000,
    ) -> Dict[str, Dict]:
        """
        Comprehensive functional geometry analysis.
        """

        # Compute manifolds
        manifolds = self.compute_functional_manifold(data_loader, num_samples)

        # Compute properties
        complexities = self.compute_functional_complexity(manifolds)
        criticalities = self.compute_functional_criticality(manifolds)

        return {
            "manifolds": manifolds,
            "complexities": complexities,
            "criticalities": criticalities,
        }
