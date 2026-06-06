"""
Representational geometry analysis for teacher model.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationalGeometryAnalyzer:
    """
    Analyzes the representational geometry of a neural network model.
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

    def compute_representational_space(
        self,
        data_loader: torch.utils.data.DataLoader,
        labels: Optional[torch.Tensor] = None,
        num_samples: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the representational space for each layer.
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
        all_labels = []
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x = batch
                    y = None

                x = x.to(self.device)
                _ = self.model(x)

                if y is not None:
                    all_labels.append(y.cpu())

                sample_count += x.size(0)

        # Remove hooks
        for handle in hooks:
            handle.remove()

        # Process activations
        representational_spaces = {}
        for name, acts in activations.items():
            if not acts:
                continue

            # Stack activations
            stacked = torch.cat(acts, dim=0)  # (N, ...)

            # Flatten to feature vectors
            flat = stacked.view(stacked.size(0), -1)  # (N, F)

            representational_spaces[name] = {
                "representations": flat,  # (N, F)
                "labels": torch.cat(all_labels) if all_labels else None,
            }

        return representational_spaces

    def compute_representational_diversity(
        self,
        representational_spaces: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Computes representational diversity for each layer.
        """

        diversities = {}

        for name, space in representational_spaces.items():
            reps = space["representations"]  # (N, F)

            # Compute pairwise distances
            # Use a sample for efficiency
            sample_size = min(500, reps.size(0))
            indices = torch.randperm(reps.size(0))[:sample_size]
            sample_reps = reps[indices]

            # Normalize
            sample_reps = F.normalize(sample_reps, p=2, dim=1)

            # Compute pairwise cosine distances
            distances = torch.cdist(sample_reps, sample_reps, p=2)

            # Diversity = average pairwise distance
            # Mask diagonal
            mask = ~torch.eye(sample_size, dtype=torch.bool)
            diversity = distances[mask].mean().item()

            diversities[name] = diversity

        return diversities

    def compute_representational_separability(
        self,
        representational_spaces: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Computes representational separability for each layer.
        """

        separabilities = {}

        for name, space in representational_spaces.items():
            reps = space["representations"]  # (N, F)
            labels = space.get("labels")

            if labels is None:
                separabilities[name] = 0.0
                continue

            # Compute class centroids
            unique_labels = torch.unique(labels)
            centroids = []

            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 0:
                    centroid = reps[mask].mean(dim=0)
                    centroids.append(centroid)

            if len(centroids) < 2:
                separabilities[name] = 0.0
                continue

            # Compute inter-class distances
            centroids = torch.stack(centroids)  # (C, F)
            inter_class_dist = torch.cdist(centroids, centroids, p=2)

            # Mask diagonal
            mask = ~torch.eye(len(centroids), dtype=torch.bool)
            avg_inter_class_dist = inter_class_dist[mask].mean().item()

            # Compute intra-class variance
            intra_class_vars = []
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 1:
                    class_reps = reps[mask]
                    centroid = class_reps.mean(dim=0)
                    var = ((class_reps - centroid) ** 2).mean().item()
                    intra_class_vars.append(var)

            avg_intra_class_var = np.mean(
                intra_class_vars) if intra_class_vars else 0.0

            # Separability = inter-class distance / intra-class variance
            separability = avg_inter_class_dist / (avg_intra_class_var + 1e-8)
            separabilities[name] = separability

        return separabilities

    def compute_representational_alignment(
        self,
        teacher_spaces: Dict[str, Dict],
        student_spaces: Dict[str, Dict],
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Computes representational alignment between teacher and student.
        """

        if layer_mapping is None:
            common_layers = set(teacher_spaces.keys()) & set(
                student_spaces.keys())
            layer_mapping = {layer: layer for layer in common_layers}

        if not layer_mapping:
            return 0.0

        alignments = []

        for t_layer, s_layer in layer_mapping.items():
            if t_layer not in teacher_spaces or s_layer not in student_spaces:
                continue

            t_space = teacher_spaces[t_layer]
            s_space = student_spaces[s_layer]

            t_reps = t_space["representations"]  # (N, F_t)
            s_reps = s_space["representations"]  # (N, F_s)

            # Align sample size
            min_n = min(t_reps.size(0), s_reps.size(0))
            t_reps = t_reps[:min_n]
            s_reps = s_reps[:min_n]

            # Project to same dimension (use PCA or linear projection)
            min_dim = min(t_reps.size(1), s_reps.size(1))
            t_reps = t_reps[:, :min_dim]
            s_reps = s_reps[:, :min_dim]

            # Normalize
            t_reps = F.normalize(t_reps, p=2, dim=1)
            s_reps = F.normalize(s_reps, p=2, dim=1)

            # Compute correlation between representations
            # Use canonical correlation analysis (simplified)
            t_cov = torch.cov(t_reps.T)
            s_cov = torch.cov(s_reps.T)
            cross_cov = torch.cov(torch.cat([t_reps.T, s_reps.T], dim=0))

            # Simplified alignment: correlation of principal components
            t_U, t_S, _ = torch.svd(t_cov)
            s_U, s_S, _ = torch.svd(s_cov)

            # Compare top principal directions
            k = min(10, t_U.size(1), s_U.size(1))
            t_top = t_U[:, :k]
            s_top = s_U[:, :k]

            # Compute alignment
            alignment = (t_top * s_top).sum().abs().item() / k
            alignments.append(alignment)

        return np.mean(alignments) if alignments else 0.0

    def analyze(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 1000,
    ) -> Dict[str, Dict]:
        """
        Comprehensive representational geometry analysis.
        """

        # Compute representational spaces
        spaces = self.compute_representational_space(
            data_loader, num_samples=num_samples
        )

        # Compute properties
        diversities = self.compute_representational_diversity(spaces)
        separabilities = self.compute_representational_separability(spaces)

        return {
            "spaces": spaces,
            "diversities": diversities,
            "separabilities": separabilities,
        }
