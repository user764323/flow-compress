"""
Unified variational divergence functional with meta-optimized parameters.
"""

from typing import Any, Dict, List, Optional, Tuple

from flow_compress.distillation.flows.alignment import compute_dalign
from flow_compress.distillation.losses.kd_logits import LogitsKDLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaOptimizedParameters(nn.Module):
    """
    Meta-learned parameters for the variational divergence functional.
    """

    def __init__(
        self,
        initial_temperature: float = 4.0,
        initial_alpha_logit: float = 1.0,
        initial_alpha_feature: float = 1.0,
        initial_alpha_relation: float = 1.0,
        initial_alpha_flow: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()

        # Temperature parameter (with softplus to ensure positivity)
        self.temperature_log = nn.Parameter(
            torch.log(torch.tensor(initial_temperature)), requires_grad=learnable
        )

        # Weight parameters (with softmax normalization)
        self.alpha_logit_log = nn.Parameter(
            torch.log(torch.tensor(initial_alpha_logit)), requires_grad=learnable
        )
        self.alpha_feature_log = nn.Parameter(
            torch.log(torch.tensor(initial_alpha_feature)), requires_grad=learnable
        )
        self.alpha_relation_log = nn.Parameter(
            torch.log(torch.tensor(initial_alpha_relation)), requires_grad=learnable
        )
        self.alpha_flow_log = nn.Parameter(
            torch.log(torch.tensor(initial_alpha_flow)), requires_grad=learnable
        )

        # Focus parameter (controls attention to different knowledge types)
        self.focus_log = nn.Parameter(
            # Initial focus = 1.0 (exp(0))
            torch.tensor(0.0),
            requires_grad=learnable,
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature (always positive)."""

        return torch.exp(self.temperature_log) + 0.1  # Minimum 0.1

    @property
    def alpha_logit(self) -> torch.Tensor:
        """Get logit weight."""

        return torch.exp(self.alpha_logit_log)

    @property
    def alpha_feature(self) -> torch.Tensor:
        """Get feature weight."""

        return torch.exp(self.alpha_feature_log)

    @property
    def alpha_relation(self) -> torch.Tensor:
        """Get relation weight."""

        return torch.exp(self.alpha_relation_log)

    @property
    def alpha_flow(self) -> torch.Tensor:
        """Get flow weight."""

        return torch.exp(self.alpha_flow_log)

    @property
    def focus(self) -> torch.Tensor:
        """Get focus parameter (controls attention)."""

        return torch.exp(self.focus_log)

    def get_normalized_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get normalized weights that sum to 1.
        """

        weights = torch.stack(
            [
                self.alpha_logit,
                self.alpha_feature,
                self.alpha_relation,
                self.alpha_flow,
            ]
        )
        normalized = F.softmax(weights, dim=0)

        return {
            "logit": normalized[0],
            "feature": normalized[1],
            "relation": normalized[2],
            "flow": normalized[3],
        }

    def adapt_to_progress(
        self,
        student_loss: torch.Tensor,
        teacher_loss: torch.Tensor,
        fidelity_score: torch.Tensor,
    ):
        """
        Adapt parameters based on student's learning progress.
        """

        # Compute progress indicators
        loss_ratio = student_loss / (teacher_loss + 1e-8)
        progress = 1.0 - loss_ratio  # Higher when student is doing better

        # Adapt temperature: lower temperature as student improves
        target_temp = 1.0 + 3.0 * (1.0 - progress)
        with torch.no_grad():
            self.temperature_log.data = torch.log(
                torch.clamp(target_temp, 0.1, 10.0))

        # Adapt focus: increase focus on difficult knowledge types
        target_focus = 1.0 + 0.5 * (1.0 - fidelity_score)
        with torch.no_grad():
            self.focus_log.data = torch.log(
                torch.clamp(target_focus, 0.5, 2.0))


class VariationalDivergenceFunctional(nn.Module):
    """
    Unified variational divergence functional
    """

    def __init__(
        self,
        initial_temperature: float = 4.0,
        learnable_params: bool = True,
    ):
        """
        Args:
            initial_temperature: Initial temperature for logit distillation
            learnable_params: Whether to meta-optimize parameters
        """

        super().__init__()

        self.meta_params = MetaOptimizedParameters(
            initial_temperature=initial_temperature,
            learnable=learnable_params,
        )

        # Logit-based loss (will use dynamic temperature)
        self.logit_loss_fn = LogitsKDLoss(temperature=initial_temperature)

    def forward(
        self,
        # Logit-based inputs
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        # Feature-based inputs
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        # Relation-based inputs
        teacher_relations: Optional[torch.Tensor] = None,
        student_relations: Optional[torch.Tensor] = None,
        # Flow-based inputs
        teacher_div: Optional[Dict[str, torch.Tensor]] = None,
        student_div: Optional[Dict[str, torch.Tensor]] = None,
        # Task loss
        task_loss: Optional[torch.Tensor] = None,
        # Additional context
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute unified variational divergence.
        """

        loss_components = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # Update temperature dynamically
        self.logit_loss_fn.temperature = self.meta_params.temperature.item()

        # 1. Logit-based divergence
        if teacher_logits is not None and student_logits is not None:
            l_logit = self.logit_loss_fn(student_logits, teacher_logits)
            weights = self.meta_params.get_normalized_weights()
            total_loss = total_loss + weights["logit"] * l_logit
            loss_components["logit"] = l_logit.detach()

        # 2. Feature-based divergence
        if teacher_features is not None and student_features is not None:
            l_feature = self._compute_feature_divergence(
                teacher_features, student_features, layer_mapping
            )
            weights = self.meta_params.get_normalized_weights()
            total_loss = total_loss + weights["feature"] * l_feature
            loss_components["feature"] = l_feature.detach()

        # 3. Relation-based divergence
        if teacher_relations is not None and student_relations is not None:
            l_relation = self._compute_relation_divergence(
                teacher_relations, student_relations
            )
            weights = self.meta_params.get_normalized_weights()
            total_loss = total_loss + weights["relation"] * l_relation
            loss_components["relation"] = l_relation.detach()

        # 4. Flow-based divergence
        if teacher_div is not None and student_div is not None:
            l_flow = self._compute_flow_divergence(teacher_div, student_div)
            weights = self.meta_params.get_normalized_weights()
            total_loss = total_loss + weights["flow"] * l_flow
            loss_components["flow"] = l_flow.detach()

        # Apply focus parameter (emphasizes important knowledge types)
        focus = self.meta_params.focus
        total_loss = total_loss * focus

        # Add task loss if provided
        if task_loss is not None:
            total_loss = total_loss + task_loss
            loss_components["task"] = task_loss.detach()

        # Store metadata
        loss_components["temperature"] = self.meta_params.temperature.detach()
        loss_components["focus"] = self.meta_params.focus.detach()
        loss_components["weights"] = {
            k: v.detach() for k, v in self.meta_params.get_normalized_weights().items()
        }

        return total_loss, loss_components

    def _compute_feature_divergence(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """
        Compute feature-based divergence using KL divergence on feature distributions.
        """

        if layer_mapping is None:
            # Use common layers
            common_layers = set(teacher_features.keys()) & set(
                student_features.keys())
            layer_mapping = {layer: layer for layer in common_layers}

        divergences = []
        for t_layer, s_layer in layer_mapping.items():
            if t_layer not in teacher_features or s_layer not in student_features:
                continue

            t_feat = teacher_features[t_layer]
            s_feat = student_features[s_layer]

            # Flatten features
            t_flat = t_feat.view(t_feat.size(0), -1)
            s_flat = s_feat.view(s_feat.size(0), -1)

            # Normalize to distributions
            t_norm = F.normalize(t_flat, p=2, dim=1)
            s_norm = F.normalize(s_flat, p=2, dim=1)

            # Compute KL divergence (symmetrized)
            t_probs = F.softmax(t_norm, dim=1) + 1e-8
            s_probs = F.softmax(s_norm, dim=1) + 1e-8

            kl_t_s = (t_probs * torch.log(t_probs / s_probs)).sum(dim=1).mean()
            kl_s_t = (s_probs * torch.log(s_probs / t_probs)).sum(dim=1).mean()

            divergence = 0.5 * (kl_t_s + kl_s_t)
            divergences.append(divergence)

        if not divergences:
            return torch.tensor(
                0.0, device=student_features[list(student_features.keys())[0]].device
            )

        return torch.stack(divergences).mean()

    def _compute_relation_divergence(
        self,
        teacher_relations: torch.Tensor,
        student_relations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relation-based divergence using pairwise feature relationships.

        Args:
            teacher_relations: Teacher relation matrix (B, N, N) or (N, N)
            student_relations: Student relation matrix (B, N, N) or (N, N)
        """

        # Ensure same shape
        if teacher_relations.dim() == 2:
            teacher_relations = teacher_relations.unsqueeze(0)
        if student_relations.dim() == 2:
            student_relations = student_relations.unsqueeze(0)

        # Normalize
        t_norm = F.normalize(
            teacher_relations.view(teacher_relations.size(0), -1), p=2, dim=1
        )
        s_norm = F.normalize(
            student_relations.view(student_relations.size(0), -1), p=2, dim=1
        )

        # Compute MSE divergence
        divergence = F.mse_loss(t_norm, s_norm)

        return divergence

    def _compute_flow_divergence(
        self,
        teacher_div: Dict[str, torch.Tensor],
        student_div: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute flow-based divergence using alignment metric.
        """

        return compute_dalign(teacher_div, student_div)

    def meta_optimize_step(
        self,
        student_loss: torch.Tensor,
        teacher_loss: torch.Tensor,
        fidelity_score: torch.Tensor,
    ):
        """
        Perform one step of meta-optimization to adapt parameters.
        """

        self.meta_params.adapt_to_progress(
            student_loss, teacher_loss, fidelity_score)

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get learnable parameters for optimization."""

        return list(self.meta_params.parameters())
