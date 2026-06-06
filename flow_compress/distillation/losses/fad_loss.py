"""
Flow-based alignment and adaptive weighting loss for flow-based distillation.
"""

from typing import Dict

from flow_compress.distillation.flows.alignment import (
    compute_dalign,
    compute_fad_loss_integrated,
    depth_weighting,
)
import torch
import torch.nn as nn


class FADLoss(nn.Module):
    """
    Implementation of LFAD + adaptive weighting L_task, L_KD, LFAD
    """

    def __init__(
        self,
        alpha_flow: float = 1.0,
        use_integrated_fad: bool = True,
        num_integration_points: int = 100,
    ):
        super().__init__()
        self.alpha_flow = alpha_flow
        self.use_integrated_fad = use_integrated_fad
        self.num_integration_points = num_integration_points

    def _compute_adaptive_weights(
        self,
        teacher_div: Dict[str, torch.Tensor],
        student_div: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes adaptive weights α, β, γ
        """

        # Compute average flow divergences across depths
        d_bar_T = torch.stack(list(teacher_div.values())).mean()
        d_bar_S = torch.stack(list(student_div.values())).mean()

        # Add small epsilon to avoid division by zero
        eps = 1e-8
        d_sum = d_bar_T + d_bar_S + eps

        # Compute components
        d_s_over_sum = d_bar_S / d_sum
        d_t_over_sum = d_bar_T / d_sum

        # Compute normalization constant Z
        Z = 1.0 + d_s_over_sum + d_t_over_sum

        # Compute weights
        alpha = 1.0 / Z
        beta = d_s_over_sum / Z
        gamma = d_t_over_sum / Z

        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }

    def forward(
        self,
        teacher_div: Dict[str, torch.Tensor],
        student_div: Dict[str, torch.Tensor],
        base_task_loss: torch.Tensor,
        base_kd_loss: torch.Tensor,
    ) -> tuple:
        """
        Main interface implementing LFAD + adaptive weighting L_task, L_KD, LFAD
        """

        # Compute L_FAD
        if self.use_integrated_fad:
            l_fad = compute_fad_loss_integrated(
                teacher_div,
                student_div,
                num_integration_points=self.num_integration_points,
                use_depth_weighting=True,
            )
        else:
            dalign = compute_dalign(teacher_div, student_div)
            l_fad = dalign

        # Apply base multiplier
        l_fad = self.alpha_flow * l_fad

        # Compute adaptive weights α, β, γ
        with torch.no_grad():
            weights = self._compute_adaptive_weights(
                teacher_div,
                student_div,
                base_task_loss.device,
            )
            alpha = weights["alpha"]
            beta = weights["beta"]
            gamma = weights["gamma"]

        # Compute L_total
        l_total = alpha * base_task_loss + beta * base_kd_loss + gamma * l_fad

        return l_total, {
            "L_task": base_task_loss.detach(),
            "L_kd": base_kd_loss.detach(),
            "L_fad": l_fad.detach(),
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "gamma": gamma.detach(),
            "lambda_task": alpha.detach(),
            "lambda_kd": beta.detach(),
            "lambda_fad": gamma.detach(),
        }
