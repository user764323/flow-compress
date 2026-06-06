"""
Student wrapper for flow-based distillation.
"""

from typing import Dict, List

from flow_compress.distillation.flows.divergence import compute_layerwise_flow_divergence
from flow_compress.distillation.flows.hooks import ActivationHook
import torch
import torch.nn as nn


class StudentWrapper(nn.Module):
    """
    Wrapper for student model: similar to TeacherWrapper, but D_T is calculated on the fly.
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_names: List[str],
    ):
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self.activation_hook = ActivationHook(self.backbone, layer_names)

    def forward_with_flows(self, x: torch.Tensor):
        """
        Forward with calculation of D_T for student (train mode).
        """

        self.activation_hook.clear()
        logits = self.backbone(x)
        divergences = compute_layerwise_flow_divergence(
            self.activation_hook.activations,
            self.layer_names,
        )
        return logits, divergences

    def forward(self, x: torch.Tensor):
        """
        Ordinary forward (e.g., for inference).
        """

        return self.backbone(x)
