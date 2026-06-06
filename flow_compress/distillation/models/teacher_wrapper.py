"""
Teacher wrapper for flow-based distillation.
"""

from typing import Dict, List, Tuple

from flow_compress.distillation.flows.divergence import compute_layerwise_flow_divergence
from flow_compress.distillation.flows.hooks import ActivationHook
import torch
import torch.nn as nn


class TeacherWrapper(nn.Module):
    """
    Wrapper for teacher model
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

    @torch.no_grad()
    def forward_with_flows(self, x: torch.Tensor):
        """
        Makes forward pass and returns logits + flow divergences.
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
        Ordinary forward (without explicit calculation of D_T). Can be used for inference.
        """

        return self.backbone(x)
