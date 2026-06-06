"""
Attention Transfer (AT) loss for knowledge distillation.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_map(features: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Compute attention map from feature tensor.
    """

    if len(features.shape) == 4:
        # Convolutional features: (B, C, H, W)
        # Sum over channels and normalize
        attention = torch.sum(torch.abs(features) ** p, dim=1)  # (B, H, W)
        attention = attention / \
            (attention.sum(dim=(1, 2), keepdim=True) + 1e-8)

    elif len(features.shape) == 3:
        # Linear/Transformer features: (B, L, D)
        # Sum over feature dimension and normalize
        attention = torch.sum(torch.abs(features) ** p, dim=2)  # (B, L)
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)

    elif len(features.shape) == 2:
        # 2D features: (B, D)
        attention = torch.abs(features) ** p
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)

    else:
        raise ValueError(f"Unsupported feature shape: {features.shape}")

    return attention


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer (AT) loss as described in "Paying More Attention to Attention".
    """

    def __init__(self, p: int = 2, reduction: str = "mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """
        Compute Attention Transfer loss.
        """

        losses = []

        # Determine layer pairs
        if layer_mapping is None:
            # Use matching layer names
            common_layers = set(teacher_features.keys()) & set(
                student_features.keys())
            layer_pairs = [(layer, layer) for layer in common_layers]
        else:
            # Use provided mapping
            layer_pairs = [
                (t_layer, s_layer)
                for t_layer, s_layer in layer_mapping.items()
                if t_layer in teacher_features and s_layer in student_features
            ]

        if not layer_pairs:
            # Return zero loss if no matching layers
            return torch.tensor(
                0.0, device=next(iter(teacher_features.values())).device
            )

        for t_layer, s_layer in layer_pairs:
            t_feat = teacher_features[t_layer]
            s_feat = student_features[s_layer]

            # Compute attention maps
            t_attn = attention_map(t_feat, p=self.p)
            s_attn = attention_map(s_feat, p=self.p)

            # Handle shape mismatch by interpolation/resizing
            if t_attn.shape != s_attn.shape:
                # Resize student attention to match teacher
                if len(t_attn.shape) == 3:  # (B, H, W)
                    s_attn = F.interpolate(
                        s_attn.unsqueeze(1),
                        size=(t_attn.shape[1], t_attn.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                elif len(t_attn.shape) == 2:  # (B, L)
                    # For sequence length, use linear interpolation
                    if s_attn.shape[1] != t_attn.shape[1]:
                        s_attn = F.interpolate(
                            s_attn.unsqueeze(1),
                            size=t_attn.shape[1],
                            mode="linear",
                            align_corners=False,
                        ).squeeze(1)

            # Compute L2 loss
            loss = F.mse_loss(s_attn, t_attn.detach(),
                              reduction=self.reduction)
            losses.append(loss)

        if self.reduction == "mean":
            return torch.stack(losses).mean()
        elif self.reduction == "sum":
            return torch.stack(losses).sum()
        else:
            return torch.stack(losses)
