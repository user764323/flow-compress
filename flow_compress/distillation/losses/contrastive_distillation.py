"""
Contrastive Distillation loss for knowledge distillation.
"""

from typing import Dict, Optional, Tuple

from flow_compress.distillation.flows.alignment import compute_dalign
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDistillationLoss(nn.Module):
    """
    Contrastive Distillation loss with FAD integration.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_fakd: float = 0.1,
        use_flow_alignment: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_fakd = lambda_fakd
        self.use_flow_alignment = use_flow_alignment

    def contrastive_loss(
        self,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between teacher and student embeddings.
        """

        # Normalize embeddings
        t_emb = F.normalize(teacher_embeddings, p=2, dim=1)
        s_emb = F.normalize(student_embeddings, p=2, dim=1)

        # Compute similarity matrix
        # sim[i, j] = similarity between student[i] and teacher[j]
        sim_matrix = torch.matmul(s_emb, t_emb.t()) / \
            self.temperature  # (B, B)

        # Positive pairs are on the diagonal
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(
        self,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor,
        teacher_div: Optional[Dict[str, torch.Tensor]] = None,
        student_div: Optional[Dict[str, torch.Tensor]] = None,
        layer_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total contrastive distillation loss with FAD alignment.
        """

        # Compute contrastive loss
        l_contrastive = self.contrastive_loss(
            teacher_embeddings, student_embeddings)

        # Compute FAD alignment term if enabled
        l_fad_alignment = torch.tensor(0.0, device=l_contrastive.device)

        if (
            self.use_flow_alignment
            and teacher_div is not None
            and student_div is not None
        ):
            # Compute alignment for each layer
            alignment_losses = []

            # Get common layers
            common_layers = set(teacher_div.keys()) & set(student_div.keys())

            for layer in common_layers:
                # Extract single-layer divergences
                t_div_layer = {layer: teacher_div[layer]}
                s_div_layer = {layer: student_div[layer]}

                # Compute D_align for this layer
                dalign = compute_dalign(t_div_layer, s_div_layer)

                # Apply layer weight if provided
                weight = layer_weights.get(
                    layer, 1.0) if layer_weights else 1.0

                alignment_losses.append(weight * dalign)

            if alignment_losses:
                # Sum over layers: Σ_l w_l · D_align(D_T,l, D_S,l)
                l_fad_alignment = torch.stack(alignment_losses).sum()

        # Total loss: L_contrastive + λ_FAKD · Σ_l w_l · D_align(D_T,l, D_S,l)
        l_total = l_contrastive + self.lambda_fakd * l_fad_alignment

        info = {
            "L_contrastive": l_contrastive.detach(),
            "L_fad_alignment": l_fad_alignment.detach(),
            "L_total": l_total.detach(),
        }

        return l_total, info
