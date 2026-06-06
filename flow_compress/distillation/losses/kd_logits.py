"""
Logits-based KD loss for flow-based distillation.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


class LogitsKDLoss(torch.nn.Module):
    """
    Classical logits-based KD (Hinton et al.), used in the composition L_total.
    """

    def __init__(self, temperature: float = 4.0):
        """
        temperature: T in softening softmax.
        """

        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates KL(student || teacher) at elevated temperature.
        """

        T = self.temperature

        # log_softmax for student
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        # softmax for teacher
        p_t = F.softmax(teacher_logits / T, dim=1).detach()

        # KL divergence by batch
        loss_kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T**2)
        return loss_kl
