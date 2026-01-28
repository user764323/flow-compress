"""
Loss functions for knowledge distillation.
"""

from flow_compress.distillation.losses.attention_transfer import AttentionTransferLoss
from flow_compress.distillation.losses.contrastive_distillation import ContrastiveDistillationLoss
from flow_compress.distillation.losses.fad_loss import FADLoss
from flow_compress.distillation.losses.generative_distillation import (
    DiffusionDistillationLoss,
    GANDistillationLoss,
    GenerativeDistillationLoss,
    VAEDistillationLoss,
)
from flow_compress.distillation.losses.kd_logits import LogitsKDLoss
from flow_compress.distillation.losses.variational_divergence import (
    MetaOptimizedParameters,
    VariationalDivergenceFunctional,
)

__all__ = [
    "FADLoss",
    "LogitsKDLoss",
    "VariationalDivergenceFunctional",
    "MetaOptimizedParameters",
    "AttentionTransferLoss",
    "ContrastiveDistillationLoss",
    "GenerativeDistillationLoss",
    "VAEDistillationLoss",
    "GANDistillationLoss",
    "DiffusionDistillationLoss",
]
