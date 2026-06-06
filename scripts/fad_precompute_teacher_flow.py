"""
Pre-calculation of D_T of teacher on validation.
"""

import logging

from flow_compress.distillation.flows.divergence import (
    compute_flow_statistics,
    compute_layerwise_flow_divergence,
)
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.trainer import fad_distillation
import torch
from torch.utils.data import DataLoader


def precompute_teacher_flows(teacher, val_loader: DataLoader, device: str = "cuda"):
    """
    Script for pre-calculation of D_T of teacher on validation.
    """

    teacher = teacher.to(device)
    teacher.eval()

    all_divs = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            teacher.activation_hook.clear()
            _ = teacher.backbone(x)
            divs = compute_layerwise_flow_divergence(
                teacher.activation_hook.activations,
                teacher.layer_names,
                fad_distillation,
            )
            all_divs.append(divs)

    stats = compute_flow_statistics(all_divs)

    logging.info(f"Teacher flow statistics precomputed: {stats}")

    return stats
