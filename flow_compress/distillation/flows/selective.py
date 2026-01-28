"""
Selective alignment for flow-based distillation.
"""

from typing import Dict, List, Tuple, Optional

import torch


def select_critical_layers(
    teacher_div: Dict[str, torch.Tensor],
    top_k_ratio: float = 0.3,
) -> List[str]:
    """
    Finds critical layers (with the highest D_T) for selective alignment.

    top_k_ratio: ratio of layers that are considered critical (0.3 → 30%).
    """
    layers = list(teacher_div.keys())
    values = torch.stack([teacher_div[l] for l in layers])

    k = max(1, int(len(layers) * top_k_ratio))
    topk_vals, topk_idx = torch.topk(values, k=k)

    critical_layers = [layers[i] for i in topk_idx.tolist()]
    return critical_layers


def detect_critical_points(
    teacher_div: Dict[str, torch.Tensor],
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Identifies layers where D_T,l > μ_T + σ_T
    """

    if not teacher_div:
        return [], torch.tensor(0.0), torch.tensor(0.0)

    layers = list(teacher_div.keys())
    values = torch.stack([teacher_div[l] for l in layers])

    # Compute mean and standard deviation
    mean_div = values.mean()
    std_div = values.std()

    # Identify critical points: D_T,l > μ_T + σ_T
    threshold = mean_div + std_div
    critical_layers = [
        layer for layer in layers if teacher_div[layer] > threshold]

    return critical_layers, mean_div, std_div


def compute_alignment_error(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Computes alignment error e_l = |D_T,l - D_S,l| / D_T,l for each layer.
    """

    alignment_errors = {}
    eps = 1e-8

    # Get common layers
    common_layers = set(teacher_div.keys()) & set(student_div.keys())

    for layer in common_layers:
        d_t = teacher_div[layer]
        d_s = student_div[layer]

        # Compute alignment error: e_l = |D_T,l - D_S,l| / D_T,l
        numerator = torch.abs(d_t - d_s)
        denominator = d_t + eps  # Add epsilon to avoid division by zero
        e_l = numerator / denominator

        alignment_errors[layer] = e_l

    return alignment_errors


def prioritized_alignment_mask(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
    error_threshold: float = 0.1,
    use_critical_points: bool = True,
) -> Dict[str, float]:
    """
    Creates a mask that focuses on layers where e_l > threshold.
    """

    critical_layers = []
    if use_critical_points:
        critical_layers, _, _ = detect_critical_points(teacher_div)

    alignment_errors = compute_alignment_error(teacher_div, student_div)

    mask = {}

    for layer in alignment_errors.keys():
        e_l = alignment_errors[layer]

        # Check if layer should be aligned
        should_align = e_l > error_threshold

        # If using critical points, also check if layer is critical
        if use_critical_points:
            should_align = should_align and (layer in critical_layers)

        # Set mask weight: 1.0 for layers to align, 0.0 otherwise
        mask[layer] = 1.0 if should_align else 0.0

    # If no layers are selected, fall back to aligning all layers
    if sum(mask.values()) == 0:
        # Fallback: align all layers with non-zero error
        for layer in alignment_errors.keys():
            if alignment_errors[layer] > 0:
                mask[layer] = 1.0

    return mask


def selective_flow_alignment(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
    error_threshold: float = 0.1,
    use_critical_points: bool = True,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Complete selective flow alignment pipeline.
    """

    # Critical point detection
    critical_layers, mean_div, std_div = detect_critical_points(teacher_div)

    # Student difficulty assessment
    alignment_errors = compute_alignment_error(teacher_div, student_div)

    # Prioritized alignment mask
    alignment_mask = prioritized_alignment_mask(
        teacher_div,
        student_div,
        error_threshold=error_threshold,
        use_critical_points=use_critical_points,
    )

    stats = {
        "mean_div": mean_div,
        "std_div": std_div,
        "critical_layers_count": len(critical_layers),
        "selected_layers_count": sum(1 for w in alignment_mask.values() if w > 0),
        "total_layers_count": len(alignment_errors),
    }

    return alignment_mask, alignment_errors, stats


def curriculum_mask(
    layers: List[str],
    stage_idx: int,
    num_stages: int,
) -> Dict[str, float]:
    """
    Implements "curriculum over depth".
    """

    L = len(layers)
    segment_size = max(1, L // num_stages)
    start = stage_idx * segment_size
    end = L if stage_idx == num_stages - 1 else (stage_idx + 1) * segment_size

    mask = {}
    for i, layer in enumerate(layers):
        mask[layer] = 1.0 if start <= i < end else 0.0
    return mask
