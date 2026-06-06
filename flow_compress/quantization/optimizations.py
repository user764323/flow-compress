from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from flow_compress.quantization.divergence import LayerProfile
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_attention_optimizations(
    profiles: Dict[str, LayerProfile],
    attention_layer_names: Optional[List[str]] = None,
) -> None:
    """Attention-Specific Optimizations"""

    if attention_layer_names is None:
        attention_layer_names = [
            name
            for name, prof in profiles.items()
            if "attention" in name.lower()
            or "attn" in name.lower()
            or prof.module_type == "MultiheadAttention"
        ]

    for name in attention_layer_names:
        if name not in profiles:
            continue

        prof = profiles[name]

        if prof.sensitivity > 0:
            prof.sensitivity = prof.sensitivity * 1.3

        if prof.bits < 4:
            prof.bits = 4


def apply_bit_allocation_optimizations(
    profiles: Dict[str, LayerProfile],
    b_target: float,
    bmin: int = 2,
    bmax: int = 8,
) -> None:

    names = list(profiles.keys())
    if not names:
        return

    for name, prof in profiles.items():
        if prof.grad_l2 > 1.0 and prof.bits < 4:
            prof.bits = max(prof.bits, 4)

    sorted_names = sorted(names)  # Simple ordering by name
    for i in range(1, len(sorted_names)):
        prev_name = sorted_names[i - 1]
        curr_name = sorted_names[i]

        prev_bits = profiles[prev_name].bits
        curr_bits = profiles[curr_name].bits

        if abs(curr_bits - prev_bits) > 2:
            if curr_bits > prev_bits:
                profiles[curr_name].bits = min(prev_bits + 2, bmax)
            else:
                profiles[curr_name].bits = max(prev_bits - 2, bmin)

    total_params = sum(prof.num_params for prof in profiles.values())
    if total_params > 0:
        param_weighted_avg = (
            sum(prof.bits * prof.num_params for prof in profiles.values())
            / total_params
        )

        if param_weighted_avg < b_target * 0.95:
            sorted_by_params = sorted(
                names, key=lambda n: profiles[n].num_params, reverse=True
            )

            num_boost = max(1, len(sorted_by_params) // 5)
            for name in sorted_by_params[:num_boost]:
                if profiles[name].bits < bmax:
                    profiles[name].bits = min(profiles[name].bits + 1, bmax)


def apply_optimizations_for_large_models(
    profiles: Dict[str, LayerProfile],
    use_fast_approximation: bool = True,
    cache_divergence: bool = True,
) -> Dict[str, float]:
    """Implementation Details Optimizations"""

    metrics = {}

    if use_fast_approximation:
        for name, prof in profiles.items():
            if prof.num_params > 10_000_000:  # 10M parameters
                metrics[f"{name}_fast_mode"] = 1.0

    if cache_divergence:
        for name, prof in profiles.items():
            if prof.divergence_count > 0:
                cached_div = prof.divergence
                metrics[f"{name}_cached_div"] = cached_div

    return metrics


def optimize_attention_heads_separately(
    module: nn.MultiheadAttention,
    profiles: Dict[str, LayerProfile],
    layer_name: str,
) -> None:
    """Optimize attention heads separately when possible."""

    if layer_name not in profiles:
        return

    prof = profiles[layer_name]
    num_heads = module.num_heads

    if prof.bits < 4:
        prof.bits = 4

    if num_heads > 8:
        prof.sensitivity = prof.sensitivity * 1.1


def apply_gradient_aware_calibration(
    profiles: Dict[str, LayerProfile],
    gradient_threshold: float = 0.1,
) -> None:
    """Gradient-Aware Calibration Optimization"""

    max_grad = max((prof.grad_l2 for prof in profiles.values()), default=1.0)
    max_grad = max(max_grad, 1e-8)

    for prof in profiles.values():
        grad_activity = prof.grad_l2 / max_grad

        if grad_activity > gradient_threshold:
            prof.sensitivity = prof.sensitivity * (1.0 + grad_activity * 0.5)


def apply_flow_aware_optimizations(
    profiles: Dict[str, LayerProfile],
    flow_threshold: float = 0.5,
) -> None:
    """Flow-Aware Optimizations:

    Layers with high information flow (divergence) should maintain
    higher precision to preserve information propagation.
    """

    # Find max divergence for normalization
    max_div = max((prof.divergence for prof in profiles.values()), default=1.0)
    max_div = max(max_div, 1e-8)

    for prof in profiles.values():
        flow_ratio = prof.divergence / max_div

        if flow_ratio > flow_threshold and prof.bits < 6:
            prof.bits = max(prof.bits, 6)
