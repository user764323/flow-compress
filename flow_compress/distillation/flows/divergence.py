"""
Divergence calculations.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-6  # stabilizer from the article (analog of 1e-6).


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function: brings activations to the size (batch, features).
    """

    if x.dim() > 2:
        return x.view(x.size(0), -1)
    return x


def flow_divergence_pair(
    T1: torch.Tensor,
    T2: torch.Tensor,
    w: float = 1.0,
) -> torch.Tensor:
    """
    Implementation of discrete approximation of flow divergence between adjacent layers.
    """

    T1_flat = _flatten(T1)
    T2_flat = _flatten(T2)

    # difference
    diff = T2_flat - T1_flat
    num = torch.norm(diff, dim=1)  # L2 by feature dimension
    denom = torch.norm(T2_flat, dim=1) + EPS

    D = num / denom

    # Return average over batch.
    return D.mean()


def compute_layerwise_flow_divergence(
    activations: Dict[str, torch.Tensor],
    ordered_layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Calculates D_T for each pair of adjacent layers by their activations.
    """

    divergences: Dict[str, torch.Tensor] = {}

    for i in range(1, len(ordered_layer_names)):
        prev_layer = ordered_layer_names[i - 1]
        curr_layer = ordered_layer_names[i]

        if prev_layer not in activations or curr_layer not in activations:
            continue

        T1 = activations[prev_layer]
        T2 = activations[curr_layer]

        D = flow_divergence_pair(T1, T2)
        divergences[curr_layer] = D.detach()

    return divergences


def variance_normalization(
    divergence_per_sample: torch.Tensor,
    var: torch.Tensor,
    max_var: torch.Tensor,
) -> torch.Tensor:
    """
    Normalization by variance.
    """

    # Normalize by maximum variance, to make scales comparable.
    return divergence_per_sample / (max_var + EPS)


def compute_flow_statistics(
    divergences_list: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collects statistics on D_T on validation: mean and max by layers.
    """

    aggregate: Dict[str, List[torch.Tensor]] = {}
    for divs in divergences_list:
        for layer, value in divs.items():
            aggregate.setdefault(layer, []).append(value)

    mean_stats: Dict[str, torch.Tensor] = {}
    max_stats: Dict[str, torch.Tensor] = {}
    for layer, vals in aggregate.items():
        stacked = torch.stack(vals)  # (num_samples,)
        mean_stats[layer] = stacked.mean()
        max_stats[layer] = stacked.max()

    return {"mean": mean_stats, "max": max_stats}
