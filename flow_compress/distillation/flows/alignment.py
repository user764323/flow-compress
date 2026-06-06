"""
Alignment calculations.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def symmetrized_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Symmetrized KL-divergence between distributions p and q.
    """

    p = p + 1e-8
    q = q + 1e-8
    log_p = torch.log(p)
    log_q = torch.log(q)

    kl_pq = (p * (log_p - log_q)).sum()
    kl_qp = (q * (log_q - log_p)).sum()
    return 0.5 * (kl_pq + kl_qp)


def normalize_over_depth(divergences: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Converts dictionary {layer_name: D_T} to a vector of probabilities by depth.
    """

    # fix order of layers
    layers = sorted(divergences.keys())
    values = torch.stack([divergences[l] for l in layers])  # (L,)
    # Softmax by depth → distribution
    probs = torch.softmax(values, dim=0)
    return probs


def compute_dalign(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Main method Dalign(D^T, D^S): alignment of divergences by depth.
    """

    p = normalize_over_depth(teacher_div)
    q = normalize_over_depth(student_div)
    return symmetrized_kl(p, q)


def compute_layer_weights(
    teacher_div: Dict[str, torch.Tensor],
    lambda_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Computes depth-dependent weights w_l.
    """

    # Get ordered layers
    layers = sorted(teacher_div.keys())

    if len(layers) < 2:
        # If only one layer, return uniform weight
        return {
            layers[0]: torch.tensor(1.0, device=next(
                iter(teacher_div.values())).device)
        }

    # Compute sum of all divergences: Σ_{k=1}^{L-1} D_T,k
    total_div = sum(teacher_div[layer] for layer in layers)
    eps = 1e-8

    weights = {}
    for i, layer in enumerate(layers):
        first_term = teacher_div[layer] / (total_div + eps)

        if i < len(layers) - 1:
            next_layer = layers[i + 1]
            diff = teacher_div[next_layer] - teacher_div[layer]
            second_term = lambda_weight * diff
        else:
            prev_layer = layers[i - 1]
            diff = teacher_div[layer] - teacher_div[prev_layer]
            second_term = lambda_weight * diff

        weights[layer] = first_term + second_term

    min_weight = min(w.item() for w in weights.values())
    if min_weight < 0:
        shift = abs(min_weight) + eps
        weights = {layer: w + shift for layer, w in weights.items()}

    total_weight = sum(weights.values())
    if total_weight > eps:
        weights = {layer: w / total_weight for layer, w in weights.items()}

    return weights


def compute_fad_loss_with_weights(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
    layer_weights: Dict[str, torch.Tensor],
    selective_mask: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Computes L_FAD.
    """

    device = next(iter(teacher_div.values())).device
    l_fad = torch.tensor(0.0, device=device)

    teacher_layers = sorted(teacher_div.keys())
    student_layers = sorted(student_div.keys())

    # For each layer, compute weighted alignment
    for i, t_layer in enumerate(teacher_layers):
        if t_layer not in layer_weights:
            continue

        # Apply selective mask if provided
        if selective_mask is not None:
            if t_layer not in selective_mask or selective_mask[t_layer] == 0.0:
                continue

        # Find corresponding student layer (by index if same structure, or by name)
        if i < len(student_layers):
            s_layer = student_layers[i]
        else:
            # Use last student layer if teacher has more layers
            s_layer = student_layers[-1]

        if s_layer not in student_div:
            continue

        # Compute D_align for this layer pair
        # For scalar divergences, we use absolute difference as alignment metric
        t_div = teacher_div[t_layer]
        s_div = student_div[s_layer]

        # Symmetrized KL-like metric for scalars
        eps = 1e-8
        t_div_norm = t_div + eps
        s_div_norm = s_div + eps

        # Alignment metric: symmetrized divergence
        dalign = 0.5 * (
            t_div_norm * torch.log(t_div_norm / s_div_norm)
            + s_div_norm * torch.log(s_div_norm / t_div_norm)
        )

        # Weighted contribution
        w_l = layer_weights[t_layer]

        # Apply selective mask weight if provided
        if selective_mask is not None and t_layer in selective_mask:
            w_l = w_l * selective_mask[t_layer]

        l_fad = l_fad + w_l * dalign

    return l_fad


def compute_dalign_at_depth(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
    normalized_depth: float,
) -> torch.Tensor:
    """
    Computes D_align at a specific normalized depth s ∈ [0, 1].
    """

    # Get ordered layers
    teacher_layers = sorted(teacher_div.keys())
    student_layers = sorted(student_div.keys())

    num_teacher_layers = len(teacher_layers)
    num_student_layers = len(student_layers)

    # Map normalized depth to layer indices
    teacher_idx = normalized_depth * (num_teacher_layers - 1)
    student_idx = normalized_depth * (num_student_layers - 1)

    # Get floor and ceil indices for interpolation
    teacher_floor = int(teacher_idx)
    teacher_ceil = min(teacher_floor + 1, num_teacher_layers - 1)
    teacher_alpha = teacher_idx - teacher_floor

    student_floor = int(student_idx)
    student_ceil = min(student_floor + 1, num_student_layers - 1)
    student_alpha = student_idx - student_floor

    # Interpolate divergences
    teacher_val = (1 - teacher_alpha) * teacher_div[
        teacher_layers[teacher_floor]
    ] + teacher_alpha * teacher_div[teacher_layers[teacher_ceil]]

    student_val = (1 - student_alpha) * student_div[
        student_layers[student_floor]
    ] + student_alpha * student_div[student_layers[student_ceil]]

    # Compute alignment metric (symmetrized KL for scalar values)
    # For scalar divergences, we use a simple divergence measure
    eps = 1e-8
    teacher_val = teacher_val + eps
    student_val = student_val + eps

    # Symmetrized KL for scalars
    kl_ts = teacher_val * torch.log(teacher_val / student_val)
    kl_st = student_val * torch.log(student_val / teacher_val)

    return 0.5 * (kl_ts + kl_st)


def compute_fad_loss_integrated(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
    num_integration_points: int = 100,
    use_depth_weighting: bool = True,
) -> torch.Tensor:
    """
    Implements L_FAD
    """

    device = next(iter(teacher_div.values())).device

    # Create integration points
    s_values = torch.linspace(0.0, 1.0, num_integration_points, device=device)

    # Compute depth-dependent weights w(s) if needed
    if use_depth_weighting:
        # Use the depth_weighting function to get weights
        # We'll create a continuous version based on teacher divergence
        weights = depth_weighting(teacher_div, alpha=0.5)
        teacher_layers = sorted(teacher_div.keys())
        num_layers = len(teacher_layers)

        # Map normalized depth to weights
        def get_weight_at_depth(s: float) -> torch.Tensor:
            idx = s * (num_layers - 1)
            floor_idx = int(idx)
            ceil_idx = min(floor_idx + 1, num_layers - 1)
            alpha = idx - floor_idx

            if floor_idx >= num_layers:
                return weights[teacher_layers[-1]]
            if floor_idx < 0:
                return weights[teacher_layers[0]]

            w_floor = weights[teacher_layers[floor_idx]]
            w_ceil = weights[teacher_layers[ceil_idx]]
            return (1 - alpha) * w_floor + alpha * w_ceil

    else:
        # Uniform weighting
        def get_weight_at_depth(s: float) -> torch.Tensor:
            return torch.tensor(1.0, device=device)

    # Integrate using trapezoidal rule
    integral_value = torch.tensor(0.0, device=device)

    for i in range(len(s_values) - 1):
        s1 = s_values[i].item()
        s2 = s_values[i + 1].item()

        # Compute D_align at both points
        dalign1 = compute_dalign_at_depth(teacher_div, student_div, s1)
        dalign2 = compute_dalign_at_depth(teacher_div, student_div, s2)

        # Get weights
        w1 = get_weight_at_depth(s1)
        w2 = get_weight_at_depth(s2)

        # Trapezoidal rule: ∫ f(s) ds ≈ (s2 - s1) * (f(s1) + f(s2)) / 2
        ds = s2 - s1
        integral_value += ds * (w1 * dalign1 + w2 * dalign2) / 2.0

    return integral_value


def depth_weighting(
    teacher_div: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Weight function ω(s).
    """

    layers = sorted(teacher_div.keys())
    values = torch.stack([teacher_div[l] for l in layers])

    # Normalize by maximum, to get [0,1].
    norm_vals = values / (values.max() + 1e-8)

    weights = (1 - alpha) * norm_vals + alpha
    weights = weights / weights.sum()  # to make the sum of weights = 1

    return {layer: w for layer, w in zip(layers, weights)}
