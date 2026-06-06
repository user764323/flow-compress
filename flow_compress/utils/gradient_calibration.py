"""
Gradient-aware calibration for distillation.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradient_magnitude(
    model: nn.Module,
    loss: torch.Tensor,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes gradient magnitudes for specified layers.
    """

    # Backward pass
    loss.backward(retain_graph=True)

    gradient_mags = {}

    if layer_names is None:
        # Compute for all named parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_mags[name] = param.grad.norm().detach()
    else:
        # Compute only for specified layers
        for name, module in model.named_modules():
            if name in layer_names:
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if param_name else name
                    if param.grad is not None:
                        gradient_mags[full_name] = param.grad.norm().detach()

    return gradient_mags


def compute_gradient_sensitivity(
    gradient_mags: Dict[str, torch.Tensor],
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Computes gradient-based sensitivity for each layer.
    """
    sensitivities = {}

    available_mags = {
        layer: gradient_mags.get(layer, torch.tensor(0.0)) for layer in layer_names
    }

    if not available_mags:
        return {layer: torch.tensor(0.0) for layer in layer_names}

    # Normalize by maximum
    max_mag = max(mag.item() for mag in available_mags.values())
    if max_mag > 0:
        for layer in layer_names:
            mag = available_mags[layer].item()
            sensitivities[layer] = torch.tensor(mag / (max_mag + 1e-8))
    else:
        sensitivities = {layer: torch.tensor(0.0) for layer in layer_names}

    return sensitivities


def calibrate_distillation_weights(
    teacher_grad_mags: Dict[str, torch.Tensor],
    student_grad_mags: Dict[str, torch.Tensor],
    base_weights: Optional[Dict[str, float]] = None,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Calibrates distillation weights based on gradient information.
    """

    # Get common layers
    common_layers = set(teacher_grad_mags.keys()) & set(
        student_grad_mags.keys())

    if not common_layers:
        return base_weights if base_weights else {}

    # Compute gradient-based importance
    grad_importance = {}
    for layer in common_layers:
        # Combine teacher and student gradient magnitudes
        combined = (
            teacher_grad_mags[layer].item() + student_grad_mags[layer].item()
        ) / 2.0
        grad_importance[layer] = combined

    # Normalize importance
    max_imp = max(grad_importance.values()
                  ) if grad_importance.values() else 1.0
    if max_imp > 0:
        grad_importance = {
            layer: imp / (max_imp + 1e-8) for layer, imp in grad_importance.items()
        }

    # Combine with base weights
    if base_weights is None:
        base_weights = {layer: 1.0 / len(common_layers)
                        for layer in common_layers}

    calibrated = {}
    for layer in common_layers:
        base = base_weights.get(layer, 0.0)
        grad = grad_importance.get(layer, 0.0)
        calibrated[layer] = (1.0 - alpha) * base + alpha * grad

    # Renormalize
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {layer: w / total for layer, w in calibrated.items()}

    return calibrated


def adaptive_temperature_calibration(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    base_temperature: float = 4.0,
    gradient_mags: Optional[Dict[str, torch.Tensor]] = None,
) -> float:
    """
    Adaptively calibrates temperature based on gradient information.
    """

    # Compute prediction alignment
    t_probs = F.softmax(teacher_logits, dim=1)
    s_probs = F.softmax(student_logits, dim=1)

    # KL divergence as misalignment measure
    kl_div = F.kl_div(
        F.log_softmax(student_logits, dim=1), t_probs.detach(), reduction="batchmean"
    )

    # Higher KL = more misalignment = need higher temperature
    alignment_factor = 1.0 / (1.0 + kl_div.item())

    # Adjust temperature: better alignment -> lower temperature
    calibrated_temp = base_temperature * (0.5 + 0.5 * alignment_factor)

    # If gradient information available, further adjust
    if gradient_mags:
        avg_grad_mag = torch.stack(list(gradient_mags.values())).mean().item()
        # Higher gradients -> more learning needed -> higher temperature
        grad_factor = 1.0 + 0.2 * min(avg_grad_mag, 1.0)
        calibrated_temp *= grad_factor

    return max(1.0, min(calibrated_temp, 10.0))  # Clamp to reasonable range
