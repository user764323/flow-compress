"""
Task-driven representational fidelity metrics.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_representational_similarity(
    teacher_features: Dict[str, torch.Tensor],
    student_features: Dict[str, torch.Tensor],
    layer_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes representational similarity between teacher and student features.
    """

    similarities = {}

    if layer_mapping is None:
        # Assume same layer names
        common_layers = set(teacher_features.keys()) & set(student_features.keys())
        layer_mapping = {layer: layer for layer in common_layers}

    for t_layer, s_layer in layer_mapping.items():
        if t_layer not in teacher_features or s_layer not in student_features:
            continue

        t_feat = teacher_features[t_layer]
        s_feat = student_features[s_layer]

        # Flatten features
        t_flat = t_feat.view(t_feat.size(0), -1)
        s_flat = s_feat.view(s_feat.size(0), -1)

        # Normalize
        t_norm = F.normalize(t_flat, p=2, dim=1)
        s_norm = F.normalize(s_flat, p=2, dim=1)

        # Cosine similarity (average over batch)
        similarity = (t_norm * s_norm).sum(dim=1).mean()
        similarities[f"{t_layer}->{s_layer}"] = similarity.detach()

    return similarities


def compute_task_driven_fidelity(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_features: Optional[Dict[str, torch.Tensor]] = None,
    student_features: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes task-driven representational fidelity.
    """

    metrics = {}

    # 1. Prediction agreement on correct samples
    t_preds = teacher_logits.argmax(dim=1)
    s_preds = student_logits.argmax(dim=1)
    correct_mask = t_preds == labels

    if correct_mask.any():
        agreement = (t_preds[correct_mask] == s_preds[correct_mask]).float().mean()
        metrics["prediction_agreement"] = agreement
    else:
        metrics["prediction_agreement"] = torch.tensor(0.0)

    # 2. Confidence alignment
    t_probs = F.softmax(teacher_logits, dim=1)
    s_probs = F.softmax(student_logits, dim=1)

    # Confidence on correct predictions
    t_conf = t_probs[torch.arange(len(labels)), labels].mean()
    s_conf = s_probs[torch.arange(len(labels)), labels].mean()

    confidence_alignment = 1.0 - torch.abs(t_conf - s_conf)
    metrics["confidence_alignment"] = confidence_alignment

    # 3. Feature similarity on task-relevant samples (if features provided)
    if teacher_features and student_features:
        # Use samples where teacher is confident and correct
        t_confidences = t_probs.max(dim=1)[0]
        high_conf_mask = (t_confidences > 0.7) & correct_mask

        if high_conf_mask.any() and len(teacher_features) > 0:
            # Compute average feature similarity for high-confidence samples
            feat_similarities = []
            for t_layer, t_feat in teacher_features.items():
                if t_layer in student_features:
                    s_feat = student_features[t_layer]
                    # Flatten and normalize
                    t_flat = t_feat.view(t_feat.size(0), -1)
                    s_flat = s_feat.view(s_feat.size(0), -1)
                    t_norm = F.normalize(t_flat, p=2, dim=1)
                    s_norm = F.normalize(s_flat, p=2, dim=1)
                    sim = (t_norm * s_norm).sum(dim=1)
                    feat_similarities.append(sim[high_conf_mask].mean())

            if feat_similarities:
                metrics["feature_fidelity"] = torch.stack(feat_similarities).mean()
            else:
                metrics["feature_fidelity"] = torch.tensor(0.0)
        else:
            metrics["feature_fidelity"] = torch.tensor(0.0)
    else:
        metrics["feature_fidelity"] = torch.tensor(0.0)

    # Overall fidelity score
    metrics["overall_fidelity"] = (
        0.4 * metrics["prediction_agreement"]
        + 0.3 * metrics["confidence_alignment"]
        + 0.3 * metrics["feature_fidelity"]
    )

    return metrics


def compute_information_preservation(
    teacher_features: Dict[str, torch.Tensor],
    student_features: Dict[str, torch.Tensor],
    layer_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes information preservation between teacher and student.
    """

    preservation_scores = {}

    if layer_mapping is None:
        common_layers = set(teacher_features.keys()) & set(student_features.keys())
        layer_mapping = {layer: layer for layer in common_layers}

    for t_layer, s_layer in layer_mapping.items():
        if t_layer not in teacher_features or s_layer not in student_features:
            continue

        t_feat = teacher_features[t_layer]
        s_feat = student_features[s_layer]

        # Flatten features
        t_flat = t_feat.view(t_feat.size(0), -1)
        s_flat = s_feat.view(s_feat.size(0), -1)

        # Compute correlation (proxy for mutual information)
        t_centered = t_flat - t_flat.mean(dim=0, keepdim=True)
        s_centered = s_flat - s_flat.mean(dim=0, keepdim=True)

        # Cross-correlation
        correlation = (t_centered * s_centered).mean(dim=0)
        t_std = t_centered.std(dim=0) + 1e-8
        s_std = s_centered.std(dim=0) + 1e-8

        # Normalized correlation
        normalized_corr = correlation / (t_std * s_std)
        preservation = normalized_corr.abs().mean()

        preservation_scores[f"{t_layer}->{s_layer}"] = preservation.detach()

    return preservation_scores


def adapt_distillation_targets(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
    fidelity_metrics: Dict[str, torch.Tensor],
    base_alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Adapts distillation targets based on representational fidelity.
    """

    # Get common layers
    common_layers = set(teacher_div.keys()) & set(student_div.keys())

    if not common_layers:
        return {}

    # Compute divergence gap (misalignment)
    divergence_gaps = {}
    for layer in common_layers:
        gap = torch.abs(teacher_div[layer] - student_div[layer])
        divergence_gaps[layer] = gap.item()

    # Normalize gaps
    max_gap = max(divergence_gaps.values()) if divergence_gaps.values() else 1.0
    if max_gap > 0:
        normalized_gaps = {
            layer: gap / (max_gap + 1e-8) for layer, gap in divergence_gaps.items()
        }
    else:
        normalized_gaps = {layer: 0.0 for layer in common_layers}

    # Adapt weights: higher gap + lower fidelity -> higher weight
    fidelity_score = fidelity_metrics.get("overall_fidelity", torch.tensor(0.5)).item()

    adapted_weights = {}
    for layer in common_layers:
        gap = normalized_gaps[layer]
        # Weight increases with gap and decreases with fidelity
        weight = base_alpha * (1.0 + gap) * (1.0 - 0.5 * fidelity_score)
        # Clamp to reasonable range
        adapted_weights[layer] = max(0.1, min(weight, 2.0))

    # Normalize
    total = sum(adapted_weights.values())
    if total > 0:
        adapted_weights = {
            layer: w / total * len(common_layers)
            for layer, w in adapted_weights.items()
        }

    return adapted_weights
