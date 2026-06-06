"""
Functions for finding the optimal candidate for distillation.
"""

from typing import Callable, Dict, List, Optional, Tuple

from flow_compress.distillation.flows.alignment import compute_dalign
from flow_compress.distillation.flows.divergence import compute_layerwise_flow_divergence
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def find_optimal_student(
    teacher: TeacherWrapper,
    student_candidates: List[Tuple[StudentWrapper, Dict]],
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 10,
    weight_alignment: float = 0.4,
    weight_accuracy: float = 0.3,
    weight_efficiency: float = 0.3,
) -> Tuple[StudentWrapper, Dict]:
    """
    Finds the optimal student for distillation.
    """

    teacher.eval()

    best_score = -float("inf")
    best_student = None
    best_metrics = None

    for student, metadata in student_candidates:
        student.eval()

        total_alignment = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                x = x.to(device)
                y = y.to(device)

                # Forward passes
                t_logits, t_div = teacher.forward_with_flows(x)
                s_logits, s_div = student.forward_with_flows(x)

                # Alignment score
                dalign = compute_dalign(t_div, s_div)
                alignment = 1.0 / (1.0 + dalign)
                total_alignment += alignment.item()

                # Accuracy
                preds = s_logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        avg_alignment = total_alignment / num_batches
        accuracy = total_correct / max(1, total_samples)

        # Efficiency (inversely proportional to the number of parameters)
        num_params = sum(p.numel() for p in student.parameters())
        efficiency = 1.0 / (1.0 + num_params / 1e6)

        # Composite score
        score = (
            weight_alignment * avg_alignment
            + weight_accuracy * accuracy
            + weight_efficiency * efficiency
        )

        metrics = {
            "alignment": avg_alignment,
            "accuracy": accuracy,
            "efficiency": efficiency,
            "num_params": num_params,
            "composite_score": score,
            "metadata": metadata,
        }

        if score > best_score:
            best_score = score
            best_student = student
            best_metrics = metrics

    return best_student, best_metrics


def search_best_distillation_candidate(
    teacher: TeacherWrapper,
    student_candidates: List[Tuple[StudentWrapper, Dict]],
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 10,
    use_sensitivity: bool = True,
) -> Tuple[StudentWrapper, Dict]:
    """
    Finds the best candidate for distillation with layer sensitivity.
    """

    teacher.eval()

    best_score = -float("inf")
    best_student = None
    best_metrics = None

    teacher_sensitivities = {}
    if use_sensitivity:
        teacher_sensitivities = _compute_layer_sensitivity(
            teacher, val_loader, device, num_batches
        )

    for student, metadata in student_candidates:
        student.eval()

        total_alignment = 0.0
        total_sensitivity_match = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                x = x.to(device)
                y = y.to(device)

                # Forward passes
                t_logits, t_div = teacher.forward_with_flows(x)
                s_logits, s_div = student.forward_with_flows(x)

                # Alignment
                dalign = compute_dalign(t_div, s_div)
                alignment = 1.0 / (1.0 + dalign)
                total_alignment += alignment.item()

                # Sensitivity matching
                if use_sensitivity:
                    t_sens = _compute_sensitivity_from_div(t_div)
                    s_sens = _compute_sensitivity_from_div(s_div)
                    match = _compute_sensitivity_correlation(t_sens, s_sens)
                    total_sensitivity_match += match

                # Accuracy
                preds = s_logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        avg_alignment = total_alignment / num_batches
        avg_sensitivity_match = (
            total_sensitivity_match / num_batches if use_sensitivity else 0.0
        )
        accuracy = total_correct / max(1, total_samples)

        # Composite score
        if use_sensitivity:
            score = 0.4 * avg_alignment + 0.3 * avg_sensitivity_match + 0.3 * accuracy
        else:
            score = 0.5 * avg_alignment + 0.5 * accuracy

        metrics = {
            "alignment": avg_alignment,
            "sensitivity_match": avg_sensitivity_match,
            "accuracy": accuracy,
            "composite_score": score,
            "metadata": metadata,
        }

        if score > best_score:
            best_score = score
            best_student = student
            best_metrics = metrics

    return best_student, best_metrics


def _compute_layer_sensitivity(
    model: TeacherWrapper,
    data_loader: DataLoader,
    device: str,
    num_batches: int,
) -> Dict[str, float]:
    """Calculates the sensitivity of layers of the model."""

    model.eval()
    sensitivities = {}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            x = x.to(device)
            _, div = model.forward_with_flows(x)

            for layer, div_value in div.items():
                if layer not in sensitivities:
                    sensitivities[layer] = []
                sensitivities[layer].append(
                    div_value.item()
                    if isinstance(div_value, torch.Tensor)
                    else div_value
                )

    # Average sensitivities
    avg_sensitivities = {
        layer: np.mean(values) for layer, values in sensitivities.items()
    }
    return avg_sensitivities


def _compute_sensitivity_from_div(div: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculates sensitivity from divergences."""

    values = [v.item() if isinstance(v, torch.Tensor)
              else v for v in div.values()]
    if not values:
        return {}

    max_val = max(values) if values else 1.0
    normalized = {
        k: (v.item() if isinstance(v, torch.Tensor) else v) / (max_val + 1e-8)
        for k, v in div.items()
    }
    return normalized


def _compute_sensitivity_correlation(
    t_sens: Dict[str, float],
    s_sens: Dict[str, float],
) -> float:
    """Calculates correlation of sensitivities between teacher and student."""

    common_layers = set(t_sens.keys()) & set(s_sens.keys())
    if not common_layers:
        return 0.0

    t_vals = np.array([t_sens[l] for l in common_layers])
    s_vals = np.array([s_sens[l] for l in common_layers])

    # Normalization
    t_vals = (t_vals - t_vals.mean()) / (t_vals.std() + 1e-8)
    s_vals = (s_vals - s_vals.mean()) / (s_vals.std() + 1e-8)

    # Correlation
    correlation = np.mean(t_vals * s_vals)
    return float(correlation)


def select_optimal_candidate_for_distillation(
    teacher: TeacherWrapper,
    student_candidates: List[Tuple[StudentWrapper, Dict]],
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 10,
    criteria_weights: Optional[Dict[str, float]] = None,
    return_all_scores: bool = False,
) -> Tuple[StudentWrapper, Dict]:
    """
    Selects the optimal candidate for distillation by multiple criteria.
    """

    if criteria_weights is None:
        criteria_weights = {
            "flow_alignment": 0.3,
            "capacity_match": 0.2,
            "performance": 0.3,
            "efficiency": 0.15,
            "stability": 0.05,
        }

    teacher.eval()

    all_candidates_scores = []
    best_score = -float("inf")
    best_student = None
    best_metrics = None

    for student, metadata in student_candidates:
        student.eval()

        metrics = _evaluate_candidate_comprehensive(
            teacher, student, val_loader, device, num_batches
        )

        # Calculate composite score
        composite_score = sum(
            criteria_weights.get(criterion, 0.0) * metrics.get(criterion, 0.0)
            for criterion in criteria_weights.keys()
        )

        metrics["composite_score"] = composite_score
        metrics["metadata"] = metadata

        all_candidates_scores.append((student, metrics, composite_score))

        if composite_score > best_score:
            best_score = composite_score
            best_student = student
            best_metrics = metrics

    result_metrics = best_metrics.copy()

    if return_all_scores:
        result_metrics["all_candidates"] = [
            {"student": s, "metrics": m, "score": score}
            for s, m, score in sorted(
                all_candidates_scores, key=lambda x: x[2], reverse=True
            )
        ]

    return best_student, result_metrics


def _evaluate_candidate_comprehensive(
    teacher: TeacherWrapper,
    student: StudentWrapper,
    val_loader: DataLoader,
    device: str,
    num_batches: int,
) -> Dict[str, float]:
    """Comprehensive evaluation of a candidate by all criteria."""

    total_alignment = 0.0
    total_capacity_match = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            x = x.to(device)
            y = y.to(device)

            # Forward passes
            t_logits, t_div = teacher.forward_with_flows(x)
            s_logits, s_div = student.forward_with_flows(x)

            # Flow alignment
            dalign = compute_dalign(t_div, s_div)
            alignment = 1.0 / (1.0 + dalign)
            total_alignment += alignment.item()

            # Capacity match (comparison of average divergences)
            t_avg_div = np.mean(
                [v.item() if isinstance(v, torch.Tensor)
                 else v for v in t_div.values()]
            )
            s_avg_div = np.mean(
                [v.item() if isinstance(v, torch.Tensor)
                 else v for v in s_div.values()]
            )
            capacity_match = 1.0 / (1.0 + abs(t_avg_div - s_avg_div))
            total_capacity_match += capacity_match

            # Performance
            preds = s_logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            # For stability
            all_predictions.append(preds.cpu().numpy())
            all_logits.append(torch.softmax(s_logits, dim=1).cpu().numpy())

    # Normalization of metrics
    flow_alignment = total_alignment / num_batches
    capacity_match = total_capacity_match / num_batches
    performance = total_correct / max(1, total_samples)

    # Efficiency (inversely proportional to parameters)
    num_params = sum(p.numel() for p in student.parameters())
    efficiency = 1.0 / (1.0 + num_params / 1e6)

    # Stability (variation of predictions between batches)
    if all_predictions:
        all_preds = np.concatenate(all_predictions)
        # Simple stability metric: fraction of most frequent prediction
        unique, counts = np.unique(all_preds, return_counts=True)
        stability = counts.max() / len(all_preds) if len(all_preds) > 0 else 0.0
    else:
        stability = 0.0

    return {
        "flow_alignment": flow_alignment,
        "capacity_match": capacity_match,
        "performance": performance,
        "efficiency": efficiency,
        "stability": stability,
        "num_params": num_params,
    }


def rank_distillation_candidates(
    teacher: TeacherWrapper,
    student_candidates: List[Tuple[StudentWrapper, Dict]],
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 5,
    top_k: int = 1,
) -> List[Tuple[StudentWrapper, Dict]]:
    """
    Ranks candidates for distillation and returns top-K.
    """

    teacher.eval()

    candidate_scores = []

    for student, metadata in student_candidates:
        student.eval()

        total_alignment = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                x = x.to(device)
                y = y.to(device)

                t_logits, t_div = teacher.forward_with_flows(x)
                s_logits, s_div = student.forward_with_flows(x)

                # Simple evaluation
                dalign = compute_dalign(t_div, s_div)
                alignment = 1.0 / (1.0 + dalign)
                total_alignment += alignment.item()

                preds = s_logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        avg_alignment = total_alignment / num_batches
        accuracy = total_correct / max(1, total_samples)

        # Simple score: average alignment and accuracy
        score = (avg_alignment + accuracy) / 2.0

        metrics = {
            "alignment": avg_alignment,
            "accuracy": accuracy,
            "score": score,
            "metadata": metadata,
        }

        candidate_scores.append((student, metrics, score))

    # Sort by score
    candidate_scores.sort(key=lambda x: x[2], reverse=True)

    # Return top-K
    return [(student, metrics) for student, metrics, _ in candidate_scores[:top_k]]


def find_optimal_student_with_constraints(
    teacher: TeacherWrapper,
    student_candidates: List[Tuple[StudentWrapper, Dict]],
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 10,
    max_params: Optional[int] = None,
    min_accuracy: Optional[float] = None,
    min_alignment: Optional[float] = None,
) -> Tuple[Optional[StudentWrapper], Dict]:
    """
    Finds the optimal student with constraints.
    """

    teacher.eval()

    # Filtering by constraints
    filtered_candidates = []

    for student, metadata in student_candidates:
        # Check parameters
        num_params = sum(p.numel() for p in student.parameters())
        if max_params is not None and num_params > max_params:
            continue

        filtered_candidates.append((student, metadata))

    if not filtered_candidates:
        return None, {"error": "No candidates satisfy constraints"}

    # Evaluation of filtered candidates
    best_score = -float("inf")
    best_student = None
    best_metrics = None

    for student, metadata in filtered_candidates:
        student.eval()

        total_alignment = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                x = x.to(device)
                y = y.to(device)

                t_logits, t_div = teacher.forward_with_flows(x)
                s_logits, s_div = student.forward_with_flows(x)

                dalign = compute_dalign(t_div, s_div)
                alignment = 1.0 / (1.0 + dalign)
                total_alignment += alignment.item()

                preds = s_logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        avg_alignment = total_alignment / num_batches
        accuracy = total_correct / max(1, total_samples)

        # Check minimum requirements
        if min_accuracy is not None and accuracy < min_accuracy:
            continue
        if min_alignment is not None and avg_alignment < min_alignment:
            continue

        # Score
        score = 0.5 * avg_alignment + 0.5 * accuracy

        metrics = {
            "alignment": avg_alignment,
            "accuracy": accuracy,
            "num_params": num_params,
            "composite_score": score,
            "metadata": metadata,
        }

        if score > best_score:
            best_score = score
            best_student = student
            best_metrics = metrics

    if best_student is None:
        return None, {"error": "No candidates satisfy all constraints"}

    return best_student, best_metrics
