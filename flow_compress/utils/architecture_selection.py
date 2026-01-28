"""
Student architecture selection framework.
"""

import logging
from typing import Dict, List, Optional, Tuple

from flow_compress.distillation.flows.alignment import compute_dalign
from flow_compress.distillation.flows.divergence import compute_layerwise_flow_divergence
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_capacity_alignment_score(
    teacher_div: Dict[str, torch.Tensor],
    student_div: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Computes teacher-student capacity alignment score.
    """

    # Compute alignment using Dalign
    dalign = compute_dalign(teacher_div, student_div)

    # Invert to get alignment score (lower divergence = higher alignment)
    alignment_score = 1.0 / (1.0 + dalign)

    return alignment_score


def compute_information_sensitivity(
    divergences: Dict[str, torch.Tensor],
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Computes layer-wise information sensitivity.
    """

    sensitivities = {}

    # Normalize divergences
    values = torch.stack(
        [divergences.get(layer, torch.tensor(0.0)) for layer in layer_names]
    )
    if values.max() > 0:
        normalized = values / (values.max() + 1e-8)
    else:
        normalized = values

    # Sensitivity is proportional to normalized divergence
    # Add depth weighting (deeper layers may be more sensitive)
    for i, layer in enumerate(layer_names):
        depth_weight = 1.0 + 0.1 * (i / max(1, len(layer_names) - 1))
        sensitivities[layer] = normalized[i] * depth_weight

    return sensitivities


def evaluate_student_candidate(
    teacher: TeacherWrapper,
    student: StudentWrapper,
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 10,
) -> Dict[str, float]:
    """
    Evaluates a student architecture candidate.
    """

    teacher.eval()
    student.eval()

    total_alignment = 0.0
    total_sensitivity_match = 0.0
    total_correct = 0
    total_samples = 0

    teacher_sensitivities_agg = {}
    student_sensitivities_agg = {}

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            x = x.to(device)
            y = y.to(device)

            # Forward passes
            t_logits, t_div = teacher.forward_with_flows(x)
            s_logits, s_div = student.forward_with_flows(x)

            # Capacity alignment
            alignment = compute_capacity_alignment_score(t_div, s_div)
            total_alignment += alignment.item()

            # Information sensitivity
            t_sens = compute_information_sensitivity(t_div, list(t_div.keys()))
            s_sens = compute_information_sensitivity(s_div, list(s_div.keys()))

            # Aggregate sensitivities
            for layer in t_sens:
                if layer not in teacher_sensitivities_agg:
                    teacher_sensitivities_agg[layer] = []
                teacher_sensitivities_agg[layer].append(t_sens[layer].item())

            for layer in s_sens:
                if layer not in student_sensitivities_agg:
                    student_sensitivities_agg[layer] = []
                student_sensitivities_agg[layer].append(s_sens[layer].item())

            # Task performance
            preds = s_logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    # Compute average sensitivities
    avg_t_sens = {
        layer: torch.tensor(values).mean().item()
        for layer, values in teacher_sensitivities_agg.items()
    }
    avg_s_sens = {
        layer: torch.tensor(values).mean().item()
        for layer, values in student_sensitivities_agg.items()
    }

    # Sensitivity matching (correlation-like metric)
    common_layers = set(avg_t_sens.keys()) & set(avg_s_sens.keys())
    if common_layers:
        t_vals = torch.tensor([avg_t_sens[l] for l in common_layers])
        s_vals = torch.tensor([avg_s_sens[l] for l in common_layers])
        # Normalize
        t_vals = (t_vals - t_vals.mean()) / (t_vals.std() + 1e-8)
        s_vals = (s_vals - s_vals.mean()) / (s_vals.std() + 1e-8)
        sensitivity_match = (t_vals * s_vals).mean().item()
    else:
        sensitivity_match = 0.0

    return {
        "capacity_alignment": total_alignment / num_batches,
        "sensitivity_match": sensitivity_match,
        "accuracy": total_correct / max(1, total_samples),
        "teacher_sensitivities": avg_t_sens,
        "student_sensitivities": avg_s_sens,
    }


def select_best_student_architecture(
    teacher: TeacherWrapper,
    student_candidates: List[Tuple[StudentWrapper, Dict]],
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 10,
    weight_alignment: float = 0.4,
    weight_sensitivity: float = 0.3,
    weight_accuracy: float = 0.3,
    use_optimization_criteria: bool = False,
) -> Tuple[StudentWrapper, Dict]:
    """
    Selects the best student architecture candidate.
    """

    # Use optimization-driven criteria if requested
    if use_optimization_criteria:
        try:
            from flow_compress.utils.optimization_criteria import StudentSelectionOptimizer

            optimizer = StudentSelectionOptimizer(
                teacher=teacher,
                device=device,
                alpha_functional=weight_alignment,
                alpha_representational=weight_sensitivity,
                alpha_performance=weight_accuracy,
            )

            optimizer.analyze_teacher(val_loader)

            best_student, metrics = optimizer.select_optimal_student(
                student_candidates,
                val_loader,
                use_graph_matching=True,
            )

            logging.info(f"Optimization criteria used: {metrics}")

            return best_student, metrics

        except Exception as e:
            logging.warning(
                f"Warning: Optimization criteria failed, falling back to basic selection: {e}"
            )

    # Fall back to basic selection
    best_score = -float("inf")
    best_student = None
    best_metrics = None

    for student, metadata in student_candidates:
        logging.info(f"Evaluating student candidate: {student}")

        metrics = evaluate_student_candidate(
            teacher, student, val_loader, device, num_batches
        )

        # Composite score
        score = (
            weight_alignment * metrics["capacity_alignment"]
            + weight_sensitivity * metrics["sensitivity_match"]
            + weight_accuracy * metrics["accuracy"]
        )

        metrics["composite_score"] = score
        metrics["metadata"] = metadata

        logging.info(f"Student candidate score: {score}")

        if score > best_score:
            best_score = score
            best_student = student
            best_metrics = metrics

    return best_student, best_metrics
