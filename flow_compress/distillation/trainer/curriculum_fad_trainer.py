"""
Curriculum Flow Alignment trainer.
"""

import logging
from typing import Any, Dict, List, Optional

from flow_compress.distillation.flows.alignment import compute_fad_loss_with_weights, compute_layer_weights
from flow_compress.distillation.flows.divergence import (
    compute_flow_statistics,
    compute_layerwise_flow_divergence,
)
from flow_compress.distillation.losses.fad_loss import FADLoss
from flow_compress.distillation.losses.kd_logits import LogitsKDLoss
from flow_compress.utils.depth_mapping import normalize_depth_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CurriculumFADTrainer:
    """
    Curriculum Flow Alignment.
    """

    def __init__(
        self,
        teacher,
        student,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        temperature: float = 4.0,
        num_stages: int = 4,
        gamma_0: float = 1.0,
        lambda_weight: float = 0.1,
        fine_tune_epochs: int = 5,
        steps_per_stage: Optional[int] = None,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_stages = num_stages
        self.gamma_0 = gamma_0
        self.lambda_weight = lambda_weight
        self.fine_tune_epochs = fine_tune_epochs
        self.steps_per_stage = steps_per_stage

        self.ce_loss = nn.CrossEntropyLoss()
        self.logits_kd = LogitsKDLoss(temperature=temperature)

        # Freeze teacher parameters
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.student_depth_mapping = self._compute_normalized_depth_mapping()
        logging.info(
            f"Computed normalized depth mapping for {len(self.student_depth_mapping)} layers"
        )

    def _compute_normalized_depth_mapping(self) -> Dict[str, float]:
        """
        Precompute normalized depth mapping s_S(l_S) for each layer.
        """

        student_layers = sorted(self.student.layer_names)
        num_layers = len(student_layers)

        # Compute normalized depth indices
        depth_indices = normalize_depth_indices(num_layers)

        # Map layer names to normalized depths
        depth_mapping = {}
        for i, layer_name in enumerate(student_layers):
            depth_mapping[layer_name] = depth_indices[i]

        return depth_mapping

    def _compute_alignment_mask(self, stage: int) -> Dict[str, float]:
        """
        Compute alignment mask M_k(l) for stage k.
        """

        # Convert to 0-indexed
        k = stage - 1

        interval_start = k / self.num_stages
        interval_end = (k + 1) / self.num_stages

        if k == self.num_stages - 1:
            interval_end = 1.0

        mask = {}
        for layer_name, normalized_depth in self.student_depth_mapping.items():
            if interval_start <= normalized_depth <= interval_end:
                mask[layer_name] = 1.0
            else:
                mask[layer_name] = 0.0

        return mask

    def _compute_stage_alignment_weight(self, stage: int) -> float:
        """
        Set stage alignment weight γ_k = γ_0 ⋅ (k/K).
        """

        k = stage
        gamma_k = self.gamma_0 * (k / self.num_stages)
        return gamma_k

    def _compute_masked_fad_loss(
        self,
        teacher_div: Dict[str, torch.Tensor],
        student_div: Dict[str, torch.Tensor],
        alignment_mask: Dict[str, float],
        layer_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute masked FAD loss.
        """

        device = next(iter(teacher_div.values())).device
        l_fad = torch.tensor(0.0, device=device)

        # Get ordered layers
        teacher_layers = sorted(teacher_div.keys())
        student_layers = sorted(student_div.keys())

        # For each layer, compute masked weighted alignment
        for i, t_layer in enumerate(teacher_layers):
            if t_layer not in layer_weights:
                continue

            # Find corresponding student layer
            if i < len(student_layers):
                s_layer = student_layers[i]
            else:
                s_layer = student_layers[-1]

            if s_layer not in student_div:
                continue

            # Get mask value for student layer
            mask_value = alignment_mask.get(s_layer, 0.0)
            if mask_value == 0.0:
                continue  # Skip layers not in current stage interval

            # Compute D_align for this layer pair
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

            # Weighted and masked contribution
            w_l = layer_weights[t_layer]
            l_fad = l_fad + mask_value * w_l * dalign

        return l_fad

    def train_stage(self, stage: int) -> Dict[str, float]:
        """
        Train one curriculum stage.
        """

        self.student.train()

        alignment_mask = self._compute_alignment_mask(stage)
        num_active_layers = sum(1 for v in alignment_mask.values() if v > 0)
        logging.info(
            f"Stage {stage}/{self.num_stages}: "
            f"Depth interval [{(stage-1)/self.num_stages:.2f}, {stage/self.num_stages:.2f}], "
            f"{num_active_layers} active layers"
        )

        gamma_k = self._compute_stage_alignment_weight(stage)

        metrics_agg = {
            "loss": 0.0,
            "L_task": 0.0,
            "L_kd": 0.0,
            "L_fad": 0.0,
            "gamma_k": gamma_k,
        }
        num_batches = 0

        # Determine number of steps for this stage
        if self.steps_per_stage is not None:
            max_steps = self.steps_per_stage
        else:
            max_steps = len(self.train_loader)

        step_count = 0
        for batch in self.train_loader:
            if step_count >= max_steps:
                break

            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = None
            x = x.to(self.device)
            if y is not None:
                y = y.to(self.device)

            # Forward teacher
            with torch.no_grad():
                t_logits, t_div = self.teacher.forward_with_flows(x)

            # Forward student
            s_logits, s_div = self.student.forward_with_flows(x)

            # Compute layer weights
            layer_weights = compute_layer_weights(t_div, self.lambda_weight)

            l_fad = self._compute_masked_fad_loss(
                t_div, s_div, alignment_mask, layer_weights
            )

            # Compute task loss
            if y is not None:
                l_task = self.ce_loss(s_logits, y)
            else:
                l_task = torch.tensor(
                    0.0, device=self.device, requires_grad=True)

            # Compute logit-KD loss
            l_kd = self.logits_kd(s_logits, t_logits)

            # Using fixed α and β, with stage-dependent γ_k
            alpha = 1.0
            beta = 1.0
            l_total = alpha * l_task + beta * l_kd + gamma_k * l_fad

            self.optimizer.zero_grad()
            l_total.backward()
            self.optimizer.step()

            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_task"] += l_task.item()
            metrics_agg["L_kd"] += l_kd.item()
            metrics_agg["L_fad"] += l_fad.item()
            num_batches += 1
            step_count += 1

        # Average metrics
        for k in metrics_agg:
            if k != "gamma_k":
                metrics_agg[k] /= max(1, num_batches)

        return metrics_agg

    def fine_tune(self) -> Dict[str, float]:
        """
        Fine-tune all student parameters with full L_FAD.

        Returns:
            Dictionary with fine-tuning metrics
        """
        logging.info(f"Fine-tuning for {self.fine_tune_epochs} epochs...")
        self.student.train()

        metrics_agg = {
            "loss": 0.0,
            "L_task": 0.0,
            "L_kd": 0.0,
            "L_fad": 0.0,
        }
        total_batches = 0

        for epoch in range(self.fine_tune_epochs):
            epoch_metrics = {"loss": 0.0, "L_task": 0.0,
                             "L_kd": 0.0, "L_fad": 0.0}
            num_batches = 0

            for batch in self.train_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = None
                x = x.to(self.device)
                if y is not None:
                    y = y.to(self.device)

                # Forward passes
                with torch.no_grad():
                    t_logits, t_div = self.teacher.forward_with_flows(x)

                s_logits, s_div = self.student.forward_with_flows(x)

                # Compute losses
                if y is not None:
                    l_task = self.ce_loss(s_logits, y)
                else:
                    l_task = torch.tensor(
                        0.0, device=self.device, requires_grad=True)

                l_kd = self.logits_kd(s_logits, t_logits)

                # Full FAD loss (no masking)
                layer_weights = compute_layer_weights(
                    t_div, self.lambda_weight)
                l_fad = compute_fad_loss_with_weights(
                    t_div, s_div, layer_weights)

                # Total loss with full weight
                alpha = 1.0
                beta = 1.0
                gamma = self.gamma_0  # Use full weight for fine-tuning
                l_total = alpha * l_task + beta * l_kd + gamma * l_fad

                # Update
                self.optimizer.zero_grad()
                l_total.backward()
                self.optimizer.step()

                # Aggregate
                epoch_metrics["loss"] += l_total.item()
                epoch_metrics["L_task"] += l_task.item()
                epoch_metrics["L_kd"] += l_kd.item()
                epoch_metrics["L_fad"] += l_fad.item()
                num_batches += 1

            # Average epoch metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= max(1, num_batches)

            # Add to total
            for k in metrics_agg:
                metrics_agg[k] += epoch_metrics[k]
            total_batches += 1

            logging.info(
                f"Fine-tuning epoch {epoch+1}/{self.fine_tune_epochs}: "
                f"Loss={epoch_metrics['loss']:.4f}, "
                f"L_task={epoch_metrics['L_task']:.4f}, "
                f"L_fad={epoch_metrics['L_fad']:.4f}"
            )

        # Average over all epochs
        for k in metrics_agg:
            metrics_agg[k] /= max(1, total_batches)

        return metrics_agg

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluation of student accuracy on validation set.
        """

        self.student.eval()
        total = 0
        correct = 0

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = None

            x = x.to(self.device)
            if y is not None:
                y = y.to(self.device)

            logits = self.student(x)
            if y is not None:
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / max(1, total) if total > 0 else 0.0
        return {"val_acc": acc}

    def train(self) -> Dict[str, Any]:
        """
        Complete curriculum flow alignment training.
        """

        history = {
            "stage_metrics": [],
            "fine_tune_metrics": None,
            "final_val_acc": None,
        }

        logging.info(
            f"Starting Curriculum Flow Alignment training with {self.num_stages} stages..."
        )

        for stage in range(1, self.num_stages + 1):
            stage_metrics = self.train_stage(stage)
            history["stage_metrics"].append(stage_metrics)

            # Evaluate after each stage
            val_metrics = self.evaluate()
            logging.info(
                f"Stage {stage}/{self.num_stages} completed: "
                f"Train Loss={stage_metrics['loss']:.4f}, "
                f"L_FAD={stage_metrics['L_fad']:.4f}, "
                f"Val Acc={val_metrics['val_acc']:.4f}"
            )

        # Fine-tuning
        fine_tune_metrics = self.fine_tune()
        history["fine_tune_metrics"] = fine_tune_metrics

        # Final evaluation
        final_val_metrics = self.evaluate()
        history["final_val_acc"] = final_val_metrics["val_acc"]

        logging.info(
            f"Curriculum training completed. Final Val Acc: {final_val_metrics['val_acc']:.4f}"
        )

        # Return optimized student
        return history
