"""
FAD trainer for flow-based distillation.
"""

import logging
from typing import Any, Dict, Optional

from tqdm import tqdm

from flow_compress.distillation.flows.alignment import (
    compute_fad_loss_integrated,
    compute_fad_loss_with_weights,
    compute_layer_weights,
)
from flow_compress.distillation.flows.divergence import (
    compute_flow_statistics,
    compute_layerwise_flow_divergence,
)
from flow_compress.distillation.flows.selective import selective_flow_alignment
from flow_compress.distillation.losses.fad_loss import FADLoss
from flow_compress.distillation.losses.kd_logits import LogitsKDLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FADTrainer:
    """
    Flow-Aligned Knowledge Distillation.
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
        lambda_weight: float = 0.1,
        use_weights: bool = True,
        precompute_teacher_divs: bool = True,
        use_selective_alignment: bool = False,
        selective_error_threshold: float = 0.1,
        selective_update_freq: int = 10,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.lambda_weight = lambda_weight
        self.use_weights = use_weights
        self.use_selective_alignment = use_selective_alignment
        self.selective_error_threshold = selective_error_threshold
        self.selective_update_freq = selective_update_freq

        self.ce_loss = nn.CrossEntropyLoss()
        self.logits_kd = LogitsKDLoss(temperature=temperature)
        self.fad_loss = FADLoss(alpha_flow=1.0)

        # Freeze teacher parameters
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Precompute teacher flow divergences on validation subset
        self.precomputed_teacher_stats = None
        if precompute_teacher_divs:
            self._precompute_teacher_flows()

        # Selective alignment mask (updated periodically)
        self.selective_mask = None
        self.batch_count = 0

    def _precompute_teacher_flows(self):
        """
        Precompute teacher flow divergences on validation subset.
        """

        logging.info(
            "Precomputing teacher flow divergences on validation subset...")
        self.teacher.eval()
        all_divs = []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    x, _ = batch
                else:
                    x = batch
                x = x.to(self.device)

                self.teacher.activation_hook.clear()
                _ = self.teacher.backbone(x)
                divs = compute_layerwise_flow_divergence(
                    self.teacher.activation_hook.activations,
                    self.teacher.layer_names,
                )
                all_divs.append(divs)

        self.precomputed_teacher_stats = compute_flow_statistics(all_divs)
        logging.info("Teacher flow divergences precomputed.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch of FAD training.
        """

        self.student.train()
        # Reset batch count for selective alignment at start of epoch
        if self.use_selective_alignment:
            self.batch_count = 0
        metrics_agg = {
            "loss": 0.0,
            "L_task": 0.0,
            "L_kd": 0.0,
            "L_fad": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            # Get batch (X, y)
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

            # L_task
            if y is not None:
                l_task = self.ce_loss(s_logits, y)
            else:
                # If no labels, use a dummy task loss (shouldn't happen in practice)
                l_task = torch.tensor(
                    0.0, device=self.device, requires_grad=True)

            # L_logit-KD
            l_kd = self.logits_kd(s_logits, t_logits)

            # Selective alignment: Update mask periodically
            if self.use_selective_alignment:
                if self.batch_count % self.selective_update_freq == 0:
                    with torch.no_grad():
                        self.selective_mask, _, stats = selective_flow_alignment(
                            teacher_div=t_div,
                            student_div=s_div,
                            error_threshold=self.selective_error_threshold,
                            use_critical_points=True,
                        )
                        if self.batch_count == 0:
                            logging.info(
                                f"Selective alignment initialized: "
                                f"{stats['selected_layers_count']}/{stats['total_layers_count']} layers selected"
                            )
            self.batch_count += 1

            # Compute weights w_l
            if self.use_weights:
                layer_weights = compute_layer_weights(
                    t_div, self.lambda_weight)
                # L_FAD with weights
                selective_mask = (
                    self.selective_mask if self.use_selective_alignment else None
                )
                l_fad = compute_fad_loss_with_weights(
                    t_div, s_div, layer_weights, selective_mask=selective_mask
                )
                # Apply base multiplier
                l_fad = self.fad_loss.alpha_flow * l_fad

                # Compute adaptive weights and L_total manually
                with torch.no_grad():
                    # Compute mean divergences
                    d_bar_T = torch.stack(list(t_div.values())).mean()
                    d_bar_S = torch.stack(list(s_div.values())).mean()

                    # Compute adaptive weights α, β, γ
                    eps = 1e-8
                    d_sum = d_bar_T + d_bar_S + eps
                    d_s_over_sum = d_bar_S / d_sum
                    d_t_over_sum = d_bar_T / d_sum
                    Z = 1.0 + d_s_over_sum + d_t_over_sum
                    alpha = 1.0 / Z
                    beta = d_s_over_sum / Z
                    gamma = d_t_over_sum / Z

                # L_total = α ⋅ L_task + β ⋅ L_logit-KD + γ ⋅ L_FAD
                l_total = alpha * l_task + beta * l_kd + gamma * l_fad

                info = {
                    "L_task": l_task.detach(),
                    "L_kd": l_kd.detach(),
                    "L_fad": l_fad.detach(),
                    "alpha": alpha.detach(),
                    "beta": beta.detach(),
                    "gamma": gamma.detach(),
                    "lambda_task": alpha.detach(),
                    "lambda_kd": beta.detach(),
                    "lambda_fad": gamma.detach(),
                }
            else:
                # Use FADLoss with integrated FAD
                l_total, info = self.fad_loss(
                    teacher_div=t_div,
                    student_div=s_div,
                    base_task_loss=l_task,
                    base_kd_loss=l_kd,
                )

            # Backward pass and parameter update
            self.optimizer.zero_grad()
            l_total.backward()
            self.optimizer.step()

            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_task"] += info["L_task"].item()
            metrics_agg["L_kd"] += info["L_kd"].item()
            metrics_agg["L_fad"] += info["L_fad"].item()
            metrics_agg["alpha"] += info.get(
                "alpha", info.get("lambda_task", 0.0)
            ).item()
            metrics_agg["beta"] += info.get("beta",
                                            info.get("lambda_kd", 0.0)).item()
            metrics_agg["gamma"] += info.get(
                "gamma", info.get("lambda_fad", 0.0)
            ).item()
            num_batches += 1

        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= max(1, num_batches)
        return metrics_agg

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Complete training loop.
        """

        history = {
            "train_metrics": [],
            "val_metrics": [],
        }

        logging.info(f"Starting FAD training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            # Training epoch
            train_metrics = self.train_epoch(epoch)
            history["train_metrics"].append(train_metrics)

            # Evaluation
            val_metrics = self.evaluate()
            history["val_metrics"].append(val_metrics)

            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.4f}"
            )

        # Return trained student
        logging.info("Training completed.")
        return history

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluation of student accuracy on validation (without FAD losses).
        """

        self.student.eval()
        total = 0
        correct = 0

        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.student(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / max(1, total)
        return {"val_acc": acc}

    def distill(
        self,
        student_model,
        teacher_model,
        dummy_input,
        train_loader,
        test_loader,
        performance_metric_function,
        initial_accuracy_top1,
        max_performance_metric_degradation_th,
        device,
        writer=None,
    ):
        """
        Perform knowledge distillation between teacher and student models.
        """

        # Select ReLU layers for feature distillation
        layers_to_distill = [
            name
            for name, module in student_model.named_modules()
            if isinstance(module, nn.ReLU)
        ]

        # Select first, middle, and last ReLU layers for efficiency
        layers_to_distill = [
            layers_to_distill[0],
            layers_to_distill[len(layers_to_distill) // 2],
            layers_to_distill[-1],
        ]

        # Initialize feature distiller
        distiller = FADFeatureDistiller(
            student_model,
            teacher_model,
            layers_for_feature_distillation=layers_to_distill,
        )

        # Extract initial features with dummy input
        with torch.no_grad():
            _ = teacher_model(dummy_input)
            _ = student_model(dummy_input)

        # Initialize alignment convolutions
        distiller._init_align_convs()

        # Initialize optimizer for both model and alignment convolutions
        optimizer = torch.optim.AdamW(
            list(student_model.parameters()) +
            list(distiller.align_convs.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )

        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.90, patience=20
        )

        # Initialize loss functions
        ce_loss_fn = nn.CrossEntropyLoss()

        # Initialize progress tracking
        progress_bar = tqdm()
        prev_loss = 0.0
        running_loss = 0.0
        it = 0

        # Main training loop
        for epoch in range(100):
            teacher_model.eval()
            student_model.train()

            # Training iterations
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # Extract teacher features
                with torch.no_grad():
                    _ = teacher_model(x)

                # Forward pass through student model
                student_outputs = student_model(x)

                # Compute combined loss
                ce_loss = ce_loss_fn(student_outputs, y)
                feat_loss = distiller.compute_feature_loss()
                total_loss = ce_loss + 0.5 * feat_loss
                total_loss.backward()

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

                # Track loss statistics
                running_loss += total_loss.item()
                avg_loss = running_loss / (it + 1)
                it += 1
                loss_difference = abs(avg_loss - prev_loss)

                # Log metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar(
                        "Distillation/Total_Loss", total_loss.item(), it)
                    writer.add_scalar("Distillation/CE_Loss",
                                      ce_loss.item(), it)
                    writer.add_scalar(
                        "Distillation/Feature_Loss", feat_loss.item(), it)
                    writer.add_scalar(
                        "Distillation/Learning_Rate",
                        optimizer.param_groups[0]["lr"],
                        it,
                    )

                # Check for convergence
                if (i > 100 or epoch > 0) and loss_difference < 1e-4:
                    distiller._remove_hooks(student_model)
                    logging.debug(
                        "Loss converged, stopping. %f, %f, %f",
                        loss_difference,
                        avg_loss,
                        prev_loss,
                    )
                    break

                prev_loss = avg_loss

                # Update progress information
                logging_text = f"Epoch: {epoch + 1} - Iter: {i + 1} - Loss: {avg_loss:.7f} - Loss Difference: {loss_difference:.7f} - lr: {optimizer.param_groups[0]['lr']:.7f}"
                logging.debug(logging_text)
                progress_bar.set_description(logging_text)
                progress_bar.update(1)

                # Update learning rate
                scheduler.step(avg_loss)

            # Evaluate current performance
            current_accuracy_top1 = performance_metric_function(
                student_model, test_loader, device, k=1
            )

            # Log epoch metrics
            if writer is not None:
                writer.add_scalar("Distillation/Accuracy",
                                  current_accuracy_top1, epoch)
                writer.add_scalar("Distillation/Avg_Loss", avg_loss, epoch)

            logging.debug(
                f"Epoch {epoch + 1} completed. Avg loss: {avg_loss:.6f}")
            logging.debug(f"Current TOP-1 accuracy: {current_accuracy_top1}%")

            # Check if performance degradation is within acceptable range
            if (
                initial_accuracy_top1 - current_accuracy_top1
                <= max_performance_metric_degradation_th
            ):
                distiller._remove_hooks(student_model)
                break

            # Check for convergence
            if loss_difference < 1e-4:
                break


class FADFeatureDistiller:
    """
    Helper class for FAD feature distillation.
    """

    def __init__(
        self, student, teacher, layers_for_feature_distillation, device="cuda"
    ):
        """
        Initialize the FAD feature distiller.
        """

        # Define models and move to device
        self.teacher = teacher.to(device)
        self.student = student.to(device)

        self.layers_for_feature_distillation = layers_for_feature_distillation
        # Dictionary to store alignment convolutions for feature dimension matching
        self.align_convs = nn.ModuleDict()

        # Buffers for extracted features from both models
        self.student_feats = {}
        self.teacher_feats = {}

        # Register hooks to capture features during forward pass
        self._register_hooks(self.student, self.student_feats, "student")
        self._register_hooks(self.teacher, self.teacher_feats, "teacher")

    def _register_hooks(self, model, feat_dict, model_type):
        """
        Register forward hooks to capture features from specified layers.
        """

        for name, module in model.named_modules():
            if name in self.layers_to_distill:
                module.register_forward_hook(self._get_hook(name, feat_dict))

    def _remove_hooks(self, model):
        """Remove all forward hooks from the model.

        Args:
            model: Model to remove hooks from
        """

        for module in model.modules():
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()

    def _get_hook(self, name, feature_dict):
        """
        Create a hook function to capture layer outputs.
        """

        def hook(module, input, output):
            feature_dict[name] = output

        return hook

    def _init_align_convs(self):
        """
        Initialize alignment convolutions for feature dimension matching.

        Creates 1x1 convolutions to match feature dimensions between teacher and student
        models when they don't match exactly.
        """

        for name in self.layers_to_distill:
            s_feat = self.student_features.get(name)
            t_feat = self.teacher_features.get(name)

            in_channels = s_feat.shape[1]
            out_channels = t_feat.shape[1]

            if in_channels != out_channels:
                key = name.replace(".", "__")
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                self.align_convs[key] = conv.to(s_feat.device)

    def compute_feature_loss(self):
        """
        Compute feature distillation loss between teacher and student models.
        """

        loss = 0.0
        for name in self.layers_to_distill:
            f_s = self.student_features.get(name)
            f_t = self.teacher_features.get(name)

            if f_s is None or f_t is None:
                continue

            # Apply alignment convolution if needed
            key = name.replace(".", "__")
            if key in self.align_convs:
                f_s = self.align_convs[key](f_s)

            if f_s.shape != f_t.shape:
                continue

            # Compute L1 loss between features
            loss += F.l1_loss(f_s, f_t)

        return loss
